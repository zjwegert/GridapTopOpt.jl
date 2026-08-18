using Gridap, Gridap.Geometry, Gridap.Fields, GridapGmsh
using GridapTopOpt
using Test

# Density-based topology optimisation for electromagnetic scattering 
# Based on the tutorial: https://gridap.github.io/Tutorials/dev/pages/t019_TopOptEMFocus/
# 
# We recommend that you at least read up to "Weak form" in the tutorial before running this script.
# The main difference between this script and the tutorial is that here we use GridapTopOpt's 
# automatic differentiation capabilities to compute the gradient of the objective function.
#
# Note that this script also requires NLopt, CairoMakie, and GridapMakie.

# Setup
λ = 532      # Wavelength (nm)
L = 600      # Width of the numerical cell (excluding PML) (nm)
h1 = 600     # Height of the air region below the source (nm)
h2 = 200     # Height of the air region above the source (nm)
dpml = 300   # Thickness of the PML (nm)

n_metal = 0.054 + 3.429im # Silver refractive index at λ = 532 nm
n_air = 1    # Air refractive index
μ = 1        # Magnetic permeability
k = 2*π/λ    # Wavenumber (nm^-1)

# Discrete Model
msh_file = "./scripts/Examples/Density/Meshes/RecCirGeometry.msh"
model = GmshDiscreteModel(msh_file)

# FE spaces for the magnetic field
order = 1
reffe = ReferenceFE(lagrangian, Float64, order)
V = TestFESpace(model, reffe, dirichlet_tags = ["DirichletEdges", "DirichletNodes"], 
    vector_type = Vector{ComplexF64})

# Numerical integration
degree = 2
Ω = Triangulation(model)
dΩ = Measure(Ω, degree)

Γ_s = BoundaryTriangulation(model; tags = ["Source"]) # Source line
dΓ_s = Measure(Γ_s, degree)

Ω_d = Triangulation(model, tags="Design")
dΩ_d = Measure(Ω_d, degree)

Ω_c = Triangulation(model, tags="Center")
dΩ_c = Measure(Ω_c, degree)

# FE spaces for the design parameters
p_reffe = ReferenceFE(lagrangian, Float64, 0)
Q = TestFESpace(Ω_d, p_reffe, vector_type = Vector{ComplexF64})

pf_reffe = ReferenceFE(lagrangian, Float64, 1)
Qf = TestFESpace(Ω_d, pf_reffe, vector_type = Vector{ComplexF64})

# PML formulation
R = 1e-10
LHp=(L/2, h1+h2)   # Start of PML for x,y > 0
LHn=(L/2, 0)       # Start of PML for x,y < 0

function s_PML(x)
    σ = -3 / 4 * log(R) / dpml / n_air
    xf = Tuple(x)
    u = @. ifelse(xf > 0 , xf - LHp, - xf - LHn)
    return @. ifelse(u > 0,  1 + (1im * σ / k) * (u / dpml)^2, $(1.0+0im))
end

function ds_PML(x)
    σ = -3 / 4 * log(R) / dpml / n_air
    xf = Tuple(x)
    u = @. ifelse(xf > 0 , xf - LHp, - xf - LHn)
    ds = @. ifelse(u > 0, (2im * σ / k) * (1 / dpml)^2 * u, $(0.0+0im))
    return ds.*sign.(xf)
end

function Λ(x)
    s_x,s_y = s_PML(x)
    return VectorValue(1/s_x, 1/s_y)
end

Fields.∇(::typeof(Λ)) = x -> TensorValue{2, 2, ComplexF64}(-(Λ(x)[1])^2 * ds_PML(x)[1], 0, 0, -(Λ(x)[2])^2 * ds_PML(x)[2])

# Filter and threshold
r = 5/sqrt(3)               # Filter radius
β = Ref(32.0)               # β∈[1,∞], threshold sharpness
η = 0.5                     # η∈[0,1], threshold center

a_f(r, u, v) = r^2 * (∇(v) ⋅ ∇(u))

p_to_pt = AffineFEStateMap((u,v,p)->∫(a_f(r, u, v))dΩ_d + ∫(v * u)dΩ_d, (v,p)->∫(v * p)dΩ_d, Qf, Qf, Q)

function Threshold(pfh)
    _β = β[]
    return ((tanh(_β * η) + tanh(_β * (pfh - η))) / (tanh(_β * η) + tanh(_β * (1.0 - η))))
end

# Weak form
ξd(p) = 1 / (n_air + (n_metal - n_air) * p)^2 - 1 / n_air^2 # in the design region

a_base(u, v) = (1 / n_air^2) * ((∇ .* (Λ * v)) ⊙ (Λ .* ∇(u))) - (k^2 * μ * (v * u))
a_design(u, v, pt) = (ξd ∘ pt) * (∇(v) ⊙ ∇(u))

a(u,v,pt) = ∫(a_base(u, v))dΩ + ∫(a_design(u, v, Threshold ∘ (real ∘ pt)))dΩ_d
l(v,pt) = ∫(v)dΓ_s
pt_to_u = AffineFEStateMap(a, l, V, V, Qf)

# Objective
x0 = VectorValue(0,300)  # Position of the field to be optimized
δ = 1
G(x) = (1/(2*π)*exp(-norm(x - x0)^2 / 2 / δ^2))
j(u,pt) = ∫(G * (∇(u)' ⋅ ∇(u)))dΩ_c
J = StateParamMap(j, pt_to_u)

# Design variable to objective map
function p_to_j(p)
  pt = p_to_pt(real(p).+0im)
  u = pt_to_u(real(pt).+0im)
  return real(J(u,pt))
end

# Plane wave incident example
ph = zero(Q)
pt = p_to_pt(ph)
u = pt_to_u(pt)
uh = FEFunction(V,u)
writevtk(Ω, "results/uh", cellfields = ["real(uh)" => real ∘ uh, "imag(uh)" => imag ∘ uh])
p_to_j(get_free_dof_values(ph)) # j

# Verify AD
using Random
np = num_free_dofs(Q)
_p0 = rand(MersenneTwister(1), np) .+ 0im
_δp = rand(MersenneTwister(2), np)*1e-8 .+ 0im
dj_δp = p_to_j(_p0+_δp)-p_to_j(_p0)
dj_ad = real(GridapTopOpt.val_and_gradient(p_to_j, _p0)[2][1])
dj_δp_ad = real(dj_ad'*_δp) # directional derivative
@test dj_δp ≈ dj_δp_ad rtol = 1e-6

# Optimization with NLopt
using NLopt

mkpath("results/output_path")
it = Ref(0)

function gf_p(p0::Vector, grad::Vector)
    j,dj = GridapTopOpt.val_and_gradient(p_to_j, p0)
    if length(grad) > 0
        grad[:] = real(first(dj))
    end
    it[] += 1
    println("Iteration $(it[]): j = $(j[])")
    open("results/output_path/gvalue.txt", "a") do io
        write(io, "$j \n")
    end
    j
end

function gf_p_optimize(p_init; TOL = 1e-8, MAX_ITER = 100)
    ##################### Optimize #################
    opt = Opt(:LD_MMA, np)
    opt.lower_bounds = 0
    opt.upper_bounds = 1
    opt.ftol_rel = TOL
    opt.maxeval = MAX_ITER
    opt.max_objective = gf_p

    (g_opt, p_opt, ret) = optimize(opt, p_init)
    @show numevals = opt.numevals # the number of function evaluations
    return g_opt, p_opt
end

p_opt = fill(0.4, np)   # Initial guess
β_list = [8.0, 16.0, 32.0]

g_opt = Ref(0.0)
for bi = 1 : 3
    β[] = β_list[bi]
    _g_opt, p_temp_opt = gf_p_optimize(p_opt)
    g_opt[] = _g_opt
    copyto!(p_opt, p_temp_opt)
end
@show g_opt[]

# Results and plot
using CairoMakie, GridapMakie

pt = p_to_pt(p_opt .+ 0im)
u = pt_to_u(pt)
uh = FEFunction(V,u)
pfh = FEFunction(Qf, pt)
pth = Threshold ∘ pfh

fig, ax, plt = plot(Ω, real ∘ pth, colormap = :binary)
Colorbar(fig[1,2], plt)
ax.aspect = AxisAspect(1)
ax.title = "Design Shape"
rplot = 110 # Region for plot
limits!(ax, -rplot, rplot, (h1)/2-rplot, (h1)/2+rplot)
CairoMakie.save("results/output_path/shape.png", fig)

maxe = 30 # Maximum electric field magnitude compared to the incident plane wave
e1=abs2(n_air^2)
e2=abs2(n_metal^2)

fig, ax, plt = plot(Ω, 2*(sqrt∘(abs((conj(∇(uh)) ⋅ ∇(uh))/(CellField(e1,Ω) + (e2 - e1) * pth)))), colormap = :hot, colorrange=(0, maxe))
Colorbar(fig[1,2], plt)
ax.title = "|E|"
ax.aspect = AxisAspect(1)
limits!(ax, -rplot, rplot, (h1)/2-rplot, (h1)/2+rplot)
CairoMakie.save("results/output_path/Field.png", fig)