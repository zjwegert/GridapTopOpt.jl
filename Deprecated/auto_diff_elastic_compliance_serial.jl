using Gridap, Gridap.TensorValues, Gridap.Geometry, Gridap.FESpaces
using PartitionedArrays

using ChainRulesCore
using Zygote
include("ChainRules.jl");

# Heaviside function
function H_η(t;η)
    M = typeof(η*t)
    if t<-η
        return zero(M)
    elseif abs(t)<=η
        return M(1/2*(1+t/η+1/pi*sin(pi*t/η)))
    elseif t>η
        return one(M)
    end
end

function DH_η(t::M;η::M) where M<:AbstractFloat
    if t<-η
        return zero(M)
    elseif abs(t)<=η
        return M(1/2/η*(1+cos(pi*t/η)))
    elseif t>η
        return zero(M)
    end
end

# Material interpolation
Base.@kwdef struct SmoothErsatzMaterialInterpolation{M<:AbstractFloat}
    η::M # Smoothing radius
    ϵₘ::M = 10^-3 # Void material multiplier
    H = x -> H_η(x,η=η)
    DH = x -> DH_η(x,η=η)
    I = φ -> (1 - H(φ)) + ϵₘ*H(φ)
    ρ = φ -> 1 - H(φ)
end

function isotropic_2d(E::M,ν::M) where M<:AbstractFloat
    λ = E*ν/((1+ν)*(1-ν)); μ = E/(2*(1+ν))
    C = [λ+2μ  λ     0
         λ    λ+2μ   0
         0     0     μ];
    SymFourthOrderTensorValue(
        C[1,1], C[3,1], C[2,1],
        C[1,3], C[3,3], C[2,3],
        C[1,2], C[3,2], C[2,2])
end

######################################################
# begin
## FE Setup
order = 1;
el_size = (200,200);
dom = (0.,1.,0.,1.);
model = CartesianDiscreteModel(dom,el_size);
## Define Γ_N and Γ_D
xmax,ymax = dom[2],dom[4]
labels = get_face_labeling(model)
entity = num_entities(labels) + 1
prop_Γ_N = 0.4
prop_Γ_D = 0.2
f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= ymax*prop_Γ_D + eps() || x[2] >= ymax-ymax*prop_Γ_D - eps())) ? true : false;
f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= ymax/2+ymax*prop_Γ_N/4 + eps()) ? true : false;
# Vertices
vtx_coords = model.grid_topology.vertex_coordinates
vtxs_Γ_D = findall(isone,f_Γ_D.(vtx_coords))
vtx_edge_connectivity = Array(model.grid_topology.n_m_to_nface_to_mfaces[1,2][vtxs_Γ_D]);
# Edges
edge_entires = [findall(x->any(x .∈  vtx_edge_connectivity[1:end.!=j]),vtx_edge_connectivity[j]) for j = 1:length(vtx_edge_connectivity)]
edge_Γ_D = unique(reduce(vcat,getindex.(vtx_edge_connectivity,edge_entires),init=[]))
labels.d_to_dface_to_entity[1][vtxs_Γ_D] .= entity
labels.d_to_dface_to_entity[2][edge_Γ_D] .= entity
add_tag!(labels,"Gamma_D",[entity])
# Γ_N
entity = num_entities(labels) + 1
# Vertices
vtxs_Γ_N = findall(isone,f_Γ_N.(vtx_coords))
vtx_edge_connectivity = Array(model.grid_topology.n_m_to_nface_to_mfaces[1,2][vtxs_Γ_N]);
# Edges
edge_entires = [findall(x->any(x .∈  vtx_edge_connectivity[(1:end.!=j)]),vtx_edge_connectivity[j]) for j = 1:length(vtx_edge_connectivity)]
edge_Γ_N = unique(reduce(vcat,getindex.(vtx_edge_connectivity,edge_entires),init=[]))
labels.d_to_dface_to_entity[1][vtxs_Γ_N] .= entity
labels.d_to_dface_to_entity[2][edge_Γ_N] .= entity
add_tag!(labels,"Gamma_N",[entity])
## Triangulations and measures
Ω = Triangulation(model)
Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
dΩ = Measure(Ω,2order)
dΓ_N = Measure(Γ_N,2order)
## Spaces
reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["Gamma_D"],
    dirichlet_masks=[(true,true)],vector_type=Vector{Float64})
U = TrialFESpace(V,[VectorValue(0.0,0.0)])
# Space for shape sensitivities
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_L2 = TestFESpace(model,reffe_scalar;conformity=:L2)
# FE space for LSF 
V_φ = TestFESpace(model,reffe_scalar;conformity=:H1)
# FE Space for shape derivatives
V_reg = TestFESpace(model,reffe_scalar;conformity=:H1,
        dirichlet_tags=["Gamma_N"],dirichlet_masks=[true])
U_reg = TrialFESpace(V_reg,[0.0])
######################################################
eΔ = (xmax,ymax)./el_size;
interp = SmoothErsatzMaterialInterpolation(η = 2*maximum(eΔ))
C = isotropic_2d(1.,0.3)
g = VectorValue(0.,-1.0)
φh = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25,V_φ)

## Weak form
I = interp.I;

function ϕ_to_ϕₕ(ϕ::AbstractArray,Q)
	ϕ = FEFunction(Q,ϕ)
end
function ϕ_to_ϕₕ(ϕ::FEFunction,Q)
	ϕ
end
function ϕ_to_ϕₕ(ϕ::CellField,Q)
	ϕ
end

function a(u,v,φ) 
    φh = ϕ_to_ϕₕ(φ,V_φ)
    ∫((I ∘ φh)*(C ⊙ ε(u) ⊙ ε(v)))dΩ
end
a(φ) = (u,v) -> a(u,v,φ)

l(v,φh) = ∫(v ⋅ g)dΓ_N
l(φ) = v -> l(v,φ)

res(u,v,φ,V_φ) = a(u,v,φ) - l(v,φ)

φ = φh.free_values

## Solve finite element problem
op = AffineFEOperator(a(φ),l(φ),U,V)
K = op.op.matrix;
## Solve
uh = solve(op)
## Compute J and v_J
_J(u,φ) = (a(u,u,φ)) # ∫(interp.ρ ∘ φ)dΩ
# _J(u,φ) = ∫(1+(u⋅u)*(u⋅u)+φ)dΩ # <- weird stuff works here


φ_to_u = AffineFEStateMap(a,l,res,V_φ,U,V)
u_to_j =  LossFunction(_J,V_φ,U)

u, u_pullback   = rrule(φ_to_u,φ)
j, j_pullback   = rrule(u_to_j,u,φ)
_, du, dϕ₍ⱼ₎    = j_pullback(1) # dj = 1
_, dϕ₍ᵤ₎        = u_pullback(du)
   dϕ           = dϕ₍ᵤ₎ + dϕ₍ⱼ₎

function φ_to_j(φ)
    u = φ_to_u(φ)
    j = u_to_j(u,φ)
end

j,dφ = Zygote.withgradient(φ_to_j,φ)

sum(_J(uh,φh))
j

## Shape derivative
# Autodiff
dϕh  = interpolate_everywhere(FEFunction(V_φ,dϕ),U_reg)

# Analytic
J′(v,v_h) = ∫(-v_h*v*(interp.DH ∘ φh)*(norm ∘ ∇(φh)))dΩ;
v_J = -(C ⊙ ε(uh) ⊙ ε(uh))
b = assemble_vector(v->J′(v,v_J),V_reg)
analytic_J′ = FEFunction(V_reg,b)

abs_error = abs(dϕh-analytic_J′)
rel_error = (abs(dϕh-analytic_J′))/abs(analytic_J′)

#############################
## Hilb ext reg
α = 4*maximum(eΔ)
A(u,v) = α^2*∫(∇(u) ⊙ ∇(v))dΩ + ∫(u ⋅ v)dΩ;
hilb_K = assemble_matrix(A,U_reg,V_reg)

## Autodiff result
# -dϕh is AD version of J′ that we plug in usually!
op = AffineFEOperator(U_reg,V_reg,hilb_K,dϕh.free_values)
dϕh_Ω = solve(op)

## Analytic result
b = assemble_vector(v->J′(v,v_J),V_reg)
op = AffineFEOperator(U_reg,V_reg,hilb_K,b)
v_J_Ω = solve(op)

hilb_abs_error = abs(dϕh_Ω-v_J_Ω)
hilb_rel_error = (abs(dϕh_Ω-v_J_Ω))/abs(v_J_Ω)

path = dirname(dirname(@__DIR__))*"/results/AutoDiffTesting";
writevtk(Ω,path,cellfields=["phi"=>φh,
    "H(phi)"=>(interp.H ∘ φh),
    "|nabla(phi)|"=>(norm ∘ ∇(φh)),
    "uh"=>uh,
    "J′_abs_error"=>abs_error,
    "J′_rel_error"=>rel_error,
    "J′_analytic"=>analytic_J′,
    "J′_autodiff"=>dϕh,
    "hilb_abs_error"=>hilb_abs_error,
    "hilb_rel_error"=>hilb_rel_error,
    "v_J_Ω"=>v_J_Ω,
    "dJϕh_Ω"=>dϕh_Ω
])
# end