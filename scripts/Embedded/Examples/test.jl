using Gridap,GridapTopOpt
using Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters

using GridapTopOpt: StateParamIntegrandWithMeasure

n = 51
order = 1

_model = CartesianDiscreteModel((0,1,0,1),(n,n))
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = ref_model.model
el_Δ = get_el_Δ(_model)
h = maximum(el_Δ)
h_refine = maximum(el_Δ)/2
f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= 0.2 + eps() || x[2] >= 0.8 - eps()))
f_Γ_N(x) = (x[1] ≈ 1 && 0.4 - eps() <= x[2] <= 0.6 + eps())
update_labels!(1,model,f_Γ_D,"Gamma_D")
update_labels!(2,model,f_Γ_N,"Gamma_N")

## Triangulations and measures
Ω = Triangulation(model)
Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
dΩ = Measure(Ω,2*order)
dΓ_N = Measure(Γ_N,2*order)
vol_D = sum(∫(1)dΩ)

## Levet-set function space and derivative regularisation space
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
U_reg = TrialFESpace(V_reg,0)
V_φ = TestFESpace(model,reffe_scalar)

## Levet-set function
φh = interpolate(x->-cos(4π*x[1])*cos(4π*x[2])-0.2,V_φ)
geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

x_φ = get_free_dof_values(φh)
idx = findall(isapprox(0.0;atol=10^-10),x_φ)
@assert isempty(idx)

Ωin = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL))
Γg = DifferentiableTriangulation(GhostSkeleton(cutgeo))
n_Γg = get_normal_vector(Γg)

dΩin = Measure(Ωin,2*order)
dΓg  = Measure(Γg,2*order)
n_Γg = get_normal_vector(Γg)

a(u,v,φ) = ∫(∇(v)⋅∇(u))dΩin #+ ∫((γg*h)*jump(n_Γg⋅∇(v))*jump(n_Γg⋅∇(u)))dΓg
l(v,φ) = ∫(v)dΓ_N

Ωact = Triangulation(cutgeo,ACTIVE)
V = TestFESpace(Ωact,reffe_scalar;dirichlet_tags=["Gamma_D"])
U = TrialFESpace(V,0.0)

uhd = zero(U)
# uhd = interpolate(x->1,U)

# ∇(a,[uhd,uhd,φh],3)
∇(φ->a(uhd,uhd,φ),φh)


function generate_model(D,n)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  base_model = UnstructuredDiscreteModel((CartesianDiscreteModel(domain,cell_partition)))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = ref_model.model
  return model
end
function level_set(shape::Symbol;N=4)
  if shape == :square
    x -> max(abs(x[1]-0.5),abs(x[2]-0.5))-0.25 # Square
  elseif shape == :corner_2d
    x -> ((x[1]-0.5)^N+(x[2]-0.5)^N)^(1/N)-0.25 # Curved corner
  elseif shape == :diamond
    x -> abs(x[1]-0.5)+abs(x[2]-0.5)-0.25-0/n/10 # Diamond
  elseif shape == :circle
    x -> sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-0.5223 # Circle
  elseif shape == :square_prism
    x -> max(abs(x[1]-0.5),abs(x[2]-0.5),abs(x[3]-0.5))-0.25 # Square prism
  elseif shape == :corner_3d
    x -> ((x[1]-0.5)^N+(x[2]-0.5)^N+(x[3]-0.5)^N)^(1/N)-0.25 # Curved corner
  elseif shape == :diamond_prism
    x -> abs(x[1]-0.5)+abs(x[2]-0.5)+abs(x[3]-0.5)-0.25-0/n/10 # Diamond prism
  elseif shape == :sphere
    x -> sqrt((x[1]-0.5)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-0.53 # Sphere
  elseif shape == :regular_2d
    x -> cos(2π*x[1])*cos(2π*x[2])-0.11 # "Regular" LSF
  elseif shape == :regular_3d
    x -> cos(2π*x[1])*cos(2π*x[2])*cos(2π*x[3])-0.11 # "Regular" LSF
  end
end

  
D = 2
n = 10
model = generate_model(D,n)
φ = level_set(:circle)
f = x -> 1.0
order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe)

φh = interpolate(φ,V_φ)
fh = interpolate(f,V_φ)

# Correction if level set is on top of a node
x_φ = get_free_dof_values(φh)
idx = findall(isapprox(0.0;atol=10^-10),x_φ)
!isempty(idx) && @info "Correcting level values!"
x_φ[idx] .+= 10*eps(eltype(x_φ))

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

# A.1) Volume integral

Ω = Triangulation(cutgeo,PHYSICAL_IN)
Ω_AD = DifferentiableTriangulation(Ω)
dΩ = Measure(Ω_AD,2*order)

Γ = EmbeddedBoundary(cutgeo)
dΓ = Measure(Γ,2*order)

J_bulk(φ) = ∫(fh)dΩ
dJ_bulk_AD = gradient(J_bulk,φh)
dJ_bulk_AD_vec = assemble_vector(dJ_bulk_AD,V_φ)

dJ_bulk_exact(q) = ∫(-fh*q/(norm ∘ (∇(φh))))dΓ
dJ_bulk_exact_vec = assemble_vector(dJ_bulk_exact,V_φ)

@test norm(dJ_bulk_AD_vec - dJ_bulk_exact_vec) < 1e-10

# A.2) Volume integral

g(fh) = ∇(fh)⋅∇(fh)
J_bulk(φ) = ∫(g(fh))dΩ
dJ_bulk_AD = gradient(J_bulk,φh)
dJ_bulk_AD_vec = assemble_vector(dJ_bulk_AD,V_φ)

dJ_bulk_exact(q) = ∫(-g(fh)*q/(norm ∘ (∇(φh))))dΓ
dJ_bulk_exact_vec = assemble_vector(dJ_bulk_exact,V_φ)

@test norm(dJ_bulk_AD_vec - dJ_bulk_exact_vec) < 1e-10