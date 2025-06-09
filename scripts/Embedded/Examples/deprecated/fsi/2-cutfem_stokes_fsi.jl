using Gridap, Gridap.Geometry, Gridap.Adaptivity
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt

path = "./results/fsi testing/"
mkpath(path)

# Cut the background model
n = 100
partition = (n,n)
D = length(partition)
_model = CartesianDiscreteModel((0,1,0,1),partition)
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = ref_model.model

el_Δ = get_el_Δ(_model)
h = maximum(el_Δ)
f_Γ_D(x) = x[1] ≈ 0
f_Γ_NoSlipTop(x) = x[2] ≈ 1
f_Γ_NoSlipBottom(x) = x[2] ≈ 0
update_labels!(1,model,f_Γ_D,"Gamma_D")
update_labels!(2,model,f_Γ_NoSlipTop,"Gamma_NoSlipTop")
update_labels!(3,model,f_Γ_NoSlipBottom,"Gamma_NoSlipBottom")

# Cut the background model
reffe_scalar = ReferenceFE(lagrangian,Float64,1)
V_φ = TestFESpace(model,reffe_scalar)
φh = interpolate(x->-max(20*abs(x[1]-0.5),3*abs(x[2]-0.2))+1,V_φ)
geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)
cutgeo_facets = cut_facets(model,geo)

# Generate the "active" model
Ω_act = Triangulation(cutgeo,ACTIVE)
Ω_act_solid = Triangulation(cutgeo,ACTIVE_OUT)

# Setup integration meshes
Ω = Triangulation(cutgeo,PHYSICAL)
Ωout = Triangulation(cutgeo,PHYSICAL_OUT)
Γ = EmbeddedBoundary(cutgeo)
Γg = GhostSkeleton(cutgeo)
Γi = SkeletonTriangulation(cutgeo_facets)

# Setup normal vectors
n_Γ = get_normal_vector(Γ)
n_Γg = get_normal_vector(Γg)
n_Γi = get_normal_vector(Γi)

# Setup Lebesgue measures
order = 1
degree = 2*order
dΩ = Measure(Ω,degree)
dΩout = Measure(Ωout,degree)
dΓ = Measure(Γ,degree)
dΓg = Measure(Γg,degree)
dΓi = Measure(Γi,degree)

# Setup FESpace

uin(x) = VectorValue(x[2]*(1-x[2]),0.0)

reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
reffe_p = ReferenceFE(lagrangian,Float64,order,space=:P)
reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

V = TestFESpace(Ω_act,reffe_u,conformity=:H1,dirichlet_tags=["Gamma_D","Gamma_NoSlipTop","Gamma_NoSlipBottom"])
Q = TestFESpace(Ω_act,reffe_p,conformity=:H1)
T = TestFESpace(Ω_act_solid,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_NoSlipBottom"])

U = TrialFESpace(V,[uin,VectorValue(0.0,0.0),VectorValue(0.0,0.0)])
P = TrialFESpace(Q)
R = TrialFESpace(T)

X = MultiFieldFESpace([U,P,R])
Y = MultiFieldFESpace([V,Q,T])

# Weak form
## Fluid
# Properties
Re = 60 # Reynolds number
ρ = 1.0 # Density
L = 1.0 # Characteristic length
u0_max = maximum(abs,get_dirichlet_dof_values(U))
μ = ρ*L*u0_max/Re # Viscosity
# Stabilization parameters
β0 = 0.25
β1 = 0.2
β2 = 0.1*h # 0.05*μ*h
β3 = 0.05*(μ/h + ρ*u0_max/6)^-1*h^2 # 0.05*h^3
γ = 100.0
# Terms
σf_n(u,p) = μ*∇(u)⋅n_Γ - p*n_Γ
a_Ω(u,v) = μ*(∇(u) ⊙ ∇(v))
b_Ω(v,p) = - (∇⋅v)*p
c_Γi(p,q) = (β0*h)*jump(p)*jump(q) # this will vanish for p∈P1
c_Ω(p,q) = (β1*h^2)*∇(p)⋅∇(q)
a_Γ(u,v) = - (n_Γ⋅∇(u))⋅v - u⋅(n_Γ⋅∇(v)) + (γ/h)*u⋅v
b_Γ(v,p) = (n_Γ⋅v)*p
i_Γg(u,v) = β2*jump(n_Γg⋅∇(u))⋅jump(n_Γg⋅∇(v))
j_Γg(p,q) = β3*jump(n_Γg⋅∇(p))*jump(n_Γg⋅∇(q)) + c_Γi(p,q)

a_fluid((u,p),(v,q)) =
  ∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p)-c_Ω(p,q) ) * dΩ +
  ∫( a_Γ(u,v)+b_Γ(u,q)+b_Γ(v,p) ) * dΓ +
  ∫( i_Γg(u,v) - j_Γg(p,q) ) * dΓg

## Structure
# Stabilization and material parameters
function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end
λs, μs = lame_parameters(1.0,0.3)
γg = (λs + 2μs)*0.1
# Terms
σ(ε) = λs*tr(ε)*one(ε) + 2*μs*ε
a_solid(d,s) = ∫(ε(s) ⊙ (σ ∘ ε(d)))dΩout +
  ∫((γg*h^3)*jump(n_Γg⋅∇(s)) ⋅ jump(n_Γg⋅∇(d)))dΓg

## Full problem
a((u,p,d),(v,q,s)) = a_fluid((u,p),(v,q)) + a_solid(d,s) +
  ∫(σf_n(u,p) ⋅ s)dΓ # plus sign because of the normal direction
l((v,q,s)) = 0.0

op = AffineFEOperator(a,l,X,Y)

uh, ph, dh = solve(op)

# Mass flow rate through surface (this should be close to zero)
@show m = sum(∫(ρ*uh⋅n_Γ)dΓ)

writevtk(Ω,path*"fsi-stokes-CutFEM_fluid",
  cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
writevtk(Ωout,path*"fsi-stokes-CutFEM_solid",
  cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])

writevtk(Γ,path*"fsi-stokes-CutFEM_interface",cellfields=["σ⋅n"=>(σ ∘ ε(dh))⋅n_Γ,"σf_n"=>σf_n(uh,ph)])