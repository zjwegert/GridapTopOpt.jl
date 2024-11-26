using Gridap,GridapTopOpt, GridapSolvers
using Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters

path = "./results/FCM_2d_navier-stokes/"
rm(path,force=true,recursive=true)
mkpath(path)
n = 50
order = 2

_model = CartesianDiscreteModel((0,1,0,1),(n,n))
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = ref_model.model
el_Δ = get_el_Δ(_model)
h = maximum(el_Δ)
h_refine = maximum(el_Δ)/2
f_Γ_D(x) = x[1] ≈ 0
f_Γ_NoSlip(x) = x[2] ≈ 0 || x[2] ≈ 1
update_labels!(1,model,f_Γ_D,"Gamma_D")
update_labels!(2,model,f_Γ_NoSlip,"Gamma_NoSlip")

uin(x) = VectorValue(x[2]*(1-x[2]),0.0)

Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)

## Levet-set function space and derivative regularisation space
reffe_scalar = ReferenceFE(lagrangian,Float64,1)
V_φ = TestFESpace(model,reffe_scalar)

## Levet-set function
φh = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.1,V_φ)
geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)
cutgeo_facets = cut_facets(model,geo)

# Setup integration meshes
Ωin = Triangulation(cutgeo,PHYSICAL)
Ωout = Triangulation(cutgeo,PHYSICAL_OUT)
Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)

degree = 2*order
dΩin = Measure(Ωin,degree)
dΩout = Measure(Ωout,degree)
dΓ = Measure(Γ,degree)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order,space=:P)
reffe_p = ReferenceFE(lagrangian,Float64,order-1,space=:P)

V = TestFESpace(Ω,reffe_u,conformity=:H1,dirichlet_tags=["Gamma_D","Gamma_NoSlip"])
Q = TestFESpace(Ω,reffe_p,conformity=:H1)

U = TrialFESpace(V,[x->uin(x),VectorValue(0.0,0.0)])
P = TrialFESpace(Q)
X = MultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V,Q])

## Problem definition
γ = 20

a((u,p),(v,q)) =
  ∫( ∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p ) * dΩin +
  ∫( ∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p ) * dΩout +
#   ∫( 1e-3*(∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p) ) * dΩout +
  ∫( (γ/h)*v⋅u - v⋅(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))⋅u + (p*n_Γ)⋅v + (q*n_Γ)⋅u ) * dΓ

l((v,q)) = 0

op = AffineFEOperator(a,l,X,Y)

xh = solve(op)

uh,ph = xh

writevtk(Ωin,path*"solution",cellfields=["uh"=>uh,"ph"=>ph])