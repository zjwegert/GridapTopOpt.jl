using Gridap, Gridap.Geometry, Gridap.Adaptivity
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt

path = "./results/fsi testing/"
mkpath(path)

# Cut the background model
L = 1.0#2.0
H = 1.0#0.5
x0 = 0.5
l = 0.4
w = 0.1
a = 0.5#0.3
b = 0.01

n = 100
partition = (n,n)
D = length(partition)
_model = CartesianDiscreteModel((0,L,0,H),partition)
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = ref_model.model

el_Δ = get_el_Δ(_model)
h = maximum(el_Δ)
f_Γ_D(x) = x[1] ≈ 0
f_Γ_NoSlipTop(x) = x[2] ≈ H
f_Γ_NoSlipBottom(x) = x[2] ≈ 0
f_NonDesign(x) = ((x0 - w/2 <= x[1] <= x0 + w/2 && 0.0 <= x[2] <= a) &&
  x0 - w/2 <= x[1] <= x0 + w/2 && 0.0 <= x[2] <= b)
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
Ω_act = Triangulation(model)

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
order = 2
degree = 2*order
dΩ = Measure(Ω,degree)
dΩout = Measure(Ωout,degree)
dΓ = Measure(Γ,degree)
dΓg = Measure(Γg,degree)
dΓi = Measure(Γi,degree)

# Setup FESpace

uin(x) = VectorValue(x[2]*(1-x[2]),0.0)

reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
reffe_p = ReferenceFE(lagrangian,Float64,order-1,space=:P)
reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

V = TestFESpace(Ω_act,reffe_u,conformity=:H1,dirichlet_tags=["Gamma_D","Gamma_NoSlipTop","Gamma_NoSlipBottom"])
Q = TestFESpace(Ω_act,reffe_p,conformity=:C0)
T = TestFESpace(Ω_act ,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_NoSlipBottom"])

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
γ = 1000.0

# Terms
σf_n(u,p) = μ*∇(u)⋅n_Γ - p*n_Γ
a_Ω(u,v) = μ*(∇(u) ⊙ ∇(v))
b_Ω(v,p) = - (∇⋅v)*p

a_fluid((u,p),(v,q)) =
  ∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p)) * dΩ +
  ∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p) + (γ/h)*u⋅v ) * dΩout

## Structure
# Stabilization and material parameters
function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end
λs, μs = lame_parameters(1.0,0.3)
ϵ = (λs + 2μs)*1e-3
# Terms
σ(ε) = λs*tr(ε)*one(ε) + 2*μs*ε
a_solid(d,s) = ∫(ε(s) ⊙ (σ ∘ ε(d)))dΩout +
  ∫(ϵ*(ε(s) ⊙ (σ ∘ ε(d))))dΩ # Ersatz

## Full problem
a((u,p,d),(v,q,s)) = a_fluid((u,p),(v,q)) + a_solid(d,s) +
  ∫(σf_n(u,p) ⋅ s)dΓ # plus sign because of the normal direction
l((v,q,s)) = 0.0

op = AffineFEOperator(a,l,X,Y)

uh, ph, dh = solve(op)

# Mass flow rate through surface (this should be close to zero)
@show m = sum(∫(ρ*uh⋅n_Γ)dΓ)

writevtk(Ω_act,path*"fsi-stokes-brinkmann-P2P1_elast-ersatz_full",
  cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
writevtk(Ω,path*"fsi-stokes-brinkmann-P2P1_elast-ersatz_fluid",
  cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
writevtk(Ωout,path*"fsi-stokes-brinkmann-P2P1_elast-ersatz_solid",
  cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])

writevtk(Γ,path*"fsi-stokes-brinkmann-P2P1_elast-ersatz_interface",cellfields=["σ⋅n"=>(σ ∘ ε(dh))⋅n_Γ,"σf_n"=>σf_n(uh,ph)])