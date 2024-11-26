using Gridap, Gridap.Geometry, Gridap.Adaptivity
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt

path = "./results/stokes & navier-stokes testing/"
mkpath(path)

# Formulation taken from
# André Massing · Mats G. Larson · Anders Logg · Marie E. Rognes,
# A Stabilized Nitsche Fictitious Domain Method for the Stokes Problem
# J Sci Comput (2014) 61:604–628 DOI 10.1007/s10915-014-9838-9

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
f_Γ_NoSlip(x) = x[2] ≈ 0 || x[2] ≈ 1
update_labels!(1,model,f_Γ_D,"Gamma_D")
update_labels!(2,model,f_Γ_NoSlip,"Gamma_NoSlip")

# Cut the background model
reffe_scalar = ReferenceFE(lagrangian,Float64,1)
V_φ = TestFESpace(model,reffe_scalar)
φh = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.1,V_φ)
geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)
cutgeo_facets = cut_facets(model,geo)

# Generate the "active" model
Ω_act = Triangulation(cutgeo,ACTIVE)

# Setup integration meshes
Ω = Triangulation(cutgeo,PHYSICAL)
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
dΓ = Measure(Γ,degree)
dΓg = Measure(Γg,degree)
dΓi = Measure(Γi,degree)

# Setup FESpace

uin(x) = VectorValue(x[2]*(1-x[2]),0.0)

reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
reffe_p = ReferenceFE(lagrangian,Float64,order,space=:P)

V = TestFESpace(Ω_act,reffe_u,conformity=:H1,dirichlet_tags=["Gamma_D","Gamma_NoSlip"])
Q = TestFESpace(Ω_act,reffe_p,conformity=:H1)

U = TrialFESpace(V,[x->uin(x),VectorValue(0.0,0.0)])
P = TrialFESpace(Q)

X = MultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V,Q])

# Stabilization parameters
β0 = 0.25
β1 = 0.2
β2 = 0.1
β3 = 0.05
γ = 10.0

# Weak form
a_Ω(u,v) = ∇(u) ⊙ ∇(v)
b_Ω(v,p) = - (∇⋅v)*p
c_Γi(p,q) = (β0*h)*jump(p)*jump(q)
c_Ω(p,q) = (β1*h^2)*∇(p)⋅∇(q)
a_Γ(u,v) = - (n_Γ⋅∇(u))⋅v - u⋅(n_Γ⋅∇(v)) + (γ/h)*u⋅v
b_Γ(v,p) = (n_Γ⋅v)*p
i_Γg(u,v) = (β2*h)*jump(n_Γg⋅∇(u))⋅jump(n_Γg⋅∇(v))
j_Γg(p,q) = (β3*h^3)*jump(n_Γg⋅∇(p))*jump(n_Γg⋅∇(q)) + c_Γi(p,q)

a((u,p),(v,q)) =
  ∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p)-c_Ω(p,q) ) * dΩ +
  ∫( a_Γ(u,v)+b_Γ(u,q)+b_Γ(v,p) ) * dΓ +
  ∫( i_Γg(u,v) - j_Γg(p,q) ) * dΓg

l((v,q)) = 0.0

op = AffineFEOperator(a,l,X,Y)

uh, ph = solve(op)

writevtk(Ω,path*"2-results",
  cellfields=["uh"=>uh,"ph"=>ph])