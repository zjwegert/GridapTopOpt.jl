module PoissonCutFEMTests

using Gridap
using GridapEmbedded
using Test

using GridapTopOpt

# Manufactured solution
u(x) = x[1] + x[2] - x[3]
∇u(x) = ∇(u)(x)
f(x) = -Δ(u)(x)
ud(x) = u(x)

# Select geometry

const _R = 0.7
n = 10
partition = (n,n,n)

# Setup background model
_geom = sphere(_R)
box = get_metadata(_geom)
bgmodel = simplexify(CartesianDiscreteModel(box.pmin,box.pmax,partition))
dp = box.pmax - box.pmin
const h = dp[1]/n

reffe = ReferenceFE(lagrangian,Float64,1)
V_φ = TestFESpace(bgmodel,reffe)
φh = interpolate(x->(x[1])^2+(x[2])^2+(x[3])^2-_R^2,V_φ)
geom = DiscreteGeometryFromFEFunction(φh,bgmodel)

# Cut the background model
cutdisc = cut(PolytopalLevelSetCutter(),bgmodel,geom)

# Setup integration meshes
Ωact = Triangulation(cutdisc,ACTIVE)
Ω = Triangulation(cutdisc,PHYSICAL)
Γ = EmbeddedBoundary(cutdisc)
Γg = GhostSkeleton(cutdisc)

# Setup normal vectors
n_Γ = get_normal_vector(Γ)
n_Γg = get_normal_vector(Γg)

# Setup Lebesgue measures
order = 1
degree = 2*order
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)
dΓg = Measure(Γg,degree)

# Setup FESpace
V = TestFESpace(Ωact,ReferenceFE(lagrangian,Float64,order),conformity=:H1)
U = TrialFESpace(V)

# Weak form Nitsche + ghost penalty (CutFEM paper Sect. 6.1)
const γd = 10.0
const γg = 0.1

A(u,v) =
  ∫( ∇(v)⋅∇(u) ) * dΩ +
  ∫( (γd/h)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u ) * dΓ +
  ∫( (γg*h)*jump(n_Γg⋅∇(v))*jump(n_Γg⋅∇(u)) ) * dΓg

L(v) =
  ∫( v*f ) * dΩ +
  ∫( (γd/h)*v*ud - (n_Γ⋅∇(v))*ud ) * dΓ

# FE problem
op = AffineFEOperator(A,L,U,V)
uh = solve(op)

e = u - uh

# Postprocess
l2(u) = sqrt(sum( ∫( u*u )*dΩ ))
h1(u) = sqrt(sum( ∫( u*u + ∇(u)⋅∇(u) )*dΩ ))

el2 = l2(e)
eh1 = h1(e)
ul2 = l2(uh)
uh1 = h1(uh)

# writevtk(Ω,"results",cellfields=["uh"=>uh])
@test el2/ul2 < 1.e-8
@test eh1/uh1 < 1.e-7

end # module
