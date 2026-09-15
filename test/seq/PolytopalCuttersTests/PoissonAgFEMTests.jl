module PoissonAgFEMTests

using Gridap
using GridapEmbedded
using Test

using GridapTopOpt

u(x) = x[1] - x[2]
f(x) = -Δ(u)(x)
ud(x) = u(x)

R = 0.5
L = 0.8*(2*R)
p1 = Point(0.0,0.0)
p2 = p1 + VectorValue(L,0.0)

t = 1.01
pmin = p1-t*R
pmax = p1+t*R

n = 30
partition = (n,n)
bgmodel = simplexify(CartesianDiscreteModel(pmin,pmax,partition))
dp = pmax - pmin
const h = dp[1]/n

reffe = ReferenceFE(lagrangian,Float64,1)
V_φ = TestFESpace(bgmodel,reffe)
φh1 = interpolate(x->(x[1]-p1[1])^2+(x[2]-p1[2])^2-R^2,V_φ)
φh2 = interpolate(x->(x[1]-p2[1])^2+(x[2]-p2[2])^2-R^2,V_φ)
geo1 = DiscreteGeometryFromFEFunction(φh1,bgmodel)
geo2 = DiscreteGeometryFromFEFunction(φh2,bgmodel)
geo3 = setdiff(geo1,geo2)

cutgeo = cut(PolytopalLevelSetCutter(),bgmodel,geo3)

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

Ω_bg = Triangulation(bgmodel)
Ω_act = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)
Γ = EmbeddedBoundary(cutgeo)

n_Γ = get_normal_vector(Γ)

order = 1
degree = 2*order
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)

model = get_active_model(Ω_act)
Vstd = FESpace(Ω_act,FiniteElements(PhysicalDomain(),model,lagrangian,Float64,order))

V = AgFEMSpace(Vstd,aggregates)
U = TrialFESpace(V)

const γd = 10.0

a(u,v) =
  ∫( ∇(v)⋅∇(u) ) * dΩ +
  ∫( (γd/h)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u ) * dΓ

l(v) =
  ∫( v*f ) * dΩ +
  ∫( (γd/h)*v*ud - (n_Γ⋅∇(v))*ud ) * dΓ

op = AffineFEOperator(a,l,U,V)
uh = solve(op)

e = u - uh

l2(u) = sqrt(sum( ∫( u*u )*dΩ ))
h1(u) = sqrt(sum( ∫( u*u + ∇(u)⋅∇(u) )*dΩ ))

el2 = l2(e)
eh1 = h1(e)
ul2 = l2(uh)
uh1 = h1(uh)

#colors = color_aggregates(aggregates,bgmodel)
#writevtk(Ω_bg,"trian",celldata=["aggregate"=>aggregates,"color"=>colors],cellfields=["uh"=>uh])
#writevtk(Ω,"trian_O",cellfields=["uh"=>uh])
#writevtk(Γ,"trian_G")
@test el2/ul2 < 1.e-8
@test eh1/uh1 < 1.e-7

end # module
