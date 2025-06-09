module PoissonMultiFieldTests

using Gridap
using Test

u1(x) = x[1]^2
u2(x) = x[2]*x[1]

f1(x) = -Δ(u1)(x)
f2(x) = u1(x)*-(Δ(u2)(x))

domain = (0,1,0,1)
cells = (2,2)
model = CartesianDiscreteModel(domain,cells)

order = 2
V = FESpace(model, ReferenceFE(lagrangian,Float64,order),conformity=:H1,dirichlet_tags="boundary")
U1 = TrialFESpace(V,u1)
U2 = TrialFESpace(V,u2)

Ω = Triangulation(model)
Γ = BoundaryTriangulation(model,tags=1)
n_Γ = get_normal_vector(Γ)

degree = 2*order
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)

a1(u1,v1) = ∫( ∇(v1)⋅∇(u1) )*dΩ
l1(v1) = ∫( v1*f1 )*dΩ

op1 = AffineFEOperator(a1,l1,U1,V)
u1h = solve(op1)

a2(u2,v2) = ∫( u1h * ∇(v2)⋅∇(u2) )*dΩ
l2(v2) = ∫( v2*f2 )*dΩ

op2 = AffineFEOperator(a2,l2,U2,V)
u2h = solve(op2)

a3((u1,u2),(v1,v2)) = ∫( ∇(v1)⋅∇(u1) )dΩ +  ∫( u1*∇(v2)⋅∇(u2))dΩ
l3((v1,v2)) = ∫( v1*f1 )*dΩ + ∫( v2*f2 )*dΩ

X = MultiFieldFESpace([U1,U2])
Y = MultiFieldFESpace([V,V])

op3 = FEOperator((u,v)->a3(u,v)-l3(v),X,Y)
solver = NLSolver(LUSolver(), show_trace=true, method=:newton, iterations=5)
(u1hm,u2hm) = solve(solver,op3)

op3 = AffineFEOperator(a3,l3,X,Y)
solve(op3)

end