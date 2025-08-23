# Automatic Differentiation of generic PDE-constrained functions

```julia
using Gridap, GridapTopOpt

model = CartesianDiscreteModel((0,1,0,1), (10,10))
Ω = Triangulation(model)
dΩ = Measure(Ω, 2)
reffe = ReferenceFE(lagrangian, Float64, 1)
K = TestFESpace(model, reffe)
V = TestFESpace(model, reffe; dirichlet_tags="boundary")
U = TrialFESpace(V,x->x[1])
g(x) = x[2]
a(u, v, κ) = ∫(κ * ∇(v) ⋅ ∇(u))dΩ
b(v, κ) = ∫(v*g)dΩ
κ_to_u = AffineFEStateMap(a,b,U,V,K)
l2_norm = GridapTopOpt.StateParamMap((u,κ) -> ∫(u ⋅ u)dΩ,κ_to_u)
u_obs = interpolate(x -> sin(2π*x[1]), V) |> get_free_dof_values
function J(κ)
  u = κ_to_u(κ)
  sqrt(l2_norm(u-u_obs, κ))
end
κ0h = interpolate(1.0, K)
val, grad = GridapTopOpt.val_and_gradient(J, get_free_dof_values(κ0h))
```

```julia
using FiniteDiff, Test
fdm_grad = FiniteDiff.finite_difference_gradient(J, get_free_dof_values(κ0h))
@test maximum(abs,grad[1] - fdm_grad)/maximum(abs, grad[1]) < 1e-7
```