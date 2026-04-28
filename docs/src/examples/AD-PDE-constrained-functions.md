# Automatic Differentiation of generic PDE-constrained functions

GridapTopOpt provides serial and distributed automatic differentiation methods for generic PDE-constrained functionals and arbitrary maps of those functionals. This works by:
1) Implementing adjoint methods for linear and non-linear finite element problems. We do this by creating new so-called `StateMap` operators [`AffineFEStateMap`](@ref) and [`NonlinearFEStateMap`](@ref), these act like `AffineFEOperator` and `FEOperator` but also (efficently) implement adjoint methods.
2) The adjoint methods implemented in (1) work by overloading the `rrule` method from [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl). As a result, more general backwards automatic differentation packages such as [`Zygote.jl`](https://github.com/FluxML/Zygote.jl) can be utilised to compute derivatives of generic (and possibly quite complicated) maps.

Let's consider the following introductory example. Suppose we wish to differentiate

$$J(u,\kappa) = \left(\int_\Omega(u(\kappa)-u_\textrm{obs})~\mathrm{d}\boldsymbol{x}\right)^{1/2}$$

where $u_\textrm{obs}$ is some observed data and $u$ depends on $\kappa$ through Poisson's equation: find $u\in H^1_g(\Omega)$ such that $a(u,v) = l(v)$ for all $v\in H^1_0(\Omega)$ where

```math
\begin{aligned}
a(u,v) &= \int_\Omega\kappa\nabla u\cdot\nabla v~\mathrm{d}\boldsymbol{x},\\
l(v) &= \int_\Omega vf~\mathrm{d}\boldsymbol{x},
\end{aligned}
```

and $f(x,y) = y$ and $g(x,y) = x$. The derivative of $J(u(\kappa),\kappa)$ with respect to $\kappa$ can be found using the following snippet:

```julia
using Gridap, GridapTopOpt

f(x) = x[2]
g(x) = x[1]

model = CartesianDiscreteModel((0,1,0,1), (10,10))
Ω = Triangulation(model)
dΩ = Measure(Ω, 2)
reffe = ReferenceFE(lagrangian, Float64, 1)
K = TestFESpace(model, reffe)
V = TestFESpace(model, reffe; dirichlet_tags="boundary")
U = TrialFESpace(V,x->x[1])
a(u, v, κ) = ∫(κ * ∇(v) ⋅ ∇(u))dΩ
b(v, κ) = ∫(v*f)dΩ
κ_to_u = AffineFEStateMap(a,b,U,V,K)
l2_norm = StateParamMap((u, κ) -> ∫(u ⋅ u)dΩ,κ_to_u)
u_obs = interpolate(x -> sin(2π*x[1]), V) |> get_free_dof_values
function J(κ)
  u = κ_to_u(κ)
  sqrt(l2_norm(u-u_obs, κ))
end
κ0h = interpolate(1.0, K)
val, grad = val_and_gradient(J, get_free_dof_values(κ0h))
```

The above is quite standard other than [`AffineFEStateMap`](@ref), [`StateParamMap`](@ref), and [`val_and_gradient`](@ref). These work as follows:
- [`AffineFEStateMap`](@ref) represents an (affine) FE operator that maps from $\kappa$ to $u$, and implements adjoint methods.
- [`StateParamMap`](@ref) is a wrapper for a function that produces a `DomainContribution` and handles partial differentiation in a Gridap-friendly. It also implements `rrule` methods so that Zygote can use the partial derivatives.
- [`val_and_gradient`](@ref) is a custom function for computing derivatives using Zygote. This works in almost the same way as [`Zygote.withgradient`](https://fluxml.ai/Zygote.jl/stable/utils/#Zygote.withgradient), except `val_and_gradient` is compatible with PartitionedArrays for distributed computation. Note that in serial, `Zygote.withgradient` will work as usual.

When $J$ maps to multiple numbers (e.g., the case of having an objective and constraints), `val_and_gradient` can be replaced with [`val_and_jacobian`](@ref).

Finally, we verify the above using finite differences as follows:

```julia
using FiniteDiff, Test
fdm_grad = FiniteDiff.finite_difference_gradient(J, get_free_dof_values(κ0h))
@test maximum(abs,grad[1] - fdm_grad)/maximum(abs, grad[1]) < 1e-7
```

Additional examples, as well as examples using GridapDistributed, can be found in [src/tests/](https://github.com/zjwegert/GridapTopOpt.jl/tree/main/test).

!!! note
    Refer to [`CustomPDEConstrainedFunctionals`](@ref) and [`CustomEmbeddedPDEConstrainedFunctionals`](@ref) when attempting to use this with the level-set topology optimisation methods in GridapTopOpt.

## Second-order AD of generic PDE-constrained functions
The capability to compute Hessian-vector products was added in GridapTopOpt v0.5. Under the hood, we use forward-over-reverse automatic differentiation, however most users will only ever need the [`Hvp`](@ref) function. Below, we demonstrate how to compute the Hessian-vector product.

```julia
using Gridap, GridapTopOpt

f(x) = x[2]
g(x) = x[1]

model = CartesianDiscreteModel((0,1,0,1), (10,10))
Ω = Triangulation(model)
dΩ = Measure(Ω, 2)
reffe = ReferenceFE(lagrangian, Float64, 1)
K = TestFESpace(model, reffe)
V = TestFESpace(model, reffe; dirichlet_tags="boundary")
U = TrialFESpace(V,x->x[1])
a(u, v, κ) = ∫(κ * ∇(v) ⋅ ∇(u))dΩ
b(v, κ) = ∫(v*f)dΩ
κ_to_u = AffineFEStateMap(a,b,U,V,K;diff_order=2)
l2_norm = StateParamMap((u, κ) -> ∫(u ⋅ u + 0κ)dΩ,κ_to_u;diff_order=2)
u_obs = interpolate(x -> sin(2π*x[1]), V) |> get_free_dof_values
function J(κ)
  u = κ_to_u(κ)
  sqrt(l2_norm(u-u_obs, κ))
end
κ0h = interpolate(1.0, K)
val, grad = val_and_gradient(J, get_free_dof_values(κ0h))
# Hessian-vector product
vh = interpolate(0.5, K)
Hv = Hvp(J, get_free_dof_values(κ0h),get_free_dof_values(vh))
```

Note that we set the optional parameter `diff_order` in `AffineFEStateMap` and `StateParamMap` to specify that we will be computing the Hessian-vector product.

We again verify using finite differences as follows:

```julia
using FiniteDiff, Test
```

TODO