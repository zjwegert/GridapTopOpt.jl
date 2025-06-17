# Upwind finite difference schemes
The most standard scheme of solving the Hamilton-Jacobi evolution equation and reinitialisation equation is a first order upwind finite difference scheme. A forward Euler in time method is provided below via `HamiltonJacobiEvolution <: LevelSetEvolution` along with an upwind finite difference stencil for the spatial discretisation via `FirstOrderStencil`.

This can be extended in several ways. For example, higher order spatial stencils can be implemented by extending the `Stencil` interface below. In addition, more advanced ODE solvers could be implemented (e.g., Runge–Kutta methods) or entirely different level set evolution methods by extending the `LevelSetEvolution` interface below.

## `HamiltonJacobiEvolution`
```@docs
GridapTopOpt.HamiltonJacobiEvolution
GridapTopOpt.HamiltonJacobiEvolution(stencil::GridapTopOpt.Stencil,model,space,tol=1.e-3,max_steps=100,max_steps_reinit=2000)
GridapTopOpt.evolve!
GridapTopOpt.reinit!
GridapTopOpt.get_dof_Δ(m::HamiltonJacobiEvolution)
```

## Spatial stencils for `HamiltonJacobiEvolution`

```@docs
GridapTopOpt.FirstOrderStencil
```

## Custom `Stencil`

```@docs
GridapTopOpt.Stencil
GridapTopOpt.evolve!(::GridapTopOpt.Stencil,φ,vel,Δt,Δx,isperiodic,caches)
GridapTopOpt.reinit!(::GridapTopOpt.Stencil,φ_new,φ,vel,Δt,Δx,isperiodic,caches)
GridapTopOpt.allocate_caches(::GridapTopOpt.Stencil,φ,vel)
GridapTopOpt.check_order
```