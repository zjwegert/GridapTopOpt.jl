# LevelSetEvolution
In GridapTopOpt, the level set is evolved and reinitialised using a `LevelSetEvolution` method. The most standard of these is the Hamilton-Jacobi evolution equation solved using a first order upwind finite difference scheme. A forward Euler in time method is provided below via `HamiltonJacobiEvolution <: LevelSetEvolution` along with an upwind finite difference stencil for the spatial discretisation via `FirstOrderStencil`.

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

## Custom `LevelSetEvolution`
To implement a custom level set evolution method, we can extend the methods below. For example, one could consider Reaction-Diffusion-based evolution of the level set function. This can be solved with a finite element method and so we can implement a new type that inherits from `LevelSetEvolution` independently of the `Stencil` types.

```@docs
GridapTopOpt.LevelSetEvolution
GridapTopOpt.evolve!(::GridapTopOpt.LevelSetEvolution,φ,args...)
GridapTopOpt.reinit!(::GridapTopOpt.LevelSetEvolution,φ,args...)
GridapTopOpt.get_dof_Δ(::GridapTopOpt.LevelSetEvolution)
```