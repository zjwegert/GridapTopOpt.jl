# LevelSetEvolution
In LevelSetTopOpt, the level set is evolved and reinitialised using a `LevelSetEvolution` method. The most standard of these is the Hamilton-Jacobi evolution equation solved using a first order upwind finite difference scheme. A forward Euler in time method is provided below via `HamiltonJacobiEvolution <: LevelSetEvolution` along with an upwind finite difference stencil for the spatial discretisation via `FirstOrderStencil`.

This can be extended in several ways. For example, higher order spatial stencils can be implemented by extending the `Stencil` interface below. In addition, more advanced ODE solvers could be implemented (e.g., Runge–Kutta methods) or entirely different level set evolution methods by extending the `LevelSetEvolution` interface below.

## `HamiltonJacobiEvolution`
```@docs
LevelSetTopOpt.HamiltonJacobiEvolution
LevelSetTopOpt.HamiltonJacobiEvolution(stencil::LevelSetTopOpt.Stencil,model,space,tol=1.e-3,max_steps=100,max_steps_reinit=2000)
LevelSetTopOpt.evolve!
LevelSetTopOpt.reinit!
LevelSetTopOpt.get_dof_Δ(m::HamiltonJacobiEvolution)
```

## Spatial stencils for `HamiltonJacobiEvolution`

```@docs
LevelSetTopOpt.FirstOrderStencil
```

## Custom `Stencil`

```@docs
LevelSetTopOpt.Stencil
LevelSetTopOpt.evolve!(::LevelSetTopOpt.Stencil,φ,vel,Δt,Δx,isperiodic,caches)
LevelSetTopOpt.reinit!(::LevelSetTopOpt.Stencil,φ_new,φ,vel,Δt,Δx,isperiodic,caches)
LevelSetTopOpt.allocate_caches(::LevelSetTopOpt.Stencil,φ,vel)
LevelSetTopOpt.check_order
```

## Custom `LevelSetEvolution`
To implement a custom level set evolution method, we can extend the methods below. For example, one could consider Reaction-Diffusion-based evolution of the level set function. This can be solved with a finite element method and so we can implement a new type that inherits from `LevelSetEvolution` independently of the `Stencil` types.

```@docs
LevelSetTopOpt.LevelSetEvolution
LevelSetTopOpt.evolve!(::LevelSetTopOpt.LevelSetEvolution,φ,args...)
LevelSetTopOpt.reinit!(::LevelSetTopOpt.LevelSetEvolution,φ,args...)
LevelSetTopOpt.get_dof_Δ(::LevelSetTopOpt.LevelSetEvolution)
```