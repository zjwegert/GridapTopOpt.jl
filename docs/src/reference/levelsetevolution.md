# LevelSetEvolution (NEED TO UPDATE THIS)

## `HamiltonJacobiEvolution`
```@docs
LevelSetTopOpt.HamiltonJacobiEvolution
LevelSetTopOpt.HamiltonJacobiEvolution(stencil::LevelSetTopOpt.Stencil,model,space,tol=1.e-3,max_steps=100,max_steps_reinit=2000)
LevelSetTopOpt.advect!
LevelSetTopOpt.reinit!
```

## Stencils

```@docs
LevelSetTopOpt.FirstOrderStencil
```

## Custom `Stencil`

```@docs
LevelSetTopOpt.Stencil
LevelSetTopOpt.advect!(::LevelSetTopOpt.Stencil,φ,vel,Δt,Δx,isperiodic,caches)
LevelSetTopOpt.reinit!(::LevelSetTopOpt.Stencil,φ_new,φ,vel,Δt,Δx,isperiodic,caches)
LevelSetTopOpt.compute_Δt(::LevelSetTopOpt.Stencil,φ,vel)
LevelSetTopOpt.allocate_caches(::LevelSetTopOpt.Stencil,φ,vel)
```