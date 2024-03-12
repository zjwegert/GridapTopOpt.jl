# Optimisers

## Lagrangian & Augmented Lagrangian method 
```@autodocs
Modules = [LevelSetTopOpt]
Pages = ["Optimisers/AugmentedLagrangian.jl"]
```

## Hilbertian projection method
```@autodocs
Modules = [LevelSetTopOpt]
Pages = ["Optimisers/HilbertianProjection.jl"]
```

```@autodocs
Modules = [LevelSetTopOpt]
Pages = ["Optimisers/OrthogonalisationMaps.jl"]
```

## Optimiser history
```@docs
LevelSetTopOpt.OptimiserHistory
LevelSetTopOpt.OptimiserHistorySlice
```

## Custom optimiser
```@docs
LevelSetTopOpt.Optimiser
LevelSetTopOpt.iterate(::LevelSetTopOpt.Optimiser)
LevelSetTopOpt.iterate(::LevelSetTopOpt.Optimiser,state)
LevelSetTopOpt.get_history(::LevelSetTopOpt.Optimiser)
LevelSetTopOpt.converged(::LevelSetTopOpt.Optimiser)
```