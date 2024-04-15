# Optimisers

In LevelSetTopOpt we implement optimisation algorithms as [iterators](https://docs.julialang.org/en/v1/manual/interfaces/) that inherit from an abstract type `Optimiser`. A concrete `Optimiser` implementation, say `OptEg`, then implements `iterate(m::OptEg) ↦ (var,state)` and `iterate(m::OptEg,state) ↦ (var,state)`, where `var` and `state` are the available items in the outer loop and internal state of the iterator, respectively. As a result we can iterate over the object `m=OptEg(...)` using `for var in m`. The benefit of this implementation is that the internals of the optimisation method can be hidden in the source code while the explicit `for` loop is still visible to the user. The body of the loop can then be used for auxiliary operations such as writing the optimiser history and other files.

The below describes the implemented optimisers along with the `OptimiserHistory` type. Custom optimisers can be implemented by creating types that inherit from `Optimiser` and extending the interfaces in [Custom optimiser](@ref). 

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

## Custom `Optimiser`
```@docs
LevelSetTopOpt.Optimiser
LevelSetTopOpt.iterate(::LevelSetTopOpt.Optimiser)
LevelSetTopOpt.iterate(::LevelSetTopOpt.Optimiser,state)
LevelSetTopOpt.get_history(::LevelSetTopOpt.Optimiser)
LevelSetTopOpt.converged(::LevelSetTopOpt.Optimiser)
```