# LevelSetEvolution
In GridapTopOpt, the a level-set function is evolved and reinitialised as a signed distance function using an `Evolver` and `Reinitialiser` implementations, respectively. These are wrapped in an object `LevelSetEvolution` for the purpose of optimisation.

List of `Evolver` types:

|             **Method**            | **Ambient mesh type** |   **Cached**   | **Preferred Method** |
|:---------------------------------:|:---------------------:|:--------------:|:--------------------:|
| [`FiniteDifferenceEvolver`](@ref) | Cartesian             |       ✓        |                      |
| [`CutFEMEvolver`](@ref)           | Unstructured          |       ✗*       |          ☆           |
_*: Caching is disabled due to a bug. This is will be improved in future._

List of `Reinitialiser` types:

|                **Method**               | **Ambient mesh type** |  **Cached** | **Differentiable** | **Preferred Method** |
|:---------------------------------------:|:---------------------:|:-----------:|:------------------:|:--------------------:|
| [`FiniteDifferenceReinitialiser`](@ref) | Cartesian             |      ✓      |          ✗         |                      |
| [`StabilisedReinitialiser`](@ref)       | Unstructured          |      ✓      |          ✗         |                      |
| [`HeatReinitialiser`](@ref)             | Unstructured          |      ✓      |          ✓         |          ☆           |

!!! note
    The finite difference methods are based on the upwinding scheme of Osher and Fedkiw ([link](https://doi.org/10.1007/b98879)). These are extremely efficient as they require no linear solves. They are, however, currently restricted to meshes built from `CartesianDiscreteModel`. Note as well that the `FiniteDifferenceReinitialiser` is can result in slight movement of the zero isosurface.

## `LevelSetEvolution`
```@autodocs
Modules = [GridapTopOpt]
Pages = ["LevelSetEvolution/LevelSetEvolution.jl"]
```

## Level-Set Evolvers
### `FiniteDifferenceEvolver`
```@autodocs
Modules = [GridapTopOpt]
Pages = ["LevelSetEvolution/Evolver/FiniteDifferenceEvolver.jl"]
```

```@docs
GridapTopOpt.FirstOrderStencil
```

### `CutFEMEvolver`
```@autodocs
Modules = [GridapTopOpt]
Pages = ["LevelSetEvolution/Evolver/CutFEMEvolver.jl"]
```

## Level-Set Reinitialisers
### `FiniteDifferenceReinitialiser`
```@autodocs
Modules = [GridapTopOpt]
Pages = ["LevelSetEvolution/Reinitialiser/FiniteDifferenceReinitialiser.jl"]
```

### `StabilisedReinitialiser`
```@autodocs
Modules = [GridapTopOpt]
Pages = ["LevelSetEvolution/Reinitialiser/StabilisedReinitialiser.jl"]
```

### `HeatReinitialiser`
```@autodocs
Modules = [GridapTopOpt]
Pages = ["LevelSetEvolution/Reinitialiser/HeatReinitialiser.jl"]
```

### `IdentityReinitialiser`
```@autodocs
Modules = [GridapTopOpt]
Pages = ["LevelSetEvolution/Reinitialiser/IdentityReinitialiser.jl"]
```

## Your own custom `Evolver`
```@autodocs
Modules = [GridapTopOpt]
Pages = ["LevelSetEvolution/Evolver/Evolver.jl"]
```

## Your own custom `Reinitialiser`
```@autodocs
Modules = [GridapTopOpt]
Pages = ["LevelSetEvolution/Reinitialiser/Reinitialiser.jl"]
```

## Custom `Stencil` for finite difference methods

```@docs
GridapTopOpt.Stencil
GridapTopOpt.evolve!(::GridapTopOpt.Stencil,φ,vel,Δt,Δx,isperiodic,caches)
GridapTopOpt.reinit!(::GridapTopOpt.Stencil,φ_new,φ,vel,Δt,Δx,isperiodic,caches)
GridapTopOpt.allocate_caches(::GridapTopOpt.Stencil,φ,vel)
GridapTopOpt.check_order
```