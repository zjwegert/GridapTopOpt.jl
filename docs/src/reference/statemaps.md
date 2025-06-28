# StateMaps

!!! compat
    Formally known as `ChainRules`.

## `PDEConstrainedFunctionals`

```@docs
GridapTopOpt.PDEConstrainedFunctionals
GridapTopOpt.evaluate!(pcf::PDEConstrainedFunctionals,φh)
GridapTopOpt.evaluate_functionals!(pcf::PDEConstrainedFunctionals,φh)
GridapTopOpt.evaluate_derivatives!(pcf::PDEConstrainedFunctionals,φh)
GridapTopOpt.get_state
```

## `StateParamMap`

```@docs
GridapTopOpt.StateParamMap
GridapTopOpt.rrule(u_to_j::GridapTopOpt.StateParamMap,uh,φh)
```

## Implemented types of `AbstractFEStateMap`

```@docs
GridapTopOpt.AbstractFEStateMap
```

### `AffineFEStateMap`
```@autodocs
Modules = [GridapTopOpt]
Pages = ["StateMaps/AffineFEStateMaps.jl"]
```

### `NonlinearFEStateMap`
```@autodocs
Modules = [GridapTopOpt]
Pages = ["StateMaps/NonlinearFEStateMaps.jl"]
```

### `RepeatingAffineFEStateMap`
```@autodocs
Modules = [GridapTopOpt]
Pages = ["StateMaps/RepeatingAffineFEStateMaps.jl"]
```

### `StaggeredAffineFEStateMap` and `StaggeredNonlinearFEStateMap`
```@autodocs
Modules = [GridapTopOpt]
Pages = ["StateMaps/StaggeredFEStateMaps.jl"]
```

## Advanced

### Inheriting from `AbstractFEStateMap`

#### Existing methods
```@docs
GridapTopOpt.rrule(φ_to_u::GridapTopOpt.AbstractFEStateMap,φh)
GridapTopOpt.pullback
```

#### Required to implement
```@docs
GridapTopOpt.forward_solve!
GridapTopOpt.adjoint_solve!
GridapTopOpt.update_adjoint_caches!
GridapTopOpt.dRdφ
GridapTopOpt.get_state(::GridapTopOpt.AbstractFEStateMap)
GridapTopOpt.get_spaces
GridapTopOpt.get_assemblers
GridapTopOpt.get_trial_space
GridapTopOpt.get_test_space
GridapTopOpt.get_aux_space
GridapTopOpt.get_deriv_space
GridapTopOpt.get_pde_assembler
GridapTopOpt.get_deriv_assembler
```

### Partial derivatives
```@autodocs
Modules = [GridapTopOpt]
Pages = ["StateMaps/StateMaps.jl"]
```