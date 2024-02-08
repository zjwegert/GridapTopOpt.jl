# ChainRules

## `PDEConstrainedFunctionals`

```@docs
LevelSetTopOpt.PDEConstrainedFunctionals
LevelSetTopOpt.evaluate!
LevelSetTopOpt.evaluate_functionals!
LevelSetTopOpt.evaluate_derivatives!
LevelSetTopOpt.get_state
```

## `StateParamIntegrandWithMeasure`

```@docs
LevelSetTopOpt.StateParamIntegrandWithMeasure
LevelSetTopOpt.rrule(u_to_j::LevelSetTopOpt.StateParamIntegrandWithMeasure,uh,φh)
```

## Implemented types of `AbstractFEStateMap`

```@docs
LevelSetTopOpt.AbstractFEStateMap
```

### `AffineFEStateMap`
```@docs
LevelSetTopOpt.AffineFEStateMap
LevelSetTopOpt.AffineFEStateMap(a::Function,l::Function,U,V,V_φ,U_reg,φh,dΩ...;assem_U = SparseMatrixAssembler(U,V),assem_adjoint = SparseMatrixAssembler(V,U),assem_deriv = SparseMatrixAssembler(U_reg,U_reg),ls::LinearSolver = LUSolver(),adjoint_ls::LinearSolver = LUSolver())
```

### `NonlinearFEStateMap`
```@docs
LevelSetTopOpt.NonlinearFEStateMap
LevelSetTopOpt.NonlinearFEStateMap(res::Function,U,V,V_φ,U_reg,φh,dΩ...;assem_U = SparseMatrixAssembler(U,V),assem_adjoint = SparseMatrixAssembler(V,U),assem_deriv = SparseMatrixAssembler(U_reg,U_reg),nls::NonlinearSolver = NewtonSolver(LUSolver();maxiter=50,rtol=1.e-8,verbose=true),adjoint_ls::LinearSolver = LUSolver())
```

### `RepeatingAffineFEStateMap`
```@docs
LevelSetTopOpt.RepeatingAffineFEStateMap
LevelSetTopOpt.RepeatingAffineFEStateMap(nblocks::Int,a::Function,l::Vector{<:Function},U0,V0,V_φ,U_reg,φh,dΩ...;assem_U = SparseMatrixAssembler(U0,V0),assem_adjoint = SparseMatrixAssembler(V0,U0),assem_deriv = SparseMatrixAssembler(U_reg,U_reg),ls::LinearSolver = LUSolver(),adjoint_ls::LinearSolver = LUSolver())
```

## Advanced

### Inheriting from `AbstractFEStateMap`

#### Existing methods
```@docs
LevelSetTopOpt.rrule(φ_to_u::LevelSetTopOpt.AbstractFEStateMap,φh)
LevelSetTopOpt.pullback
```

#### Required to implement
```@docs
LevelSetTopOpt.forward_solve!
LevelSetTopOpt.adjoint_solve!
LevelSetTopOpt.update_adjoint_caches!
LevelSetTopOpt.dRdφ
LevelSetTopOpt.get_state(::LevelSetTopOpt.AbstractFEStateMap)
LevelSetTopOpt.get_measure
LevelSetTopOpt.get_spaces
LevelSetTopOpt.get_assemblers
LevelSetTopOpt.get_trial_space
LevelSetTopOpt.get_test_space
LevelSetTopOpt.get_aux_space
LevelSetTopOpt.get_deriv_space
LevelSetTopOpt.get_pde_assembler
LevelSetTopOpt.get_deriv_assembler
```

### `IntegrandWithMeasure`

```@docs
LevelSetTopOpt.IntegrandWithMeasure
LevelSetTopOpt.gradient
LevelSetTopOpt.jacobian
```