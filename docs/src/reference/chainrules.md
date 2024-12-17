# ChainRules

## `PDEConstrainedFunctionals`

```@docs
GridapTopOpt.PDEConstrainedFunctionals
GridapTopOpt.evaluate!
GridapTopOpt.evaluate_functionals!
GridapTopOpt.evaluate_derivatives!
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
```@docs
GridapTopOpt.AffineFEStateMap
GridapTopOpt.AffineFEStateMap(a::Function,l::Function,U,V,V_φ,U_reg,φh,dΩ...;assem_U = SparseMatrixAssembler(U,V),assem_adjoint = SparseMatrixAssembler(V,U),assem_deriv = SparseMatrixAssembler(U_reg,U_reg),ls::LinearSolver = LUSolver(),adjoint_ls::LinearSolver = LUSolver())
```

### `NonlinearFEStateMap`
```@docs
GridapTopOpt.NonlinearFEStateMap
GridapTopOpt.NonlinearFEStateMap(res::Function,U,V,V_φ,U_reg,φh,dΩ...;assem_U = SparseMatrixAssembler(U,V),assem_adjoint = SparseMatrixAssembler(V,U),assem_deriv = SparseMatrixAssembler(U_reg,U_reg),nls::NonlinearSolver = NewtonSolver(LUSolver();maxiter=50,rtol=1.e-8,verbose=true),adjoint_ls::LinearSolver = LUSolver())
```

### `RepeatingAffineFEStateMap`
```@docs
GridapTopOpt.RepeatingAffineFEStateMap
GridapTopOpt.RepeatingAffineFEStateMap(nblocks::Int,a::Function,l::Vector{<:Function},U0,V0,V_φ,U_reg,φh,dΩ...;assem_U = SparseMatrixAssembler(U0,V0),assem_adjoint = SparseMatrixAssembler(V0,U0),assem_deriv = SparseMatrixAssembler(U_reg,U_reg),ls::LinearSolver = LUSolver(),adjoint_ls::LinearSolver = LUSolver())
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
GridapTopOpt.get_measure
GridapTopOpt.get_spaces
GridapTopOpt.get_assemblers
GridapTopOpt.get_trial_space
GridapTopOpt.get_test_space
GridapTopOpt.get_aux_space
GridapTopOpt.get_deriv_space
GridapTopOpt.get_pde_assembler
GridapTopOpt.get_deriv_assembler
```

### `IntegrandWithMeasure`

```@docs
GridapTopOpt.IntegrandWithMeasure
GridapTopOpt.gradient
GridapTopOpt.jacobian
```