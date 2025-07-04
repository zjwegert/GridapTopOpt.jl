This page will be updated in the event that a breaking change is introduced into
the source code.

# Updating from v0.1 to v0.2
In v0.2 we made several quality of life changes and enabled compatability
with GridapEmbedded. Below we list any breaking changes that will require
changes to scripts implemented in v0.1:

- Automatic differentiation capability has now been added to GridapDistributed.
  As a result, the `IntegrandWithMeasure` structure has been removed. In addition
  functionals previously required the measures to be passed as arguments, e.g.,
  ```
  J(u,φ,dΩ,dΓ_N) = ∫(f(u,φ))dΩ + ∫(g(u,φ))dΓ_N
  ```
  This is no longer required and the above should instead be written as
  ```
  J(u,φ) = ∫(f(u,φ))dΩ + ∫(g(u,φ))dΓ_N
  ```

# Updating from v0.2 to v0.3
In v0.3 we added Zygote compatability for backwards differentiation in serial and
parallel. Below we list any breaking changes that will require changes to scripts implemented in
v0.2:

- `StateMaps` now always differentiate onto the space of the primal variable. See #76 for details.
  This introduces a breaking API change as `U_reg` is removed from constructors of `StateMaps`. E.g.,
  ```julia
    AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh)
  ```
  becomes
  ```julia
    AffineFEStateMap(a,l,U,V,V_φ,φh)
  ```
  Backwards compatability has been added here to ensure that the old API still works, however `U_reg` will not be used.
- The way that we allocate vectors for distributed has been reworked. Now, we always create derivatives using `zero(<:FESpace)`, we then move this to the correct type of array when required. For example, if needing to interpolate onto a RHS vector (this doesn't have ghosts in distributed), we can use the functionality `_interpolate_onto_rhs!`. This change cleans up a lot of the constructors for `<:StateMap` and `StateMapWithParam`. **Any** code implemented that relies on the old approach for allocating vectors should be adjusted accordingly.