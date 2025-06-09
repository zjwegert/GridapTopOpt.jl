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