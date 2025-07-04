# Zygote & GridapTopOpt Compatability

As of v0.3.0, Zygote can be used for backwards AD in GridapTopOpt in serial
and distributed.

```@docs
GridapTopOpt.CustomPDEConstrainedFunctionals
```

## Staggered-type problems
Staggered-type problems can be handled purely with Zygote and the other
existing StateMap implementations. This is preferred over the StaggeredStateMap
implementations.

For example, we can solve a problem where the second FE problem depends on the first
via the following:
```julia
## Weak forms
a1(u1,v1,φ) = ...
l1(v1,φ) = ...
# Treat (u1,φ) as the primal variable
a2(u2,v2,(u1,φ)) = ...
l2(v2,(u1,φ)) = ...

## Build StateMaps
φ_to_u1 = AffineFEStateMap(a1,l1,U1,V,V_φ,φh)
# u1φ_to_u2 has a MultiFieldFESpace V_u1φ of primal vars
u1φ_to_u2 = AffineFEStateMap(a2,l2,U2,V,V_u1φ,interpolate([1,φh],V_u1φ))
# The StateParamMap F needs to take a MultiFieldFEFunction u1u2h ∈ U_u1u2
F = GridapTopOpt.StateParamMap(F,U_u1u2,V_φ,assem_U_u1u2,assem_V_φ)

function φ_to_j(φ)
  u1 = φ_to_u1(φ)
  u1φ = combine_fields(V_u1φ,u1,φ) # Combine vectors of DOFs
  u2 = u1φ_to_u2(u1φ)
  u1u2 = combine_fields(U_u1u2,u1,u2)
  F(u1u2,φ)
end

pcf = CustomPDEConstrainedFunctionals(...)
```

## GridapTopOpt + GridapEmbedded + Zygote
```@docs
GridapTopOpt.CustomEmbeddedPDEConstrainedFunctionals
```
