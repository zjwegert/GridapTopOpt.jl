# Embedded

!!! warning
    This page is still under construction! Please refer to the tutorial
    for a discussion of functionality.

## Isolated volumes
```@autodocs
Modules = [GridapTopOpt]
Pages = ["Embedded/IsolatedVolumes.jl",
  "Embedded/PolytopalCutters.jl"]
```

## `EmbeddedCollection` and `EmbeddedPDEConstrainedFunctionals`
```@docs
GridapTopOpt.EmbeddedCollection
```

We implement special structures that update the state map etc. on the fly using
`EmbeddedPDEConstrainedFunctionals`:

```@docs
GridapTopOpt.EmbeddedCollection_in_φh
GridapTopOpt.EmbeddedPDEConstrainedFunctionals
GridapTopOpt.evaluate_functionals!(pcf::EmbeddedPDEConstrainedFunctionals,φh;update_space::Bool=true)
GridapTopOpt.evaluate_derivatives!(pcf::EmbeddedPDEConstrainedFunctionals,φh;update_space::Bool=true)
GridapTopOpt.evaluate!(pcf::EmbeddedPDEConstrainedFunctionals,φh;update_space::Bool=true)
```

## Automatic shape differentiation
Automatic shape differentiation has been moved to [GridapEmbedded](https://gridap.github.io/GridapEmbedded.jl/stable/GeometricalDerivatives/#Geometrical-Derivatives).