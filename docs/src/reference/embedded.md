!!! warning
    This page is still under construction! Please refer to the tutorial
    for a discussion of functionality.

# Level-set evolution and reinitalisation
```@autodocs
Modules = [GridapTopOpt]
Pages = ["LevelSetEvolution/UnfittedEvolution/UnfittedEvolution.jl",
  "LevelSetEvolution/UnfittedEvolution/CutFEMEvolve.jl",
  "LevelSetEvolution/UnfittedEvolution/StabilisedReinit.jl"]
```

# Isolated volumes
```@autodocs
Modules = [GridapTopOpt]
Pages = ["Embedded/IsolatedVolumes.jl",
  "Embedded/PolytopalCutters.jl"]
```

# `EmbeddedCollection` and `EmbeddedPDEConstrainedFunctionals`
```@docs
GridapTopOpt.EmbeddedCollection
```

We implement special structures that update the state map etc. on the fly using
`EmbeddedPDEConstrainedFunctionals`:

```@docs
GridapTopOpt.EmbeddedPDEConstrainedFunctionals
GridapTopOpt.evaluate_functionals!(pcf::EmbeddedPDEConstrainedFunctionals,φh;update_space::Bool=true)
GridapTopOpt.evaluate_derivatives!(pcf::EmbeddedPDEConstrainedFunctionals,φh;update_space::Bool=true)
GridapTopOpt.evaluate!(pcf::EmbeddedPDEConstrainedFunctionals,φh;update_space::Bool=true)
GridapTopOpt.EmbeddedCollection_in_φh
```

# Shape derivatives
## Automatic differentation
```@docs
GridapTopOpt.DifferentiableTriangulation
```

## Analytic derivatives
Examples of calculating analytic derivatives can be found in
`test\seq\EmbeddedTests\EmbeddedDifferentiationTests.jl`.