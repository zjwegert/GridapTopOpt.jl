# Polytopal Cutters and Automatic Shape Differentiation

The polytopal geometric differentiation capabilities are based on the following work:

!!! note "Reference"
    Zachary J. Wegert, Martin Berggren, and Vivien J. Challis (2026). "Shape calculus and automatic differentiation for multi-phase level-set topology optimisation with unfitted finite elements".

To see examples of usage, please refer to the tests in `test/seq/PolytopalCuttersTests/`. Additional shape and topology optimisation examples can be found [here](https://github.com/zjwegert/Wegert_et_al_2026_MP).

## Single level-set function
For a single level-set function, please refer to the methods available in [GridapEmbedded](https://gridap.github.io/GridapEmbedded.jl/stable/GeometricalDerivatives/#Geometrical-Derivatives).

## Polytopal cutters

```@docs
GridapTopOpt.PolytopalLevelSetCutter
GridapTopOpt.DiscreteGeometryFromFEFunction
```

## Automatic shape differentiation

```@docs
GridapTopOpt.DifferentiableCutPolyTriangulation
```

## Example
See `scripts/Examples/UnfittedMultiphase/AD_Example.jl` for an example of basic usage.