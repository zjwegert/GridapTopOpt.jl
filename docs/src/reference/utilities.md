# Utilities

## Ersatz material interpolation
```@docs
GridapTopOpt.SmoothErsatzMaterialInterpolation
```

## Mesh labelling
```@docs
GridapTopOpt.update_labels!
```

## Helpers

```@docs
GridapTopOpt.initial_lsf
GridapTopOpt.isotropic_elast_tensor
GridapTopOpt.get_cartesian_element_sizes
GridapTopOpt.get_element_diameters
GridapTopOpt.get_element_diameter_field
```

!!! warning "`get_cartesian_element_sizes` vs. `get_element_diameters`"
    Currently, we use different naming for returning the size of elements for
    a `CartesianDiscreteModel` and a general `DiscreteModel`.
    - The function `get_element_diameters` returns a list of element diameters and
      should be used for `TRI` and `TET` polytopes (this can be extended to
      `QUAD` and `HEX` if needed).
    - The legacy function `get_cartesian_element_sizes` returns a tuple of a
      single element size for a mesh with homogenous sizes.