# IO
In GridapTopOpt, the usual IO from [Gridap](https://github.com/gridap/Gridap.jl/) is available. In addition, we also implement the below IO for convenience.

## Optimiser history
```@docs
GridapTopOpt.write_history
```

## Object IO in serial
```@docs
GridapTopOpt.save
GridapTopOpt.load
GridapTopOpt.load!
```

## Object IO in parallel
```@docs
GridapTopOpt.psave
GridapTopOpt.pload
GridapTopOpt.pload!
```