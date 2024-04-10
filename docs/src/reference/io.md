# IO
In LevelSetTopOpt, the usual IO from [Gridap](https://github.com/gridap/Gridap.jl/) is available. In addition, we also implement the below IO for convenience.

## Optimiser history
```@docs
LevelSetTopOpt.write_history
```

## Object IO in serial
```@docs
LevelSetTopOpt.save
LevelSetTopOpt.load
LevelSetTopOpt.load!
```

## Object IO in parallel
```@docs
LevelSetTopOpt.psave
LevelSetTopOpt.pload
LevelSetTopOpt.pload!
```