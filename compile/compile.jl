using PackageCompiler

create_sysimage([:LevelSetTopOpt],
  sysimage_path=joinpath(@__DIR__,"..","LevelSetTopOpt.so"),
  precompile_execution_file=joinpath(@__DIR__,"warmup.jl"))
