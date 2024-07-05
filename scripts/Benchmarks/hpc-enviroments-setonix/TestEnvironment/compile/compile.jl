using PackageCompiler

create_sysimage([:TestEnvironment],
  sysimage_path=joinpath(@__DIR__,"..","TestEnvironment.so"),
  precompile_execution_file=joinpath(@__DIR__,"warmup.jl"))
