using PackageCompiler

create_sysimage([:GridapTopOpt],
  sysimage_path=joinpath(@__DIR__,"..","GridapTopOpt.so"),
  precompile_execution_file=joinpath(@__DIR__,"warmup.jl"))
