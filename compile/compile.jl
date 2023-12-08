using PackageCompiler

create_sysimage([:LSTO_Distributed],
  sysimage_path=joinpath(@__DIR__,"..","LSTO_Distributed.so"),
  precompile_execution_file=joinpath(@__DIR__,"warmup.jl"))
