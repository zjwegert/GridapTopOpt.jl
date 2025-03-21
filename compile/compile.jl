using PackageCompiler
using Pkg

pkgs = Symbol[]
append!(pkgs, [Symbol(v.name) for v in values(Pkg.dependencies()) if v.is_direct_dep],)

create_sysimage([:GridapTopOpt],#pkgs,
  sysimage_path=joinpath(@__DIR__,"..","GridapTopOpt.so"),
  precompile_execution_file=joinpath(@__DIR__,"warmup.jl"),
  #include_transitive_dependencies=false,
  # sysimage_build_args=`-O3 --check-bounds=no`
  )
