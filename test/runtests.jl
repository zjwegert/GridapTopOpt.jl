using GridapTopOpt
using Test

TESTCASE = get(ENV, "TESTCASE", "seq")

# Sequential tests

if TESTCASE ∈ ("all", "seq", "seq-embedded")
  include("seq/EmbeddedTests/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-evolvers")
  include("seq/EvolverTests/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-gridaptopopt")
  include("seq/GridapTopOptTests/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-reinitialisers")
  include("seq/ReinitialiserTests/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-statemaps")
  include("seq/StateMapTests/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-utility")
  include("seq/UtilityTests/runtests.jl")
end

# MPI tests

if TESTCASE ∈ ("all", "mpi", "mpi-embedded")
  include("mpi/EmbeddedTests/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-evolvers")
  include("mpi/EvolverTests/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-gridaptopopt")
  include("mpi/GridapTopOptTests/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-reinitialisers")
  include("mpi/ReinitialiserTests/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-statemaps")
  include("mpi/StateMapTests/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-utility")
  include("mpi/UtilityTests/runtests.jl")
end

if TESTCASE ∈ ("all", "mpi", "mpi-velext")
  include("mpi/VelocityExtensionTests/runtests.jl")
end

# Extensions

if TESTCASE ∈ ("all", "extlibs")
  include("mpi/GridapTopOptExtLibTests/runtests.jl")
end
