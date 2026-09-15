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

if TESTCASE ∈ ("all", "seq", "seq-polycut")
  include("seq/PolytopalCuttersTests/runtests.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-polycut-analytic1")
  include("seq/PolytopalCuttersTests/runtests1.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-polycut-analytic2-2d")
  include("seq/PolytopalCuttersTests/runtests2_2d.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-polycut-analytic2-3d-vol")
  include("seq/PolytopalCuttersTests/runtests2_3d_vol.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-polycut-analytic2-3d-boundary-caseB")
  include("seq/PolytopalCuttersTests/runtests2_3d_boundary_caseB.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-polycut-analytic2-3d-boundary-caseC")
  include("seq/PolytopalCuttersTests/runtests2_3d_boundary_caseC.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-polycut-analytic2-3d-boundary-MF")
  include("seq/PolytopalCuttersTests/runtests2_3d_boundary_MF.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-polycut-analytic3-2d")
  include("seq/PolytopalCuttersTests/runtests3_2d.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-polycut-analytic3-3d-vol")
  include("seq/PolytopalCuttersTests/runtests3_3d_vol.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-polycut-analytic3-3d-boundary-1")
  include("seq/PolytopalCuttersTests/runtests3_3d_boundary_1.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-polycut-analytic3-3d-boundary-2")
  include("seq/PolytopalCuttersTests/runtests3_3d_boundary_2.jl")
end

if TESTCASE ∈ ("all", "seq", "seq-polycut-analytic3-3d-boundary-3")
  include("seq/PolytopalCuttersTests/runtests3_3d_boundary_3.jl")
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

if TESTCASE ∈ ("all", "mpi", "mpi-gridaptopopt-unfitted")
  include("mpi/GridapTopOptTests_unfitted/runtests.jl")
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

if TESTCASE ∈ ("all", "mpi", "mpi-polycut")
  include("mpi/PolytopalCuttersTests/runtests.jl")
end

# Extensions

if TESTCASE ∈ ("all", "extlibs")
  include("mpi/GridapTopOptExtLibTests/runtests.jl")
end
