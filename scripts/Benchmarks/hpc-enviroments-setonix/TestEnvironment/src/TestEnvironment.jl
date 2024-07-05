module TestEnvironment

using DrWatson
using Gridap, GridapDistributed, GridapPETSc, GridapGmsh, GridapP4est
using PartitionedArrays

export OPTIONS_CG_JACOBI, OPTIONS_CG_AMG, OPTIONS_MUMPS, OPTIONS_NEUTON_MUMPS, OPTIONS_MINRES
export main_poisson, main_p4est

const OPTIONS_CG_JACOBI = "-pc_type jacobi -ksp_type cg -ksp_converged_reason -ksp_rtol 1.0e-10"
const OPTIONS_CG_AMG = "-pc_type gamg -ksp_type cg -ksp_converged_reason -ksp_rtol 1.0e-10"
const OPTIONS_MUMPS = "-pc_type lu -ksp_type preonly -ksp_converged_reason -pc_factor_mat_solver_type mumps"
const OPTIONS_MINRES = "-ksp_type minres -ksp_converged_reason -ksp_rtol 1.0e-10"

include("poisson_driver.jl")
include("p4est_driver.jl")

end # module TestEnvironment
