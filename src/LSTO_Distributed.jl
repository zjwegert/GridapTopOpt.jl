module LSTO_Distributed

using MPI
using BlockArrays, SparseArrays, CircularArrays
using LinearAlgebra, SparseMatricesCSR
using ChainRulesCore
using DelimitedFiles, Printf

using Gridap
using Gridap.Helpers, Gridap.Algebra, Gridap.TensorValues
using Gridap.Geometry, Gridap.CellData, Gridap.Fields
using Gridap.ReferenceFEs, Gridap.FESpaces,  Gridap.MultiField
using Gridap.Geometry: get_faces

using GridapDistributed
using GridapDistributed: DistributedDiscreteModel, DistributedTriangulation, 
  DistributedFESpace, DistributedDomainContribution, to_parray_of_arrays,
  allocate_in_domain, DistributedCellField, DistributedMultiFieldFEBasis,
  BlockPMatrix, BlockPVector, change_ghost

using GridapPETSc, GridapPETSc.PETSC
using GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code

using PartitionedArrays
using PartitionedArrays: getany, tuple_of_arrays, matching_ghost_indices

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.NonlinearSolvers, GridapSolvers.BlockSolvers
using GridapSolvers.SolverInterfaces: SolverVerboseLevel, SOLVER_VERBOSE_NONE, SOLVER_VERBOSE_LOW, SOLVER_VERBOSE_HIGH

include("GridapExtensions.jl")

include("ChainRules.jl")
export PDEConstrainedFunctionals
export AffineFEStateMap
export NonlinearFEStateMap
export RepeatingAffineFEStateMap
export get_state
export evaluate_functionals!
export evaluate_derivatives!

include("Utilities.jl")
export gen_lsf
export get_Δ
export update_labels!
export isotropic_2d
export isotropic_3d
export make_dir
export print_history
export write_vtk

include("Advection.jl")
export AdvectionStencil
export FirstOrderStencil
export advect!
export reinit!

include("Solvers.jl")
export ElasticitySolver
export BlockDiagonalPreconditioner
export MUMPSSolver

include("VelocityExtension.jl")
export VelocityExtension
export project!

include("MaterialInterpolation.jl")
export SmoothErsatzMaterialInterpolation

include("Optimisers/Optimisers.jl")
export AbstractOptimiser
export get_history
export write_history
export AugmentedLagrangian
export HilbertianProjection
export HPModifiedGramSchmidt

include("Benchmarks.jl")
export benchmark_optimizer
export benchmark_forward_problem
export benchmark_advection
export benchmark_reinitialisation
export benchmark_velocity_extension

end