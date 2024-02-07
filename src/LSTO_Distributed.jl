module LSTO_Distributed

using MPI
using BlockArrays, SparseArrays, CircularArrays
using LinearAlgebra, SparseMatricesCSR
using ChainRulesCore
using DelimitedFiles, Printf
using ChaosTools

using Gridap
using Gridap.Helpers, Gridap.Algebra, Gridap.TensorValues
using Gridap.Geometry, Gridap.CellData, Gridap.Fields
using Gridap.ReferenceFEs, Gridap.FESpaces,  Gridap.MultiField
using Gridap.Geometry: get_faces
using Gridap: writevtk

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

using JLD2: save_object, load_object

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
export SmoothErsatzMaterialInterpolation
export update_labels!
export initial_lsf
export get_el_size
export isotropic_elast_tensor
export make_dir
export write_vtk
export gen_lsf # TODO: Remove
export get_Δ # TODO: Remove
export isotropic_2d # TODO: Remove
export isotropic_3d # TODO: Remove

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

include("Optimisers/Optimisers.jl")
export Optimiser # TODO: Remove from exports
export OptimiserHistory # TODO: Remove from exports
export get_history # TODO: Add get_niter
export write_history
export AugmentedLagrangian
export HilbertianProjection
export HPModifiedGramSchmidt # TODO: Remove from exports

include("Benchmarks.jl")
export benchmark_optimizer # TODO: Remove from exports
export benchmark_forward_problem # TODO: Remove from exports
export benchmark_advection # TODO: Remove from exports
export benchmark_reinitialisation # TODO: Remove from exports
export benchmark_velocity_extension # TODO: Remove from exports
export benchmark_hilbertian_projection_ma # TODO: Remove from exports

include("Io.jl")
export save, load, load!
export psave, pload, pload!

end