module GridapTopOpt

using GridapPETSc, GridapPETSc.PETSC
using GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code

using MPI
using BlockArrays, CircularArrays, FillArrays
using LinearAlgebra
using ChainRulesCore
using DelimitedFiles, Printf

using Gridap
using Gridap.Helpers, Gridap.Algebra, Gridap.TensorValues
using Gridap.Geometry, Gridap.CellData, Gridap.Fields, Gridap.Arrays
using Gridap.ReferenceFEs, Gridap.FESpaces,  Gridap.MultiField

using Gridap.Geometry: get_faces, num_nodes
using Gridap.FESpaces: get_assembly_strategy
using Gridap: writevtk

using GridapDistributed
using GridapDistributed: DistributedDiscreteModel, DistributedTriangulation,
  DistributedFESpace, DistributedDomainContribution, to_parray_of_arrays,
  allocate_in_domain, DistributedCellField, DistributedMultiFieldCellField,
  DistributedMultiFieldFEBasis, BlockPMatrix, BlockPVector, change_ghost

using PartitionedArrays
using PartitionedArrays: getany, tuple_of_arrays, matching_ghost_indices

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.NonlinearSolvers, GridapSolvers.BlockSolvers
using GridapSolvers.SolverInterfaces: SolverVerboseLevel, SOLVER_VERBOSE_NONE, SOLVER_VERBOSE_LOW, SOLVER_VERBOSE_HIGH

using GridapEmbedded
using GridapEmbedded.LevelSetCutters, GridapEmbedded.Interfaces
using GridapEmbedded.Interfaces: SubFacetData, SubCellTriangulation, SubFacetTriangulation

using JLD2: save_object, load_object, jldsave

import Base: +

include("GridapExtensions.jl")

include("ChainRules.jl")
export PDEConstrainedFunctionals
export AffineFEStateMap
export NonlinearFEStateMap
export RepeatingAffineFEStateMap
export get_state
export evaluate_functionals!
export evaluate_derivatives!

include("Embedded/Embedded.jl")
export DifferentiableTriangulation
export SubFacetBoundaryTriangulation, SubFacetSkeletonTriangulation
export EmbeddedCollection, update_collection!, add_recipe!

include("Utilities.jl")
export SmoothErsatzMaterialInterpolation
export update_labels!
export initial_lsf
export get_el_Δ
export isotropic_elast_tensor

include("LevelSetEvolution/LevelSetEvolution.jl")
export HamiltonJacobiEvolution
export FirstOrderStencil
export evolve!
export reinit!

include("Solvers.jl")
export ElasticitySolver

include("VelocityExtension.jl")
export VelocityExtension
export project!

include("Optimisers/Optimisers.jl")
export get_history
export write_history
export AugmentedLagrangian
export HilbertianProjection

include("Benchmarks.jl")

include("Io.jl")
export save, load, load!
export psave, pload, pload!

end