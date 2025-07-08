module GridapTopOpt

using GridapPETSc, GridapPETSc.PETSC
using GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code

using MPI
using BlockArrays, CircularArrays, FillArrays
using LinearAlgebra
using ChainRulesCore
using DelimitedFiles, Printf
using DataStructures

using Gridap
using Gridap.Helpers, Gridap.Algebra, Gridap.TensorValues
using Gridap.Geometry, Gridap.CellData, Gridap.Fields, Gridap.Arrays
using Gridap.ReferenceFEs, Gridap.FESpaces,  Gridap.MultiField, Gridap.Polynomials

using Gridap.Geometry: get_faces, num_nodes, TriangulationView
using Gridap.FESpaces: get_assembly_strategy
using Gridap.ODEs: ODESolver
using Gridap: writevtk

using GridapDistributed
using GridapDistributed: DistributedDiscreteModel, DistributedTriangulation,
  DistributedFESpace, DistributedDomainContribution, to_parray_of_arrays,
  allocate_in_domain, DistributedCellField, DistributedMultiFieldCellField,
  DistributedMultiFieldFEBasis, BlockPMatrix, BlockPVector, change_ghost,
  DistributedMultiFieldFEFunction, DistributedSingleFieldFEFunction, DistributedMultiFieldFESpace

using PartitionedArrays
using PartitionedArrays: getany, tuple_of_arrays, matching_ghost_indices

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.NonlinearSolvers, GridapSolvers.BlockSolvers
using GridapSolvers.SolverInterfaces: SolverVerboseLevel, SOLVER_VERBOSE_NONE, SOLVER_VERBOSE_LOW, SOLVER_VERBOSE_HIGH,
  SOLVER_CONVERGED_ATOL, SOLVER_CONVERGED_RTOL, ConvergenceLog, finished_flag
using GridapSolvers.BlockSolvers: combine_fespaces, get_solution

using GridapEmbedded
using GridapEmbedded.LevelSetCutters, GridapEmbedded.Interfaces
using GridapEmbedded.Interfaces: SubFacetData, SubCellTriangulation, SubFacetTriangulation
using GridapEmbedded.LevelSetCutters: DifferentiableTriangulation

using Zygote
using JLD2: save_object, load_object, jldsave

import Base: +
import Gridap: solve!
import PartitionedArrays: default_find_rcv_ids
import GridapDistributed: remove_ghost_cells

__init__() = begin
  include((@__DIR__)*"/Extensions/GridapExtensions.jl")
  include((@__DIR__)*"/LevelSetEvolution/Utilities/MutableRungeKutta.jl")

  function default_find_rcv_ids(::MPIArray)
    PartitionedArrays.find_rcv_ids_ibarrier
  end
end

include("Embedded/Embedded.jl")
export EmbeddedCollection, update_collection!, add_recipe!
export EmbeddedCollection_in_φh
export CUT
export get_isolated_volumes_mask_polytopal
export DifferentiableTriangulation

include("StateMaps/StateMaps.jl")
export PDEConstrainedFunctionals
export EmbeddedPDEConstrainedFunctionals
export CustomPDEConstrainedFunctionals
export CustomEmbeddedPDEConstrainedFunctionals
export AffineFEStateMap
export NonlinearFEStateMap
export RepeatingAffineFEStateMap
export StaggeredAffineFEStateMap
export StaggeredNonlinearFEStateMap
export get_state
export evaluate_functionals!
export evaluate_derivatives!

include("Utilities.jl")
export SmoothErsatzMaterialInterpolation
export update_labels!
export initial_lsf
export get_el_Δ
export get_cartesian_element_sizes
export get_element_diameters
export get_element_diameter_field
export isotropic_elast_tensor

include("LevelSetEvolution/LevelSetEvolution.jl")
export LevelSetEvolution
export evolve!
export reinit!
export FirstOrderStencil
export FiniteDifferenceEvolver
export FiniteDifferenceReinitialiser
export CutFEMEvolver
export IdentityReinitialiser
export StabilisedReinitialiser
export MultiStageStabilisedReinitialiser
export ArtificialViscosity
export InteriorPenalty
export HeatReinitialiser

# For backwards compat/errors
export CutFEMEvolve
export StabilisedReinit
export HamiltonJacobiEvolution
export UnfittedFEEvolution

include("Solvers.jl")
export ElasticitySolver

include("VelocityExtension.jl")
export VelocityExtension
export IdentityVelocityExtension
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

include("Extensions/ZygoteExtensions.jl")
export combine_fields

end