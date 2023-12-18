module LSTO_Distributed

using Gridap
using Gridap.TensorValues, Gridap.Geometry, Gridap.FESpaces, Gridap.Helpers,
  Gridap.ReferenceFEs, Gridap.Algebra, Gridap.CellData, Gridap.MultiField
using GridapDistributed
using GridapPETSc, GridapPETSc.PETSC
using PartitionedArrays
using MPI
using BlockArrays
using SparseArrays
using SparseMatricesCSR
using ChainRulesCore
using DelimitedFiles
using BlockArrays
using LinearAlgebra
using CircularArrays
using Printf

import Gridap.solve!
import Gridap.FESpaces: AffineFEOperator, assemble_matrix_and_vector, 
  assemble_matrix!, assemble_matrix, allocate_vector, assemble_vector,
  assemble_vector!
import Gridap.Algebra: LinearSolver, SymbolicSetup, NonlinearSolver
import Gridap.Geometry: get_faces
import GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code
import GridapDistributed: DistributedDiscreteModel, DistributedTriangulation, 
  DistributedFESpace, DistributedDomainContribution, to_parray_of_arrays,
  allocate_in_domain, DistributedCellField, DistributedMultiFieldFEBasis
import PartitionedArrays: getany, tuple_of_arrays

include("ChainRules.jl")
export PDEConstrainedFunctionals
export AffineFEStateMap
export NonlinearFEStateMap
export get_state
export evaluate_functionals!
export evaluate_derivatives!

include("Optimisers/Optimisers.jl")
export AbstractOptimiser
export get_optimiser_history
export get_level_set
export write_history
export AugmentedLagrangian
export HilbertianProjection
export HPModifiedGramSchmidt

include("Utilities.jl")
export gen_lsf
export get_Δ
export update_labels!
export isotropic_2d
export isotropic_3d
export make_dir
export print_history
export write_vtk

include("DiagonalBlockMatrixAssembler.jl")
export DiagonalBlockMatrixAssembler

include("Advection.jl")
export AdvectionStencil
export FirstOrderStencil
export advect!
export reinit!

include("Solvers.jl")
export ElasticitySolver
export BlockDiagonalPreconditioner
export MUMPSSolver
export NRSolver

include("VelocityExtension.jl")
export VelocityExtension
export project!

include("MaterialInterpolation.jl")
export SmoothErsatzMaterialInterpolation

end