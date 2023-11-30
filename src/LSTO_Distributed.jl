module LSTO_Distributed

using Gridap
using Gridap.TensorValues
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.Helpers
using Gridap.ReferenceFEs
using GridapDistributed
using GridapPETSc
using PartitionedArrays
using SparseMatricesCSR
using ChainRulesCore
using DelimitedFiles
using BlockArrays
using LinearAlgebra

import Gridap.Algebra: LinearSolver, SymbolicSetup
import Gridap.Geometry: get_faces
import GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code
import GridapDistributed: DistributedDiscreteModel, DistributedTriangulation, 
    DistributedFESpace,DistributedDomainContribution, to_parray_of_arrays
import PartitionedArrays: getany, tuple_of_arrays

include("ChainRules.jl")
export PDEConstrainedFunctionals
export AffineFEStateMap
export NonlinearFEStateMap
export get_state
export evaluate_functionals!
export evaluate_derivatives!

include("Optimiser.jl")
export AbstractOptimiser
export get_optimiser_history
export get_level_set
export write_history
export AugmentedLagrangian

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

end