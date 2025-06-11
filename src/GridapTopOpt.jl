__precompile__(false)
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
  gradient, jacobian, DistributedMultiFieldFEFunction, DistributedSingleFieldFEFunction

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

using Zygote

using JLD2: save_object, load_object, jldsave

import Base: +
import Gridap: solve!
import PartitionedArrays: default_find_rcv_ids
import GridapDistributed: remove_ghost_cells

__init__() = begin
  include((@__DIR__)*"/GridapExtensions.jl")
  include((@__DIR__)*"/LevelSetEvolution/UnfittedEvolution/MutableRungeKutta.jl") # <- commented out in "LevelSetEvolution/LevelSetEvolution.jl"

  function default_find_rcv_ids(::MPIArray)
    PartitionedArrays.find_rcv_ids_ibarrier
  end
end

include("Embedded/Embedded.jl")
export EmbeddedCollection, update_collection!, add_recipe!
export CUT
export get_isolated_volumes_mask_polytopal

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
export HamiltonJacobiEvolution
export FirstOrderStencil
export UnfittedFEEvolution
export CutFEMEvolve
export StabilisedReinit
export ArtificialViscosity
export InteriorPenalty
export evolve!
export reinit!

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

# using PrecompileTools: @setup_workload, @compile_workload, @recompile_invalidations

# @setup_workload begin
#   np = (2,1)
#   ranks = DebugArray([1,2])

#   model = UnstructuredDiscreteModel(CartesianDiscreteModel(ranks,np,(0,1,0,1),(4,4)))

#   reffe_m = ReferenceFE(lagrangian,Float64,1)
#   M = FESpace(model,reffe_m)

#   ls(x) = ifelse(x[1] > 0.8,-1.0,1.0)
#   φh = interpolate(ls,M)

#   geo = DiscreteGeometry(φh,model)
#   cutgeo = cut(model,geo)
#   cutgeo_facets = cut_facets(model,geo)

#   order = 1
#   degree = 2order
#   Ω_act = Triangulation(model)
#   dΩ_act = Measure(Ω_act,degree)
#   Ωs = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),M)
#   Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),M)
#   Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),M)

#   # Ωs = Triangulation(cutgeo,PHYSICAL)
#   # Ωf = Triangulation(cutgeo,PHYSICAL_OUT)
#   # Γ  = EmbeddedBoundary(cutgeo)
#   Γg = GhostSkeleton(cutgeo)
#   Ω_act_s = Triangulation(cutgeo,ACTIVE)
#   Ω_act_f = Triangulation(cutgeo,ACTIVE_OUT)
#   Γi = SkeletonTriangulation(cutgeo_facets,ACTIVE_OUT)
#   dΩs      = Measure(Ωs,degree)
#   dΩf      = Measure(Ωf,degree)
#   dΓg      = Measure(Γg,degree)
#   n_Γg     = get_normal_vector(Γg)
#   dΓ       = Measure(Γ,degree)
#   n_Γ      = get_normal_vector(Γ);
#   dΩ_act_s = Measure(Ω_act_s,degree)
#   dΩ_act_f = Measure(Ω_act_f,degree)
#   dΓi      = Measure(Γi,degree)
#   n_Γi     = get_normal_vector(Γi)

#   # Setup spaces
#   reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
#   reffe_p = ReferenceFE(lagrangian,Float64,order-1)
#   reffe_d = ReferenceFE(lagrangian,VectorValue{2,Float64},order)

#   # Test spaces
#   V = TestFESpace(Ω_act_f,reffe_u,conformity=:H1)
#   Q = TestFESpace(Ω_act_f,reffe_p,conformity=:L2)
#   T = TestFESpace(Ω_act_s,reffe_d,conformity=:H1)

#   # Trial spaces
#   U = TrialFESpace(V)
#   P = TrialFESpace(Q)
#   R = TrialFESpace(T)

#   # Multifield spaces
#   UP = MultiFieldFESpace([U,P])
#   VQ = MultiFieldFESpace([V,Q])

#   ### Weak form
#   ## Fluid
#   # Properties
#   μ = 1.0

#   # Stabilization parameters
#   α_Nu = 100
#   α_u  = 0.1
#   α_p  = 0.25

#   # Stabilization functions
#   hₕ = CellField(1,Ω_act)

#   γ_Nu(h) = α_Nu*μ/h
#   γ_u(h) = α_u*μ*h
#   γ_p(h) = α_p*h/μ
#   k_p    = 1.0
#   γ_Nu_h = γ_Nu ∘ hₕ
#   γ_u_h = mean(γ_u ∘ hₕ)
#   γ_p_h = mean(γ_p ∘ hₕ)

#   # Terms
#   σf_n(u,p,n) = μ*∇(u) ⋅ n - p*n
#   a_Ω(∇u,∇v) = μ*(∇u ⊙ ∇v)
#   b_Ω(div_v,p) = -p*(div_v)
#   a_Γ(u,∇u,v,∇v,n) = - μ*n⋅(∇u ⋅ v + ∇v⋅ u) + γ_Nu_h*(u⋅v)
#   b_Γ(v,p,n) = (n⋅v)*p
#   ju(∇u,∇v) = γ_u_h*(jump(n_Γg ⋅ ∇u) ⋅ jump(n_Γg ⋅ ∇v))
#   jp(p,q) = γ_p_h*(jump(p) * jump(q))

#   function a_fluid((),(u,p),(v,q),φ)
#     ∇u = ∇(u); ∇v = ∇(v);
#     div_u = ∇⋅u; div_v = ∇⋅v
#     n_Γ = -get_normal_vector(Γ)
#     return ∫(a_Ω(∇u,∇v) + b_Ω(div_v,p) + b_Ω(div_u,q))dΩf +
#       ∫(a_Γ(u,∇u,v,∇v,n_Γ) + b_Γ(v,p,n_Γ) + b_Γ(u,q,n_Γ))dΓ +
#       ∫(ju(∇u,∇v))dΓg - ∫(jp(p,q))dΓi
#   end

#   l_fluid((),(v,q),φ) =  ∫(0q)dΩf
#   xdh = zero(UP);

#   ## Structure
#   _I = one(SymTensorValue{2,Float64})
#   # Material parameters
#   function lame_parameters(E,ν)
#     λ = (E*ν)/((1+ν)*(1-2*ν))
#     μ = E/(2*(1+ν))
#     (λ, μ)
#   end
#   λs, μs = lame_parameters(0.1,0.05)
#   # Stabilization
#   α_Gd = 1e-3
#   k_d = 1.0
#   γ_Gd(h) = α_Gd*(λs + μs)*h^3
#   γ_Gd_h = mean(γ_Gd ∘ hₕ)
#   # Terms
#   σ(ε) = λs*tr(ε)*_I + 2*μs*ε
#   a_s_Ω(d,s) = ε(s) ⊙ (σ ∘ ε(d)) # Elasticity
#   j_s_k(d,s) = γ_Gd_h*(jump(n_Γg ⋅ ∇(s)) ⋅ jump(n_Γg ⋅ ∇(d)))
#   v_s_ψ(d,s) = (k_d*ψ_s)*(d⋅s) # Isolated volume term

#   function a_solid(((u,p),),d,s,φ)
#     return ∫(a_s_Ω(d,s))dΩs +
#       ∫(j_s_k(d,s))dΓg
#   end
#   function l_solid(((u,p),),s,φ)
#     n = -get_normal_vector(Γ)
#     return ∫(-σf_n(u,p,n) ⋅ s)dΓ
#   end

#   d0h = zero(R);
#   @compile_workload begin
#     λᵀ1_∂R1∂φ = ∇(φ -> a_fluid((),xdh,xdh,φ) - l_fluid((),xdh,φ),φh)
#     vecdata = collect_cell_vector(M,λᵀ1_∂R1∂φ)
#     assem_deriv = SparseMatrixAssembler(M,M)
#     Σ_λᵀs_∂Rs∂φ = allocate_vector(assem_deriv,vecdata)

#     λᵀ2_∂R2∂φ = ∇(φ -> a_solid((xdh,),d0h,d0h,φ) - l_solid((xdh,),d0h,φ),φh);
#     vecdata2 = collect_cell_vector(M,λᵀ2_∂R2∂φ);
#     Σ_λᵀ2_∂R2∂φ = allocate_vector(assem_deriv,vecdata2);
#   end
# end

end
