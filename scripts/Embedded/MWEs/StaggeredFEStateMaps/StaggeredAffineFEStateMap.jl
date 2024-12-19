using GridapTopOpt
using Gridap
using Gridap.Algebra, Gridap.Geometry, Gridap.MultiField, Gridap.Arrays,
  Gridap.FESpaces, Gridap.CellData
using GridapTopOpt: AbstractFEStateMap

using BlockArrays
using GridapSolvers
using GridapSolvers.BlockSolvers, GridapSolvers.LinearSolvers, GridapSolvers.NonlinearSolvers
using GridapSolvers.BlockSolvers: combine_fespaces, get_solution

using LinearAlgebra
using ChainRulesCore

import GridapTopOpt: forward_solve!,dRdφ,update_adjoint_caches!,adjoint_solve!,get_state,get_spaces,get_assemblers,
  get_trial_space,get_test_space,pullback

struct StaggeredAffineFEStateMap{NB,SB,A,B,C,D,E,F} <: AbstractFEStateMap
  biforms    :: Vector{<:Function}
  liforms    :: Vector{<:Function}
  spaces     :: A
  assems     :: B
  solvers    :: C
  plb_caches :: D
  fwd_caches :: E
  adj_caches :: F

  function StaggeredAffineFEStateMap(
      op              :: StaggeredAffineFEOperator{NB,SB},
      V_φ             :: FESpace,
      U_reg           :: FESpace,
      φh;
      assem_deriv     :: Assembler = SparseMatrixAssembler(U_reg,U_reg),
      solver         :: StaggeredFESolver{NB} = StaggeredFESolver(fill(LUSolver(),length(op.biforms))),
      adjoint_solver :: StaggeredFESolver{NB} = StaggeredFESolver(fill(LUSolver(),length(op.biforms)))
    ) where {NB,SB}

    ## Pullback cache (this is a temporary solution before we refactor ChainRules)
    uhd = zero(op.trial)
    xhs, λᵀs_∂Rs∂φ = (), ()
    for k in 1:NB
      xh_k = get_solution(op,uhd,k)
      _a(uk,vk,φh) = op.biforms[k](xhs,uk,vk,φh)
      _l(vk,φh) = op.liforms[k](xhs,vk,φh)
      λᵀk_∂Rk∂φ = ∇((uk,vk,φh) -> _a(uk,vk,φh) - _l(vk,φh),[xh_k,xh_k,φh],3)
      xhs, λᵀs_∂Rs∂φ = (xhs...,xh_k), (λᵀs_∂Rs∂φ...,λᵀk_∂Rk∂φ)
    end
    vecdata = collect_cell_vector(U_reg,sum(λᵀs_∂Rs∂φ))
    Σ_λᵀs_∂Rs∂φ = allocate_vector(assem_deriv,vecdata)
    plb_caches = (Σ_λᵀs_∂Rs∂φ,assem_deriv)

    ## Forward cache
    op_at_φ = get_staggered_operator_at_φ(op,φh)
    xh_one = one(op.trial)
    op_cache = _instantiate_caches(xh_one,solver,op_at_φ)
    fwd_caches = (zero_free_values(op.trial),op.trial,op_cache,op_at_φ)

    ## Adjoint cache
    xh_one_comb = _get_solutions(op_at_φ,xh_one)
    op_adjoint = dummy_generate_adjoint_operator(op_at_φ,xh_one_comb)
    op_cache = _instantiate_caches(xh_one,adjoint_solver,op_adjoint)
    adj_caches = (zero_free_values(op_adjoint.trial),op_adjoint.trial,op_cache,op_adjoint)

    spaces = (;trial=op_at_φ.trial,test=op_at_φ.test,aux_space=V_φ,deriv_space=U_reg,trials=op_at_φ.trials,tests=op_at_φ.tests)
    assems = (;assems=op_at_φ.assems,assem_deriv,adjoint_assems=op_adjoint.assems)
    _solvers = (;solver,adjoint_solver)
    A,B,C,D,E,F = typeof(spaces), typeof(assems), typeof(_solvers),
      typeof(plb_caches), typeof(fwd_caches), typeof(adj_caches)
    new{NB,SB,A,B,C,D,E,F}(op.biforms,op.liforms,spaces,assems,_solvers,plb_caches,fwd_caches,adj_caches)
  end
end

get_state(m::StaggeredAffineFEStateMap) = FEFunction(m.fwd_caches[2],m.fwd_caches[1])
get_spaces(m::StaggeredAffineFEStateMap) = m.spaces
get_assemblers(m::StaggeredAffineFEStateMap) = m.assems

function forward_solve!(φ_to_u::StaggeredAffineFEStateMap,φ::AbstractVector)
  φh = FEFunction(GridapTopOpt.get_aux_space(φ_to_u),φ)
  return forward_solve!(φ_to_u,φh)
end

function forward_solve!(φ_to_u::StaggeredAffineFEStateMap,φh)
  solvers = φ_to_u.solvers
  x, X, cache, _ = φ_to_u.fwd_caches

  op = get_staggered_operator_at_φ(φ_to_u,φh)
  solve!(FEFunction(X,x),solvers.solver,op,cache);
  return x
end

function dRdφ(φ_to_u::StaggeredAffineFEStateMap{NB},uh,λh,φh) where NB
  biforms, liforms = φ_to_u.biforms, φ_to_u.liforms
  _,_,_,init_op = φ_to_u.fwd_caches
  _,_,_,init_adjoint_op = φ_to_u.adj_caches
  xhs, ∂Rs∂φ = (), ()
  for k in 1:NB
    xh_k = get_solution(init_op,uh,k)
    λh_k = get_solution(init_adjoint_op,λh,NB-k+1)
    _a(uk,vk,φh) = biforms[k](xhs,uk,vk,φh)
    _l(vk,φh) = liforms[k](xhs,vk,φh)
    ∂Rk∂φ = ∇((uk,vk,φh) -> _a(uk,vk,φh) - _l(vk,φh),[xh_k,λh_k,φh],3)
    xhs, ∂Rs∂φ = (xhs...,xh_k), (∂Rs∂φ...,∂Rk∂φ)
  end
  return ∂Rs∂φ
end