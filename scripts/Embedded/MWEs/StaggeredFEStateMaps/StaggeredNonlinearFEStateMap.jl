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

# This is mutable for now, in future we will refactor ChainRules to remove storage of caches
mutable struct StaggeredNonlinearFEStateMap{NB,SB,A,B,C,D,E,F} <: AbstractFEStateMap
  const residuals  :: Vector{<:Function}
  const jacobians  :: Vector{<:Function}
  const spaces     :: A
  const assems     :: B
  const solvers    :: C
  const plb_caches :: D
  fwd_caches       :: E
  const adj_caches :: F

  function StaggeredNonlinearFEStateMap(
      op             :: StaggeredNonlinearFEOperator{NB,SB},
      V_φ            :: FESpace,
      U_reg          :: FESpace,
      φh;
      assem_deriv    :: Assembler = SparseMatrixAssembler(U_reg,U_reg),
      solver         :: StaggeredFESolver{NB} = StaggeredFESolver(
        fill(NewtonSolver(LUSolver();maxiter=50,rtol=1.e-8,verbose=true),length(op.residuals))),
      adjoint_solver :: StaggeredFESolver{NB} = StaggeredFESolver(fill(LUSolver(),length(op.residuals)))
    ) where {NB,SB}

    ## Pullback cache (this is a temporary solution before we refactor ChainRules)
    uhd = zero(op.trial)
    xhs, λᵀs_∂Rs∂φ = (), ()
    for k in 1:NB
      xh_k = get_solution(op,uhd,k)
      res(uk,vk,φh) = op.residuals[k](xhs,uk,vk,φh)
      λᵀk_∂Rk∂φ = ∇(res,[xh_k,xh_k,φh],3)
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
    new{NB,SB,A,B,C,D,E,F}(op.residuals,op.jacobians,spaces,assems,_solvers,plb_caches,fwd_caches,adj_caches)
  end
end

StaggeredFEStateMapTypes{NB} = Union{StaggeredNonlinearFEStateMap{NB},StaggeredAffineFEStateMap{NB}}

get_state(m::StaggeredNonlinearFEStateMap) = FEFunction(get_trial_space(m),m.fwd_caches[1])
get_spaces(m::StaggeredNonlinearFEStateMap) = m.spaces
get_assemblers(m::StaggeredNonlinearFEStateMap) = m.assems

function forward_solve!(φ_to_u::StaggeredNonlinearFEStateMap,φ::AbstractVector)
  φh = FEFunction(GridapTopOpt.get_aux_space(φ_to_u),φ)
  return forward_solve!(φ_to_u,φh)
end

function forward_solve!(φ_to_u::StaggeredNonlinearFEStateMap,φh)
  solvers = φ_to_u.solvers
  x, X, cache, init_op = φ_to_u.fwd_caches

  op = get_staggered_operator_at_φ(φ_to_u,φh)
  _, new_cache = solve!(FEFunction(X,x),solvers.solver,op,cache);
  φ_to_u.fwd_caches = (x,X,new_cache,init_op)

  return x
end

function dRdφ(φ_to_u::StaggeredNonlinearFEStateMap{NB},uh,λh,φh) where NB
  _,_,_,init_op = φ_to_u.fwd_caches
  _,_,_,init_adjoint_op = φ_to_u.adj_caches
  xhs, ∂Rs∂φ = (), ()
  for k in 1:NB
    xh_k = get_solution(init_op,uh,k)
    λh_k = get_solution(init_adjoint_op,λh,NB-k+1)
    _res_k(uk,vk,φh) = φ_to_u.residuals[k](xhs,uk,vk,φh)
    ∂Rk∂φ = ∇(_res_k,[xh_k,λh_k,φh],3)
    xhs, ∂Rs∂φ = (xhs...,xh_k), (∂Rs∂φ...,∂Rk∂φ)
  end
  return ∂Rs∂φ
end

function adjoint_solve!(φ_to_u::StaggeredFEStateMapTypes,xh,dFdxj_at_φ::Function)
  solvers = φ_to_u.solvers
  x_adjoint,X_adjoint,cache,_ = φ_to_u.adj_caches
  op_at_φ = get_staggered_operator_at_φ(φ_to_u,φh)
  xh_comb = _get_solutions(op_at_φ,xh)
  op_adjoint = generate_adjoint_operator(op_at_φ,xh_comb,j->dFdxj_at_φ(j,xh_comb))

  solve!(FEFunction(X_adjoint,x_adjoint),solvers.adjoint_solver,op_adjoint,cache);
  return x_adjoint
end

function pullback(φ_to_u::StaggeredFEStateMapTypes{NB},xh,φh,dFdxj::Function;kwargs...) where NB
  Σ_λᵀs_∂Rs∂φ, assem_deriv = φ_to_u.plb_caches
  U_reg = GridapTopOpt.get_deriv_space(φ_to_u)

  ## Adjoint Solve
  λ  = adjoint_solve!(φ_to_u,xh,(j,xh_comb)->dFdxj(j,φh,xh_comb))
  λh = FEFunction(get_test_space(φ_to_u),λ)

  ## Compute grad
  λᵀ∂Rs∂φ = dRdφ(φ_to_u,xh,λh,φh)
  fill!(Σ_λᵀs_∂Rs∂φ,zero(eltype(Σ_λᵀs_∂Rs∂φ)))
  for k in 1:NB
    vecdata = collect_cell_vector(U_reg,λᵀ∂Rs∂φ[k])
    assemble_vector_add!(Σ_λᵀs_∂Rs∂φ,assem_deriv,vecdata)
  end
  rmul!(Σ_λᵀs_∂Rs∂φ, -1)

  return (NoTangent(),Σ_λᵀs_∂Rs∂φ)
end

function ChainRulesCore.rrule(φ_to_u::StaggeredFEStateMapTypes,φh)
  u  = forward_solve!(φ_to_u,φh)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  return u, du -> pullback(φ_to_u,uh,φh,du)
end

### Helpers
# Staggered Ops
function _get_staggered_nonlinear_operator_at_φ(residuals,jacobians,trials,tests,assems,φh)
  residuals_at_φ = map(r->((xhs,uk,vk) -> r(xhs,uk,vk,φh)),residuals)
  jacs_at_φ = map(j->((xhs,uk,duk,dvk) -> j(xhs,uk,duk,dvk,φh)),jacobians)
  return StaggeredNonlinearFEOperator(residuals_at_φ,jacs_at_φ,trials,tests,assems)
end

function get_staggered_operator_at_φ(φ_to_u::StaggeredNonlinearFEStateMap,φh)
  residuals, jacobians, trials, tests, assems = φ_to_u.residuals,φ_to_u.jacobians,
    φ_to_u.spaces.trials,φ_to_u.spaces.tests,φ_to_u.assems.assems
  _get_staggered_nonlinear_operator_at_φ(residuals, jacobians, trials, tests, assems, φh)
end

function get_staggered_operator_at_φ(op::StaggeredNonlinearFEOperator,φh)
  residuals, jacobians, trials, tests, assems = op.residuals,op.jacobians,
    op.trials,op.tests,op.assems
  _get_staggered_nonlinear_operator_at_φ(residuals, jacobians, trials, tests, assems, φh)
end

# Adjoint Ops
function dummy_generate_adjoint_operator(op_at_φ::StaggeredNonlinearFEOperator{NB},xh_comb) where NB
  xhs,cs = (),()
  for k = 1:NB
    xh_k = xh_comb[k]
    dxk = get_fe_basis(op_at_φ.trials[k])
    l_dummy(vk) = op_at_φ.residuals[k](xhs,xh_k,vk) # TODO: unify this function for both operators
    cs = (cs...,l_dummy(dxk))
    xhs = (xhs...,xh_k)
  end
  generate_adjoint_operator(op_at_φ,xh_comb,i->cs[i])
end

function _get_kth_jacobian(op::StaggeredNonlinearFEOperator{NB},xh_comb,k::Int) where NB
  jac(xhs,λk,Λk) = op.jacobians[k](xh_comb[1:end-NB+k-1],xh_comb[k],Λk,λk)
end

function _get_kth_jacobian(op::StaggeredAffineFEOperator{NB},xh_comb,k::Int) where NB
  jac(xhs,λk,Λk) = op.biforms[k](xh_comb[1:end-NB+k-1],Λk,λk)
end

function generate_adjoint_operator(op_at_φ::StaggeredFEOperator{NB},xh_comb,dFdxj::Function) where NB
  a_adj,l_adj=(),()
  for k = 1:NB-1
    a_adj_k(xhs,λk,Λk) = _get_kth_jacobian(op_at_φ,xh_comb,k)(xhs,λk,Λk)
    l_adj_k(xhs,Λk) = dFdxj(k) - sum(∂Rk∂xhi(op_at_φ,xh_comb,k,i)(xhs[NB-i+1]) for i = k+1:NB)
    a_adj = (a_adj...,a_adj_k)
    l_adj = (l_adj...,l_adj_k)
  end
  a_adj = (a_adj...,_get_kth_jacobian(op_at_φ,xh_comb,NB))
  l_adj = (l_adj...,(xhs,Λk) -> dFdxj(NB))
  StaggeredAffineFEOperator(collect(reverse(a_adj)),collect(reverse(l_adj)),
    reverse(op_at_φ.tests),reverse(op_at_φ.trials),reverse(op_at_φ.assems))
end

function ∂Rk∂xhi(op::StaggeredNonlinearFEOperator{NB}, xh_comb, i::Int, k::Int) where NB
  @assert NB >= k && 1 <= i < k
  res_k_at_xhi(xhi,vk) = op.residuals[k]((xh_comb[1:i-1]...,xhi,xh_comb[i+1:end-1]...),xh_comb[k],vk)
  ∂res_k_at_xhi(vk) = ∇(res_k_at_xhi,[xh_comb[i],vk],1)
end