using GridapTopOpt
using Gridap
using Gridap.Algebra, Gridap.Geometry, Gridap.MultiField, Gridap.Arrays,
  Gridap.FESpaces, Gridap.CellData
using GridapTopOpt: AbstractFEStateMap

using BlockArrays
using GridapSolvers
using GridapSolvers.BlockSolvers, GridapSolvers.LinearSolvers, GridapSolvers.NonlinearSolvers
using GridapSolvers.BlockSolvers: combine_fespaces, get_solution

import GridapTopOpt: forward_solve!,dRdφ,update_adjoint_caches!,adjoint_solve!,get_state,get_spaces,get_assemblers,
  get_trial_space,get_test_space

struct StaggeredAffineFEStateMap{NB,SB,A,B,C} <: AbstractFEStateMap
  biforms :: Vector{<:Function}
  liforms :: Vector{<:Function}
  spaces  :: A
  assems  :: B
  solvers :: C

  function StaggeredAffineFEStateMap(
      op              :: StaggeredAffineFEOperator{NB,SB},
      V_φ             :: FESpace,
      U_reg           :: FESpace;
      assems_adjoint  :: Vector{<:Assembler} = op.assems,
      assem_deriv     :: Assembler = SparseMatrixAssembler(U_reg,U_reg),
      solver         :: StaggeredFESolver{NB} = StaggeredFESolver(fill(LUSolver(),length(op.biforms))),
      adjoint_solver :: StaggeredFESolver{NB} = StaggeredFESolver(fill(LUSolver(),length(op.biforms)))
    ) where {NB,SB}

    spaces = (;trials=op.trials,tests=op.tests,aux_space=V_φ,deriv_space=U_reg,
      trial=op.trial, test=op.test)
    assems = (;assmes=op.assems,assem_deriv,assems_adjoint)
    _solvers = (;solver,adjoint_solver)
    A,B,C = typeof(spaces), typeof(assems), typeof(_solvers)
    new{NB,SB,A,B,C}(op.biforms,op.liforms,spaces,assems,_solvers)
  end
end

# get_state(m::StaggeredAffineFEStateMap) = isdefined(m,:fwd_caches) ? FEFunction(m.fwd_caches.X,m.fwd_caches.x) :
#   error("fwd_caches not defined, call `forward_solve!` to create the cache")
get_state(::StaggeredAffineFEStateMap) = error("This method has been deprecated")
get_spaces(m::StaggeredAffineFEStateMap) = m.spaces
get_assemblers(m::StaggeredAffineFEStateMap) = m.assems

function forward_solve!(φ_to_u::StaggeredAffineFEStateMap,φh,::Nothing)
  biforms, liforms, spaces, assmes, solvers = φ_to_u.biforms,
    φ_to_u.liforms, φ_to_u.spaces, φ_to_u.assems, φ_to_u.solvers
  a_at_φ = map(a->((xhs,uk,vk) -> a(xhs,uk,vk,φh)),biforms)
  l_at_φ = map(l->((xhs,vk) -> l(xhs,vk,φh)),liforms)
  op = StaggeredAffineFEOperator(a_at_φ,l_at_φ,spaces.trials,spaces.tests,assmes.assmes)
  X = combine_fespaces(spaces.trials)

  xh = zero(X)
  xh, cache = solve!(xh,solvers.solver,op);

  x = get_free_dof_values(xh)
  return xh, (x,X,op,cache)
end

function forward_solve!(φ_to_u::StaggeredAffineFEStateMap,φh,cache)
  biforms, liforms, spaces, assmes, solvers = φ_to_u.biforms,
    φ_to_u.liforms, φ_to_u.spaces, φ_to_u.assems, φ_to_u.solvers
  a_at_φ = map(a->((xhs,uk,vk) -> a(xhs,uk,vk,φh)),biforms)
  l_at_φ = map(l->((xhs,vk) -> l(xhs,vk,φh)),liforms)
  op = StaggeredAffineFEOperator(a_at_φ,l_at_φ,spaces.trials,spaces.tests,assmes.assmes)

  x, X, _, last_cache = cache
  xh, cache_updated = solve!(FEFunction(X,x),solvers.solver,op,last_cache);

  return xh, (x,X,op,cache_updated)
end

function dRdφ(φ_to_u::StaggeredAffineFEStateMap{NB},uhs,λhs,φh) where NB
  biforms, liforms, spaces, assmes = φ_to_u.biforms, φ_to_u.liforms, φ_to_u.spaces, φ_to_u.assems
  dummy_op = StaggeredAffineFEOperator(biforms,liforms,spaces.trials,spaces.tests,assmes.assmes)
  xhs, ∂Rs∂φ = (), ()
  for k in 1:NB
    xh_k = get_solution(op,uhs,k)
    λh_k = get_solution(op,λhs,k)
    _a(uk,vk,φh) = dummy_op.biforms[k](xhs,uk,vk,φh)
    _l(vk,φh) = dummy_op.liforms[k](xhs,vk,φh)
    ∂Rk∂φ = ∇((uk,vk,φh) -> _a(uk,vk,φh) - _l(vk,φh),[xh_k,λh_k,φh],3)
    xhs, ∂Rs∂φ = (xhs...,xh_k), (∂Rs∂φ...,∂Rk∂φ)
  end
  return ∂Rs∂φ
end

function adjoint_solve!(φ_to_u::StaggeredAffineFEStateMap,du::AbstractVector,::Nothing)

end

function adjoint_solve!(φ_to_u::StaggeredAffineFEStateMap,du::AbstractVector,cache)

end

function pullback(φ_to_u::StaggeredAffineFEStateMap,uh,φh,du,cache)
  dudφ_vec, old_adjoint_cache = cache

  ## Adjoint Solve
  λ, adjoint_cache = adjoint_solve!(φ_to_u, du, old_adjoint_cache)
  λh = FEFunction(get_test_space(φ_to_u),λ)

  ## Compute grad
  dudφ_vecdata = collect_cell_vector(get_deriv_space(φ_to_u),dRdφ(φ_to_u,uh,λh,φh))
  assemble_vector!(dudφ_vec,get_deriv_assembler(φ_to_u),dudφ_vecdata)
  rmul!(dudφ_vec, -1)

  return (NoTangent(),dudφ_vec)
end