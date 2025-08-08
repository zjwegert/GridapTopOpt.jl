"""
    struct ReverseNonlinearFEStateMap{A,B,C,D,E} <: AbstractFEStateMap

A structure to enable the forward problem and pullback for nonlinear finite
element operators.

# Parameters

- `res`: a `Function` defining the residual of the problem.
- `jacs`: a tuple of `Function`s defining the Jacobian of the residual and
  the Jacobian for the adjoint. This allows the user to specify a different Jacobian
  for the forward problem/adjoint problem (e.g., Picard iterations).
- `spaces`: `Tuple` of finite element spaces.
- `assems`: `Tuple` of assemblers
- `cache`: An AffineFEStateMapCache
"""
struct ReverseNonlinearFEStateMap{A,B,C,D,E} <: AbstractFEStateMap
  res         :: A
  jacs        :: B
  spaces      :: C
  assems      :: D
  cache       :: E

  @doc """
      ReverseNonlinearFEStateMap(
        res::Function,jac::Function,U,V,V_φ;
        assem_U = SparseMatrixAssembler(U,V),
        assem_adjoint = SparseMatrixAssembler(V,U),
        assem_deriv = SparseMatrixAssembler(V_φ,V_φ),
        nls::NonlinearSolver = NewtonSolver(LUSolver();maxiter=50,rtol=1.e-8,verbose=true),
        adjoint_ls::LinearSolver = LUSolver(),
        adjoint_jac::Function = jac
      )

  Create an instance of `ReverseNonlinearFEStateMap` given the residual `res` as a `Function` type,
  trial and test spaces `U` and `V`, the FE space `V_φ` for `φh` and derivatives,
  and the measures as additional arguments.

  Optional arguments enable specification of assemblers, nonlinear solver, and
  adjoint (linear) solver. In addition, the jacobian `adjoint_jac` can be optionally
  specified for the purposes of solving the adjoint problem. This can be computed
  with AD or by hand, but and allows the user to specify a different jacobian
  for the forward problem (e.g., for picard iterations).
  """
  function ReverseNonlinearFEStateMap(
    res::Function,jac::Function,U,V,V_φ,V_diff;
    assem_U = SparseMatrixAssembler(U,V),
    assem_adjoint = SparseMatrixAssembler(V,U),
    assem_deriv = SparseMatrixAssembler(V_φ,V_φ),
    nls::NonlinearSolver = NewtonSolver(LUSolver();maxiter=50,rtol=1.e-8,verbose=true),
    adjoint_ls::LinearSolver = LUSolver(),
    adjoint_jac::Function = jac
  )
    jacs = (jac,adjoint_jac)
    spaces = (U,V,V_φ,V_diff)
    assems = (;assem_U,assem_deriv,assem_adjoint)
    cache = FEStateMapCache(nls,adjoint_ls)

    A, B, C = typeof(res), typeof(jacs), typeof(spaces)
    D, E = typeof(assems), typeof(cache)
    return new{A,B,C,D,E}(res,jacs,spaces,assems,cache)
  end
end

function ReverseNonlinearFEStateMap(res::Function,U,V,V_φ,V_diff;kwargs...)
  println("nw")
  jac = (u,du,v,φh) -> Gridap.jacobian(res,[u,v,φh],1)
  ReverseNonlinearFEStateMap(res,jac,U,V,V_φ,V_diff;kwargs...)
end

# Caching
function build_cache!(state_map::ReverseNonlinearFEStateMap,φh)
  assem_U, assem_deriv, assem_adjoint = state_map.assems
  U,V,V_φ = state_map.spaces
  res = state_map.res
  jac, adjoint_jac = state_map.jacs
  cache = state_map.cache
  nls, adjoint_ls = cache.solvers[1], cache.solvers[2]

  ## Pullback cache
  dudφ_vec = get_free_dof_values(zero(V_φ))
  cache.plb_cache = (dudφ_vec,assem_deriv)

  ## Forward cache
  x = zero_free_values(U)
  _res(u,v) = res(u,v,φh)
  _jac(u,du,v) = jac(u,du,v,φh)
  op = get_algebraic_operator(FEOperator(_res,_jac,U,V,assem_U))
  nls_cache = instantiate_caches(x,nls,op)
  cache.fwd_cache = (nls,nls_cache,x)

  ## Adjoint cache
  uhd = zero(U)
  _jac_adj(du,v) = adjoint_jac(uhd,du,v,φh)
  adjoint_K  = assemble_adjoint_matrix(_jac_adj,assem_adjoint,U,V)
  adjoint_x  = allocate_in_domain(adjoint_K); fill!(adjoint_x,zero(eltype(adjoint_x)))
  adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
  cache.adj_cache = (adjoint_ns,adjoint_K,adjoint_x)

  ## Update cache status
  cache.cache_built = true

  return cache
end

# Getters
function get_state(m::ReverseNonlinearFEStateMap)
  @assert is_cache_built(m.cache) """
    You must build the cache before using get_state. This can be achieved by either
    solving your problem with my_state_map(φh) or by running build_cache!(my_state_map,φh)
  """
  FEFunction(get_trial_space(m),m.cache.fwd_cache[3])
end
get_plb_cache(m::ReverseNonlinearFEStateMap) = m.cache.plb_cache
get_spaces(m::ReverseNonlinearFEStateMap) = m.spaces
get_assemblers(m::ReverseNonlinearFEStateMap) = m.assems

function forward_solve!(φ_to_u::ReverseNonlinearFEStateMap,φh)
  U, V, _, _ = φ_to_u.spaces
  assem_U = φ_to_u.assems.assem_U
  res=φ_to_u.res
  jac,_=φ_to_u.jacs
  if !is_cache_built(φ_to_u.cache)
    build_cache!(φ_to_u,φh)
  end
  nls, nls_cache, x = φ_to_u.cache.fwd_cache

  _res(u,v) = res(u,v,φh)
  _jac(u,du,v) = jac(u,du,v,φh)
  op = get_algebraic_operator(FEOperator(_res,_jac,U,V,assem_U))
  solve!(x,nls,op,nls_cache)
  return x
end

function forward_solve!(φ_to_u::ReverseNonlinearFEStateMap,φ::AbstractVector)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return forward_solve!(φ_to_u,φh)
end

function dRdφ(φ_to_u::ReverseNonlinearFEStateMap,uh,vh,φh)
  res = φ_to_u.res
  V_diff = φ_to_u.spaces[4]
  function _res(φ)
    φh = FEFunction(V_diff, φ)
    sum(res(uh,vh,φh))
  end                                                                    
  return ReverseDiff.gradient(_res,φh.free_values)
end

function update_adjoint_caches!(φ_to_u::ReverseNonlinearFEStateMap,uh,φh)
  if !is_cache_built(φ_to_u.cache)
    build_cache!(φ_to_u,φh)
  end
  assem_adjoint = φ_to_u.assems.assem_adjoint
  adjoint_ns, adjoint_K, _ = φ_to_u.cache.adj_cache
  _,adjoint_jac=φ_to_u.jacs
  U, V, _ = φ_to_u.spaces
  jac(du,v) =  adjoint_jac(uh,du,v,φh)
  assemble_adjoint_matrix!(jac,adjoint_K,assem_adjoint,U,V)
  numerical_setup!(adjoint_ns,adjoint_K)
  return φ_to_u.cache.adj_cache
end

function adjoint_solve!(φ_to_u::ReverseNonlinearFEStateMap,du::AbstractVector)
  adjoint_ns, _, adjoint_x = φ_to_u.cache.adj_cache
  solve!(adjoint_x,adjoint_ns,du)
  return adjoint_x
end

function pullback(φ_to_u::ReverseNonlinearFEStateMap,uh,φh,du;updated=false)
  dudφ_vec, assem_deriv = get_plb_cache(φ_to_u)
  V_φ = get_deriv_space(φ_to_u)

  ## Adjoint Solve
  if !updated
    update_adjoint_caches!(φ_to_u,uh,φh)
  end
  λ  = adjoint_solve!(φ_to_u,du)
  λh = FEFunction(get_test_space(φ_to_u),λ)

  ## Compute grad
  dudφ_vec = dRdφ(φ_to_u,uh,λh,φh)
  rmul!(dudφ_vec, -1)

  return (NoTangent(),dudφ_vec)
end

# ## Backwards compat
# function ReverseNonlinearFEStateMap(res::Function,jac::Function,U,V,V_φ,U_reg,φh; kwargs...)
#   error(_msg_v0_3_0(ReverseNonlinearFEStateMap))
# end

# function ReverseNonlinearFEStateMap(res::Function,U,V,V_φ,U_reg,φh;kwargs...)
#   error(_msg_v0_3_0(ReverseNonlinearFEStateMap))
# end

# function ReverseNonlinearFEStateMap(res::Function,jac::Function,U,V,V_φ,φh; kwargs...)
#   error(_msg_v0_4_0(ReverseNonlinearFEStateMap))
# end

# function ReverseNonlinearFEStateMap(res::Function,U,V,V_φ,φh;kwargs...)
#   error(_msg_v0_4_0(ReverseNonlinearFEStateMap))
# end