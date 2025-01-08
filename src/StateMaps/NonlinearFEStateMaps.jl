"""
    struct NonlinearFEStateMap{A,B,C,D,E,F} <: AbstractFEStateMap

A structure to enable the forward problem and pullback for nonlinear finite
element operators.

# Parameters

- `res::A`: a `Function` defining the residual of the problem.
- `jac::B`: a `Function` defining Jacobian of the residual.
- `spaces::C`: `Tuple` of finite element spaces.
- `plb_caches::D`: A cache for the pullback operator.
- `fwd_caches::E`: A cache for the forward problem.
- `adj_caches::F`: A cache for the adjoint problem.
"""
struct NonlinearFEStateMap{A,B,C,D,E,F} <: AbstractFEStateMap
  res        :: A
  jac        :: B
  spaces     :: C
  plb_caches :: D
  fwd_caches :: E
  adj_caches :: F

  @doc """
      NonlinearFEStateMap(
        res::Function,jac::Function,U,V,V_φ,U_reg,φh;
        assem_U = SparseMatrixAssembler(U,V),
        assem_adjoint = SparseMatrixAssembler(V,U),
        assem_deriv = SparseMatrixAssembler(U_reg,U_reg),
        nls::NonlinearSolver = NewtonSolver(LUSolver();maxiter=50,rtol=1.e-8,verbose=true),
        adjoint_ls::LinearSolver = LUSolver()
      )

  Create an instance of `NonlinearFEStateMap` given the residual `res` as a `Function` type,
  trial and test spaces `U` and `V`, the FE space `V_φ` for `φh`, the FE space `U_reg`
  for derivatives, and the measures as additional arguments.

  Optional arguments enable specification of assemblers, nonlinear solver, and adjoint (linear) solver.
  """
  function NonlinearFEStateMap(
    res::Function,jac::Function,U,V,V_φ,U_reg,φh;
    assem_U = SparseMatrixAssembler(U,V),
    assem_adjoint = SparseMatrixAssembler(V,U),
    assem_deriv = SparseMatrixAssembler(U_reg,U_reg),
    nls::NonlinearSolver = NewtonSolver(LUSolver();maxiter=50,rtol=1.e-8,verbose=true),
    adjoint_ls::LinearSolver = LUSolver()
  )
    spaces = (U,V,V_φ,U_reg)

    ## Pullback cache
    uhd = zero(U)
    vecdata = collect_cell_vector(U_reg,∇(res,[uhd,uhd,φh],3))
    dudφ_vec = allocate_vector(assem_deriv,vecdata)
    plb_caches = (dudφ_vec,assem_deriv)

    ## Forward cache
    x = zero_free_values(U)
    _res(u,v) = res(u,v,φh)
    _jac(u,du,v) = jac(u,du,v,φh)
    op = get_algebraic_operator(FEOperator(_res,_jac,U,V,assem_U))
    nls_cache = instantiate_caches(x,nls,op)
    fwd_caches = (nls,nls_cache,x,assem_U)

    ## Adjoint cache
    _jac_adj(du,v) = jac(uhd,du,v,φh)
    adjoint_K  = assemble_adjoint_matrix(_jac_adj,assem_adjoint,U,V)
    adjoint_x  = allocate_in_domain(adjoint_K); fill!(adjoint_x,zero(eltype(adjoint_x)))
    adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
    adj_caches = (adjoint_ns,adjoint_K,adjoint_x,assem_adjoint,adjoint_ls,jac)

    A, B, C = typeof(res), typeof(jac), typeof(spaces)
    D, E, F = typeof(plb_caches), typeof(fwd_caches), typeof(adj_caches)
    return new{A,B,C,D,E,F}(res,jac,spaces,plb_caches,fwd_caches,adj_caches)
  end

  @doc """
      NonlinearFEStateMap(
        res::Function,jac::Function,adjoint_jac::Function,U,V,V_φ,U_reg,φh;
        assem_U = SparseMatrixAssembler(U,V),
        assem_adjoint = SparseMatrixAssembler(V,U),
        assem_deriv = SparseMatrixAssembler(U_reg,U_reg),
        nls::NonlinearSolver = NewtonSolver(LUSolver();maxiter=50,rtol=1.e-8,verbose=true),
        adjoint_ls::LinearSolver = LUSolver()
      )

  In addition to the above, pass the jacobian `adjoint_jac` for the purposes of
  solving the adjoint problem. This can be computed with AD or by hand, but and
  allows the user to specify a different jacobian for the forward problem (e.g.,
  for picard iterations).
  """
  function NonlinearFEStateMap(
    res::Function,jac::Function,adjoint_jac::Function,U,V,V_φ,U_reg,φh;
    assem_U = SparseMatrixAssembler(U,V),
    assem_adjoint = SparseMatrixAssembler(V,U),
    assem_deriv = SparseMatrixAssembler(U_reg,U_reg),
    nls::NonlinearSolver = NewtonSolver(LUSolver();maxiter=50,rtol=1.e-8,verbose=true),
    adjoint_ls::LinearSolver = LUSolver()
  )
    spaces = (U,V,V_φ,U_reg)

    ## Pullback cache
    uhd = zero(U)
    vecdata = collect_cell_vector(U_reg,∇(res,[uhd,uhd,φh],3))
    dudφ_vec = allocate_vector(assem_deriv,vecdata)
    plb_caches = (dudφ_vec,assem_deriv)

    ## Forward cache
    x = zero_free_values(U)
    _res(u,v) = res(u,v,φh)
    _jac(u,du,v) = jac(u,du,v,φh)
    op = get_algebraic_operator(FEOperator(_res,_jac,U,V,assem_U))
    nls_cache = instantiate_caches(x,nls,op)
    fwd_caches = (nls,nls_cache,x,assem_U)

    ## Adjoint cache
    _jac_adj(du,v) = adjoint_jac(uhd,du,v,φh)
    adjoint_K  = assemble_adjoint_matrix(_jac_adj,assem_adjoint,U,V)
    adjoint_x  = allocate_in_domain(adjoint_K); fill!(adjoint_x,zero(eltype(adjoint_x)))
    adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
    adj_caches = (adjoint_ns,adjoint_K,adjoint_x,assem_adjoint,adjoint_ls,adjoint_jac)

    A, B, C = typeof(res), typeof(jac), typeof(spaces)
    D, E, F = typeof(plb_caches), typeof(fwd_caches), typeof(adj_caches)
    return new{A,B,C,D,E,F}(res,jac,spaces,plb_caches,fwd_caches,adj_caches)
  end
end

function NonlinearFEStateMap(res::Function,U,V,V_φ,U_reg,φh;jac=nothing,kwargs...)
  if isnothing(jac)
    jac = (u,du,v,φh) -> jacobian(res,[u,v,φh],1)
  end
  NonlinearFEStateMap(res,jac,U,V,V_φ,U_reg,φh;kwargs...)
end

get_state(m::NonlinearFEStateMap) = FEFunction(get_trial_space(m),m.fwd_caches[3])
get_spaces(m::NonlinearFEStateMap) = m.spaces
get_assemblers(m::NonlinearFEStateMap) = (m.fwd_caches[4],m.plb_caches[2],m.adj_caches[4])

function forward_solve!(φ_to_u::NonlinearFEStateMap,φh)
  U, V, _, _ = φ_to_u.spaces
  nls, nls_cache, x, assem_U = φ_to_u.fwd_caches

  res(u,v) = φ_to_u.res(u,v,φh)
  jac(u,du,v) = φ_to_u.jac(u,du,v,φh)
  op = get_algebraic_operator(FEOperator(res,jac,U,V,assem_U))
  solve!(x,nls,op,nls_cache)
  return x
end

function forward_solve!(φ_to_u::NonlinearFEStateMap,φ::AbstractVector)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return forward_solve!(φ_to_u,φh)
end

function dRdφ(φ_to_u::NonlinearFEStateMap,uh,vh,φh)
  res = φ_to_u.res
  return ∇(res,[uh,vh,φh],3)
end

function update_adjoint_caches!(φ_to_u::NonlinearFEStateMap,uh,φh)
  adjoint_ns, adjoint_K, _, assem_adjoint, _, adjoint_jac = φ_to_u.adj_caches
  U, V, _, _ = φ_to_u.spaces
  jac(du,v) =  adjoint_jac(uh,du,v,φh)
  assemble_adjoint_matrix!(jac,adjoint_K,assem_adjoint,U,V)
  numerical_setup!(adjoint_ns,adjoint_K)
  return φ_to_u.adj_caches
end

function adjoint_solve!(φ_to_u::NonlinearFEStateMap,du::AbstractVector)
  adjoint_ns, _, adjoint_x, _, _, _ = φ_to_u.adj_caches
  solve!(adjoint_x,adjoint_ns,du)
  return adjoint_x
end