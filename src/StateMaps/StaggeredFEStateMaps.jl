"""
    struct StaggeredAffineFEStateMap{NB,SB} <: AbstractFEStateMap{NB,SB}
      biforms    :: Vector{<:Function}
      liforms    :: Vector{<:Function}
      ∂Rk∂xhi    :: Tuple{Vararg{Tuple{Vararg{Function}}}}
      spaces     :: A
      assems     :: B
      solvers    :: C
      plb_caches :: D
      fwd_caches :: E
      adj_caches :: F
    end

Affine staggered state map for the equivalent StaggeredAffineFEOperator,
used to solve staggered problems where the k-th equation is linear in `u_k`.

Similar to the StaggeredAffineFEOperator counterpart, we expect a set of
bilinear/linear form pairs that also depend on φ:

    a_k((u_1,...,u_{k-1}),u_k,v_k,φ) = ∫(...)
    l_k((u_1,...,u_{k-1}),v_k,φ) = ∫(...)

These can be assembled into a set of linear systems:

    A_k u_k = b_k

where `A_k` and `b_k` only depend on the previous variables `u_1,...,u_{k-1}`.

!!! warning
    The current implementation of the rrules is not compatible with Zygote.
    This will be fixed in a future release.
"""
struct StaggeredAffineFEStateMap{NB,SB,A,B,C,D,E,F} <: AbstractFEStateMap
  biforms    :: Vector{<:Function}
  liforms    :: Vector{<:Function}
  ∂Rk∂xhi    :: Tuple{Vararg{Tuple{Vararg{Function}}}}
  spaces     :: A
  assems     :: B
  solvers    :: C
  plb_caches :: D
  fwd_caches :: E
  adj_caches :: F

  function StaggeredAffineFEStateMap(
      op              :: StaggeredAffineFEOperator{NB,SB},
      ∂Rk∂xhi         :: Tuple{Vararg{Tuple{Vararg{Function}}}},
      V_φ,
      φh;
      assem_deriv     :: Assembler = SparseMatrixAssembler(V_φ,V_φ),
      assems_adjoint  :: Vector{<:Assembler} = map(SparseMatrixAssembler,op.tests,op.trials),
      solver          :: StaggeredFESolver{NB} = StaggeredFESolver(fill(LUSolver(),length(op.biforms))),
      adjoint_solver  :: StaggeredFESolver{NB} = StaggeredFESolver(fill(LUSolver(),length(op.biforms)))
    ) where {NB,SB}

    @assert length(∂Rk∂xhi) == NB-1 && all(map(length,∂Rk∂xhi) .== 1:NB-1) """\n
    We expect k ∈ 2:NB and i ∈ 1:k-1.

    E.g.,
      ∂Rk∂xhi[1] = ∂R2∂xhi = (∂R2∂xh1,)
      ∂Rk∂xhi[2] = ∂R3∂xhi = (∂R3∂xh1,∂R3∂xh2,)
      ...
      ∂Rk∂xhi[k] = ∂R{k}∂xhi = (∂R{k}∂xh1,∂R{k}∂xh2,...,∂R{k}∂xh{k-1},)
    """

    Σ_λᵀs_∂Rs∂φ = get_free_dof_values(zero(V_φ))
    plb_caches = (Σ_λᵀs_∂Rs∂φ,assem_deriv)

    ## Forward cache
    op_at_φ = get_staggered_operator_at_φ(op,φh)
    xh = one(op.trial)
    op_cache = _instantiate_caches(xh,solver,op_at_φ)
    fwd_caches = (zero_free_values(op.trial),op.trial,op_cache,op_at_φ)

    ## Adjoint cache
    xh_adj = one(op.trial)
    op_adjoint = dummy_generate_adjoint_operator(op_at_φ,assems_adjoint,φh,xh_adj,∂Rk∂xhi)
    op_cache = _instantiate_caches(xh_adj,adjoint_solver,op_adjoint)
    adj_caches = (zero_free_values(op_adjoint.trial),op_adjoint.trial,op_cache,op_adjoint)

    spaces = (;trial=op_at_φ.trial,test=op_at_φ.test,aux_space=V_φ,trials=op_at_φ.trials,tests=op_at_φ.tests)
    assems = (;assems=op_at_φ.assems,assem_deriv,adjoint_assems=assems_adjoint)
    _solvers = (;solver,adjoint_solver)
    A,B,C,D,E,F = typeof(spaces), typeof(assems), typeof(_solvers),
      typeof(plb_caches), typeof(fwd_caches), typeof(adj_caches)
    new{NB,SB,A,B,C,D,E,F}(op.biforms,op.liforms,∂Rk∂xhi,spaces,assems,_solvers,plb_caches,fwd_caches,adj_caches)
  end
end

"""
    StaggeredAffineFEStateMap(
        op              :: StaggeredAffineFEOperator{NB,SB},
        V_φ,
        φh;
        assem_deriv     :: Assembler = SparseMatrixAssembler(V_φ,V_φ),
        assems_adjoint  :: Vector{<:Assembler} = map(SparseMatrixAssembler,op.tests,op.trials),
        solver          :: StaggeredFESolver{NB} = StaggeredFESolver(fill(LUSolver(),length(op.biforms))),
        adjoint_solver  :: StaggeredFESolver{NB} = StaggeredFESolver(fill(LUSolver(),length(op.biforms)))
    ) where {NB,SB}

Create an instance of `StaggeredAffineFEStateMap` given a
StaggeredAffineFEOperator `op`, the auxiliary space `V_φ` for `φh` and
derivatives, and the parameter `φh`.

Otional arguemnts:
- `assem_deriv` is the assembler for the derivative space.
- `assems_adjoint` is a vector of assemblers for the adjoint space.
- `solver` is a `StaggeredFESolver` for the forward problem.
- `adjoint_solver` is a `StaggeredFESolver` for the adjoint problem.
"""
function StaggeredAffineFEStateMap(
  op              :: StaggeredAffineFEOperator{NB,SB},
  V_φ,
  φh;
  assem_deriv     :: Assembler = SparseMatrixAssembler(V_φ,V_φ),
  assems_adjoint  :: Vector{<:Assembler} = map(SparseMatrixAssembler,op.tests,op.trials),
  solver          :: StaggeredFESolver{NB} = StaggeredFESolver(fill(LUSolver(),length(op.biforms))),
  adjoint_solver  :: StaggeredFESolver{NB} = StaggeredFESolver(fill(LUSolver(),length(op.biforms)))
) where {NB,SB}

  ∂Rk∂xhi = ()
  for k = 2:NB
    _∂Rk∂xhi = ()
    for i = 1:k-1
      __∂Rk∂xhi(dxj,xhs,xhk,vhk,φ) = ∇(
        xi->op.biforms[k]((xhs[1:i-1]...,xi,xhs[i+1:end]...),xhk,vhk,φ) -
            op.liforms[k]((xhs[1:i-1]...,xi,xhs[i+1:end]...),vhk,φ)
      )(xhs[i])
      _∂Rk∂xhi = (_∂Rk∂xhi...,__∂Rk∂xhi)
    end
    ∂Rk∂xhi = (∂Rk∂xhi...,_∂Rk∂xhi)
  end

  return StaggeredAffineFEStateMap(op,∂Rk∂xhi,V_φ,φh;assem_deriv,assems_adjoint,solver,adjoint_solver)
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

# Fixed staggered operators at φ
function _get_staggered_affine_operator_at_φ(biforms,liforms,trials,tests,assems,φh)
  a_at_φ = map(a->((xhs,uk,vk) -> a(xhs,uk,vk,φh)),biforms)
  l_at_φ = map(l->((xhs,vk) -> l(xhs,vk,φh)),liforms)
  return StaggeredAffineFEOperator(a_at_φ,l_at_φ,trials,tests,assems)
end

function get_staggered_operator_at_φ(φ_to_u::StaggeredAffineFEStateMap,φh)
  biforms, liforms, trials, tests, assems = φ_to_u.biforms,φ_to_u.liforms,
    φ_to_u.spaces.trials,φ_to_u.spaces.tests,φ_to_u.assems.assems
  _get_staggered_affine_operator_at_φ(biforms, liforms, trials, tests, assems, φh)
end

function get_staggered_operator_at_φ(op::StaggeredAffineFEOperator,φh)
  biforms, liforms, trials, tests, assems = op.biforms,op.liforms,
    op.trials,op.tests,op.assems
  _get_staggered_affine_operator_at_φ(biforms, liforms, trials, tests, assems, φh)
end

"""
    mutable struct StaggeredNonlinearFEStateMap{NB,SB} <: AbstractFEStateMap{NB,SB}
      const residuals         :: Vector{<:Function}
      const jacobians         :: Vector{<:Function}
      const adjoint_jacobians :: Vector{<:Function}
      const ∂Rk∂xhi           :: Tuple{Vararg{Tuple{Vararg{Function}}}}
      const spaces            :: A
      const assems            :: B
      const solvers           :: C
      const plb_caches        :: D
      fwd_caches              :: E
      const adj_caches        :: F
    end

Staggered nonlinear state map for the equivalent StaggeredNonlinearFEOperator,
used to solve staggered problems where the k-th equation is nonlinear in `u_k`.

Similar to the previous structure and the StaggeredNonlinearFEOperator counterpart,
we expect a set of residual/jacobian pairs that also depend on φ:

  jac_k((u_1,...,u_{k-1},φ),u_k,du_k,dv_k) = ∫(...)
  res_k((u_1,...,u_{k-1},φ),u_k,v_k) = ∫(...)

!!! info
    This is mutable for now, in future we will refactor ChainRules to remove storage of caches

!!! warning
    The current implementation of the rrules is not compatible with Zygote.
    This will be fixed in a future release.
"""
mutable struct StaggeredNonlinearFEStateMap{NB,SB,A,B,C,D,E,F} <: AbstractFEStateMap
  const residuals         :: Vector{<:Function}
  const jacobians         :: Vector{<:Function}
  const adjoint_jacobians :: Vector{<:Function}
  const ∂Rk∂xhi           :: Tuple{Vararg{Tuple{Vararg{Function}}}}
  const spaces            :: A
  const assems            :: B
  const solvers           :: C
  const plb_caches        :: D
  fwd_caches              :: E
  const adj_caches        :: F

  function StaggeredNonlinearFEStateMap(
      op                :: StaggeredNonlinearFEOperator{NB,SB},
      ∂Rk∂xhi           :: Tuple{Vararg{Tuple{Vararg{Function}}}},
      V_φ,
      φh;
      assem_deriv       :: Assembler = SparseMatrixAssembler(V_φ,V_φ),
      assems_adjoint    :: Vector{<:Assembler} = map(SparseMatrixAssembler,op.tests,op.trials),
      solver            :: StaggeredFESolver{NB} = StaggeredFESolver(
        fill(NewtonSolver(LUSolver();maxiter=50,rtol=1.e-8,verbose=true),length(op.residuals))),
      adjoint_solver    :: StaggeredFESolver{NB} = StaggeredFESolver(fill(LUSolver(),length(op.residuals))),
      adjoint_jacobians :: Vector{<:Function} = op.jacobians
    ) where {NB,SB}

    @assert length(∂Rk∂xhi) == NB-1 && all(map(length,∂Rk∂xhi) .== 1:NB-1) """\n
    We expect k ∈ 2:NB and i ∈ 1:k-1.

    E.g.,
      ∂Rk∂xhi[1] = ∂R2∂xhi = (∂R2∂xh1,)
      ∂Rk∂xhi[2] = ∂R3∂xhi = (∂R3∂xh1,∂R3∂xh2,)
      ...
      ∂Rk∂xhi[k] = ∂R{k}∂xhi = (∂R{k}∂xh1,∂R{k}∂xh2,...,∂R{k}∂xh{k-1},)
    """

    Σ_λᵀs_∂Rs∂φ = get_free_dof_values(zero(V_φ))
    plb_caches = (Σ_λᵀs_∂Rs∂φ,assem_deriv)

    ## Forward cache
    op_at_φ = get_staggered_operator_at_φ(op,φh)
    xh = one(op.trial)
    op_cache = _instantiate_caches(xh,solver,op_at_φ)
    fwd_caches = (zero_free_values(op.trial),op.trial,op_cache,op_at_φ)

    ## Adjoint cache
    xh_adj = one(op.trial)
    op_adjoint = dummy_generate_adjoint_operator(op_at_φ,assems_adjoint,φh,xh_adj,∂Rk∂xhi)
    op_cache = _instantiate_caches(xh_adj,adjoint_solver,op_adjoint)
    adj_caches = (zero_free_values(op_adjoint.trial),op_adjoint.trial,op_cache,op_adjoint)

    spaces = (;trial=op_at_φ.trial,test=op_at_φ.test,aux_space=V_φ,trials=op_at_φ.trials,tests=op_at_φ.tests)
    assems = (;assems=op_at_φ.assems,assem_deriv,adjoint_assems=assems_adjoint)
    _solvers = (;solver,adjoint_solver)
    A,B,C,D,E,F = typeof(spaces), typeof(assems), typeof(_solvers),
      typeof(plb_caches), typeof(fwd_caches), typeof(adj_caches)
    new{NB,SB,A,B,C,D,E,F}(op.residuals,op.jacobians,adjoint_jacobians,∂Rk∂xhi,spaces,assems,_solvers,plb_caches,fwd_caches,adj_caches)
  end
end

"""
    function StaggeredNonlinearFEStateMap(
      op                :: StaggeredNonlinearFEOperator{NB,SB},
      V_φ,
      φh;
      assem_deriv       :: Assembler = SparseMatrixAssembler(V_φ,V_φ),
      assems_adjoint    :: Vector{<:Assembler} = map(SparseMatrixAssembler,op.tests,op.trials),
      solver            :: StaggeredFESolver{NB} = StaggeredFESolver(
        fill(NewtonSolver(LUSolver();maxiter=50,rtol=1.e-8,verbose=true),length(op.residuals))),
      adjoint_solver    :: StaggeredFESolver{NB} = StaggeredFESolver(fill(LUSolver(),length(op.residuals))),
      adjoint_jacobians :: Vector{<:Function} = op.jacobians
    ) where {NB,SB}

Create an instance of `StaggeredNonlinearFEStateMap` given a
`StaggeredNonlinearFEOperator` `op`, the auxiliary space `V_φ` for `φh`
and derivatives, and the parameter `φh`.

Otional arguemnts:
- `assem_deriv` is the assembler for the derivative space.
- `assems_adjoint` is a vector of assemblers for the adjoint space.
- `solver` is a `StaggeredFESolver` for the forward problem.
- `adjoint_solver` is a `StaggeredFESolver` for the adjoint problem.
- `adjoint_jacobians` is a vector of jacobians for the adjoint problem.
"""
function StaggeredNonlinearFEStateMap(
  op                :: StaggeredNonlinearFEOperator{NB,SB},
  V_φ,
  φh;
  assem_deriv       :: Assembler = SparseMatrixAssembler(V_φ,V_φ),
  assems_adjoint    :: Vector{<:Assembler} = map(SparseMatrixAssembler,op.tests,op.trials),
  solver            :: StaggeredFESolver{NB} = StaggeredFESolver(
    fill(NewtonSolver(LUSolver();maxiter=50,rtol=1.e-8,verbose=true),length(op.residuals))),
  adjoint_solver    :: StaggeredFESolver{NB} = StaggeredFESolver(fill(LUSolver(),length(op.residuals))),
  adjoint_jacobians :: Vector{<:Function} = op.jacobians
) where {NB,SB}

  ∂Rk∂xhi = ()
  for k = 2:NB
    _∂Rk∂xhi = ()
    for i = 1:k-1
      __∂Rk∂xhi(dxj,xhs,xhk,vhk,φ) = ∇(
        xi->op.residuals[k]((xhs[1:i-1]...,xi,xhs[i+1:end]...),xhk,vhk,φ)
      )(xhs[i])
      _∂Rk∂xhi = (_∂Rk∂xhi...,__∂Rk∂xhi)
    end
    ∂Rk∂xhi = (∂Rk∂xhi...,_∂Rk∂xhi)
  end

  return StaggeredNonlinearFEStateMap(op,∂Rk∂xhi,V_φ,φh;assem_deriv,assems_adjoint,solver,adjoint_solver,adjoint_jacobians)
end

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

# Fixed staggered operators at φ
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

function get_staggered_operator_at_φ_with_adjoint_jacs(φ_to_u::StaggeredNonlinearFEStateMap,φh)
  residuals, adjoint_jacobians, trials, tests, assems = φ_to_u.residuals,φ_to_u.adjoint_jacobians,
    φ_to_u.spaces.trials,φ_to_u.spaces.tests,φ_to_u.assems.assems
  _get_staggered_nonlinear_operator_at_φ(residuals, adjoint_jacobians, trials, tests, assems, φh)
end

function get_staggered_operator_at_φ_with_adjoint_jacs(φ_to_u::StaggeredAffineFEStateMap,φh)
  get_staggered_operator_at_φ(φ_to_u,φh)
end

## Generic methods on both types
StaggeredFEStateMapTypes{NB} = Union{StaggeredNonlinearFEStateMap{NB},StaggeredAffineFEStateMap{NB}}

# Adjoint solve and pullback
function adjoint_solve!(φ_to_u::StaggeredFEStateMapTypes,xh,φh,dFdxj)
  solvers = φ_to_u.solvers
  ∂Rk∂xhi = φ_to_u.∂Rk∂xhi
  x_adjoint,X_adjoint,cache,_ = φ_to_u.adj_caches
  adjoint_assems = φ_to_u.assems.adjoint_assems
  op_at_φ = get_staggered_operator_at_φ_with_adjoint_jacs(φ_to_u,φh)
  op_adjoint = generate_adjoint_operator(op_at_φ,adjoint_assems,φh,xh,dFdxj,∂Rk∂xhi)

  solve!(FEFunction(X_adjoint,x_adjoint),solvers.adjoint_solver,op_adjoint,cache);
  return x_adjoint
end

# TODO: Caching the adjoint is disabled in MPI mode as the ghost information is incorrect if cached with
#       a fake adjoint as we do in serial. This is a temporary solution and needs to be fixed
function adjoint_solve!(φ_to_u::StaggeredFEStateMapTypes,xh,φh::DistributedCellField,dFdxj)
  solvers = φ_to_u.solvers
  ∂Rk∂xhi = φ_to_u.∂Rk∂xhi
  x_adjoint,X_adjoint,cache,_ = φ_to_u.adj_caches
  adjoint_assems = φ_to_u.assems.adjoint_assems
  op_at_φ = get_staggered_operator_at_φ_with_adjoint_jacs(φ_to_u,φh)
  op_adjoint = generate_adjoint_operator(op_at_φ,adjoint_assems,φh,xh,dFdxj,∂Rk∂xhi)

  solve!(FEFunction(X_adjoint,x_adjoint),solvers.adjoint_solver,op_adjoint);
  return x_adjoint
end

function pullback(φ_to_u::StaggeredFEStateMapTypes{NB},xh,φh,dFdxj;kwargs...) where NB
  Σ_λᵀs_∂Rs∂φ, assem_deriv = φ_to_u.plb_caches
  Λ = last(φ_to_u.adj_caches).test
  V_φ = GridapTopOpt.get_deriv_space(φ_to_u)

  # Adjoint Solve
  λ  = adjoint_solve!(φ_to_u,xh,φh,dFdxj)
  λh = FEFunction(Λ,λ)

  # Compute Σ_λᵀs_∂Rs∂φ
  λᵀ∂Rs∂φ = dRdφ(φ_to_u,xh,λh,φh)
  fill!(Σ_λᵀs_∂Rs∂φ,zero(eltype(Σ_λᵀs_∂Rs∂φ)))
  for k in 1:NB
    vecdata = collect_cell_vector(V_φ,λᵀ∂Rs∂φ[k])
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

# Building adjoint operators
function generate_adjoint_operator(op_at_φ::StaggeredFEOperator{NB},adjoint_assems,φh,xh,dFdxj,∂Rk∂xhi) where NB
  xh_comb = _get_solutions(op_at_φ,xh)
  a_adj,l_adj=(),()
  for k = 1:NB-1
    dFdxk(Λk) = dFdxj[k](Λk,xh_comb,φh)
    ∑ᵢ∂Ri∂xhk(xhs,Λk) = sum(∂Rk∂xhi[i-1][k](Λk,xh_comb[1:i-1],xh_comb[i],xhs[NB-i+1],φh) for i = k+1:NB)

    a_adj_k(xhs,λk,Λk) = _get_kth_jacobian(op_at_φ,xh_comb,k)(xhs,λk,Λk)
    l_adj_k(xhs,Λk) = dFdxk(Λk) - ∑ᵢ∂Ri∂xhk(xhs,Λk)

    a_adj = (a_adj...,a_adj_k)
    l_adj = (l_adj...,l_adj_k)
  end
  a_adj = (a_adj...,_get_kth_jacobian(op_at_φ,xh_comb,NB))
  l_adj = (l_adj...,(xhs,Λk) -> dFdxj[NB](Λk,xh_comb,φh))
  StaggeredAdjointAffineFEOperator(collect(reverse(a_adj)),collect(reverse(l_adj)),
    reverse(op_at_φ.trials),reverse(op_at_φ.tests),reverse(adjoint_assems))
end

# Jacobian of kth residual
function _get_kth_jacobian(op::StaggeredNonlinearFEOperator{NB},xh_comb,k::Int) where NB
  jac(xhs,λk,Λk) = op.jacobians[k](xh_comb[1:end-NB+k-1],xh_comb[k],λk,Λk)
end

function _get_kth_jacobian(op::StaggeredAffineFEOperator{NB},xh_comb,k::Int) where NB
  jac(xhs,λk,Λk) = op.biforms[k](xh_comb[1:end-NB+k-1],λk,Λk)
end

# Dummy adjoint operator for setting up the cache
function dummy_generate_adjoint_operator(op_at_φ::StaggeredFEOperator{NB},adjoint_assems,φh,xh,∂Rk∂xhi) where NB
  xhs,cs = (),()
  for k = 1:NB
    xh_k = get_solution(op_at_φ,xh,k)
    dxk = get_fe_basis(op_at_φ.trials[k])
    l(Λk,xh_comb,φh) = dummy_linear_form(op_at_φ,xhs,xh_k,dxk,k)
    cs = (cs...,l)
    xhs = (xhs...,xh_k)

  end
  generate_adjoint_operator(op_at_φ,adjoint_assems,φh,xh,cs,∂Rk∂xhi)
end

function dummy_linear_form(op_at_φ::StaggeredAffineFEOperator,xhs,xh_k,dxk,k)
  op_at_φ.liforms[k](xhs,dxk)
end

function dummy_linear_form(op_at_φ::StaggeredNonlinearFEOperator,xhs,xh_k,dxk,k)
  op_at_φ.residuals[k](xhs,xh_k,dxk)
end

## Helpers
# Get all solutions
function _get_solutions(op::StaggeredFEOperator{NB},xh) where NB
  map(i->get_solution(op,xh,i),Tuple(1:NB))
end

function _get_solutions(spaces::Vector{<:FESpace},xh)
  map(i->get_solution(spaces,xh,i),Tuple(1:length(spaces)))
end

# Instantiate caches for staggered operator
function _instantiate_caches(xh,solver::StaggeredFESolver{NB},op::StaggeredFEOperator{NB}) where NB
  solvers = solver.solvers
  xhs, caches, operators = (), (), ()
  for k in 1:NB
    xh_k = GridapSolvers.BlockSolvers.get_solution(op,xh,k)
    x_k = get_free_dof_values(xh_k)
    op_k = GridapSolvers.BlockSolvers.get_operator(op,xhs,k)
    algebraic_op_k = get_algebraic_operator(op_k)
    cache_k = GridapTopOpt.instantiate_caches(x_k,solvers[k],algebraic_op_k)
    xhs, caches, operators = (xhs...,xh_k), (caches...,cache_k), (operators...,op_k)
  end
  return (caches,operators)
end

## StaggeredStateParamMap
# This structure is currently required for compatibility with the current ChainRules API.
#  In particular, the `rrule` method is a hack to get this working for the PDEConstrainedFunctionals
#  type.
#
# This will be refactored/removed in the future.
struct StaggeredStateParamMap{NB,A,B,C,D} <: GridapTopOpt.AbstractStateParamMap
  F       :: A
  spaces  :: B
  assems  :: C
  caches  :: D
  ∂F∂xhi  :: Tuple{Vararg{Function}}
end

function StaggeredStateParamMap(F::Function,∂F∂xhi::Tuple{Vararg{Function}},φ_to_u::StaggeredFEStateMapTypes)
  Us = φ_to_u.spaces.trials
  V_φ = GridapTopOpt.get_aux_space(φ_to_u)
  assem_deriv = GridapTopOpt.get_deriv_assembler(φ_to_u)
  assem_U = GridapTopOpt.get_pde_assembler(φ_to_u)
  StaggeredStateParamMap(F,∂F∂xhi,Us,V_φ,assem_U,assem_deriv)
end

function StaggeredStateParamMap(F::Function,φ_to_u::StaggeredFEStateMapTypes)
  Us = φ_to_u.spaces.trials
  V_φ = GridapTopOpt.get_aux_space(φ_to_u)
  assem_deriv = GridapTopOpt.get_deriv_assembler(φ_to_u)
  assem_U = GridapTopOpt.get_pde_assembler(φ_to_u)

  @assert length(Us) == length(assem_U)
  NB = length(Us)

  ∂F∂xhi = ()
  for j = 1:NB
    _∂F∂xhj(dxj,xhs,φ) = ∇(xj->F((xhs[1:j-1]...,xj,xhs[j+1:end]...),φ))(xhs[j])
    ∂F∂xhi = (∂F∂xhi...,_∂F∂xhj)
  end

  StaggeredStateParamMap(F,∂F∂xhi,Us,V_φ,assem_U,assem_deriv)
end

function StaggeredStateParamMap(
  F,∂F∂xhi::Tuple{Vararg{Function}},trials::Vector{<:FESpace},V_φ::FESpace,
  assem_U::Vector{<:Assembler},assem_deriv::Assembler
)
  @assert length(trials) == length(assem_U)
  ∂F∂φ_vec = get_free_dof_values(zero(V_φ))
  assems = (assem_U,assem_deriv)
  spaces = (trials,combine_fespaces(trials),V_φ)
  caches = ∂F∂φ_vec
  A,B,C,D = typeof(F),typeof(spaces),typeof(assems),typeof(caches)
  return StaggeredStateParamMap{length(trials),A,B,C,D}(F,spaces,assems,caches,∂F∂xhi)
end

function get_∂F∂φ_vec(u_to_j::StaggeredStateParamMap)
  u_to_j.caches
end

function (u_to_j::StaggeredStateParamMap)(u::AbstractVector,φ::AbstractVector)
  _,trial,V_φ = u_to_j.spaces
  uh = FEFunction(trial,u)
  φh = FEFunction(V_φ,φ)
  return u_to_j(uh,φh)
end

function (u_to_j::StaggeredStateParamMap)(uh,φh)
  trials,_,_ = u_to_j.spaces
  uh_comb = _get_solutions(trials,uh)
  sum(u_to_j.F(uh_comb,φh))
end

# The following is a hack to get this working in the current GridapTopOpt ChainRules API.
#   This will be refactored in the future
#
# TODO: This should be refactored when we refactor StateParamMap
function ChainRulesCore.rrule(u_to_j::StaggeredStateParamMap,uh,φh)
  F = u_to_j.F
  trials,_,V_φ = u_to_j.spaces
  _,assem_deriv = u_to_j.assems
  ∂F∂φ_vec = u_to_j.caches
  ∂F∂xhi = u_to_j.∂F∂xhi

  uh_comb = _get_solutions(trials,uh)

  function u_to_j_pullback(dj)
    ## Compute ∂F/∂uh(uh,φh) and ∂F/∂φh(uh,φh)
    ∂F∂φ = ∇((φ->F((uh_comb...,),φ)))(φh)
    ∂F∂φ_vecdata = collect_cell_vector(V_φ,∂F∂φ)
    assemble_vector!(∂F∂φ_vec,assem_deriv,∂F∂φ_vecdata)
    dj_∂F∂xhi = map(∂F∂xh->(x...)->dj*∂F∂xh(x...),∂F∂xhi)
    ∂F∂φ_vec .*= dj

    (  NoTangent(), dj_∂F∂xhi, ∂F∂φ_vec)
    # As above, this is really bad as dFdxj is a tuple of functions and ∂F∂φ_vec is a vector. This is temporary
  end
  return u_to_j(uh,φh), u_to_j_pullback
end

function ChainRulesCore.rrule(u_to_j::StaggeredStateParamMap,u::AbstractVector,φ::AbstractVector)
  _,trial,V_φ = u_to_j.spaces
  uh = FEFunction(trial,u)
  φh = FEFunction(V_φ,φ)
  return ChainRulesCore.rrule(u_to_j,uh,φh)
end

## Backwards compat
function StaggeredAffineFEStateMap(
    op::StaggeredAffineFEOperator,∂Rk∂xhi::Tuple{Vararg{Tuple{Vararg{Function}}}},V_φ,U_reg,φh; kwargs...)
  @warn _msg_v0_3_0 maxlog=1
  return StaggeredAffineFEStateMap(op,∂Rk∂xhi,V_φ,φh; kwargs...)
end

function StaggeredAffineFEStateMap(op::StaggeredAffineFEOperator,V_φ,U_reg,φh; kwargs...)
  @warn _msg_v0_3_0 maxlog=1
  return StaggeredAffineFEStateMap(op,V_φ,φh; kwargs...)
end

function StaggeredNonlinearFEStateMap(
    op::StaggeredNonlinearFEOperator,∂Rk∂xhi::Tuple{Vararg{Tuple{Vararg{Function}}}},V_φ,U_reg,φh; kwargs...)
  @warn _msg_v0_3_0 maxlog=1
  return StaggeredNonlinearFEStateMap(op,∂Rk∂xhi,V_φ,φh; kwargs...)
end

function StaggeredNonlinearFEStateMap(op::StaggeredNonlinearFEOperator,V_φ,U_reg,φh; kwargs...)
  @warn _msg_v0_3_0 maxlog=1
  return StaggeredNonlinearFEStateMap(op,V_φ,φh; kwargs...)
end

function StaggeredStateParamMap(F,∂F∂xhi::Tuple{Vararg{Function}},trials::Vector{<:FESpace},V_φ::FESpace,
    U_reg::FESpace,assem_U::Vector{<:Assembler},assem_deriv::Assembler)
  @warn _msg_v0_3_0 maxlog=1
  return StaggeredStateParamMap(F,∂F∂xhi,trials,V_φ,assem_U,assem_deriv)
end