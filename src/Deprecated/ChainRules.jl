"""
    Gridap.gradient(F,uh::Vector,K::Int)

Given a function `F` that returns a DomainContribution when called, and a vector of
`FEFunctions` `uh`, evaluate the partial derivative of `F` with respect to `uh[K]`.

# Example

Suppose `uh` and `φh` are FEFunctions with measures `dΩ` and `dΓ_N`.
Then the partial derivative of a function `J` wrt to `φh` is computed via
````
J(u,φ) = ∫(f(u,φ))dΩ + ∫(g(u,φ))dΓ_N
∂J∂φh = ∇(J,[uh,φh],2)
````
where `f` and `g` are user defined.
"""
function Gridap.gradient(F,uh::Vector{<:CellField},K::Int)
  @check 0 < K <= length(uh)
  _f(uk) = F(uh[1:K-1]...,uk,uh[K+1:end]...)
  return Gridap.gradient(_f,uh[K])
end

"""
    Gridap.jacobian(F,uh::Vector,K::Int)

Given a function `F` that returns a DomainContribution when called, and a
vector of `FEFunctions` or `CellField` `uh`, evaluate the Jacobian
`F` with respect to `uh[K]`.
"""
function Gridap.jacobian(F,uh::Vector{<:CellField},K::Int)
  @check 0 < K <= length(uh)
  _f(uk) = F(uh[1:K-1]...,uk,uh[K+1:end]...)
  return Gridap.jacobian(_f,uh[K])
end

function GridapDistributed.to_parray_of_arrays(a::NTuple{N,T}) where {N,T<:DebugArray}
  indices = linear_indices(first(a))
  map(indices) do i
    map(a) do aj
      aj.items[i]
    end
  end
end

function GridapDistributed.to_parray_of_arrays(a::NTuple{N,T}) where {N,T<:MPIArray}
  indices = linear_indices(first(a))
  map(indices) do i
    map(a) do aj
      PartitionedArrays.getany(aj)
    end
  end
end

abstract type AbstractStateParamMap end # <- this is only needed for compat with staggered state maps

"""
    struct StateParamMap{A,B,C,D} <: AbstractStateParamMap

A wrapper to handle partial differentation of a function F
of a specific form (see below) in a `ChainRules.jl` compatible way with caching.

# Assumptions

We assume that we have a function F of the following form:

`F(u,φ) = ∫(f(u,φ))dΩ₁ + ∫(g(u,φ))dΩ₂ + ...,`.

where `u` and `φ` are each expected to inherit from `Union{FEFunction,MultiFieldFEFunction}`
or the GridapDistributed equivalent.
"""
struct StateParamMap{A,B,C,D} <: AbstractStateParamMap
  F       :: A
  spaces  :: B
  assems  :: C
  caches  :: D
end

"""
    StateParamMap(F,U::FESpace,V_φ::FESpace,
    U_reg::FESpace,assem_U::Assembler,assem_deriv::Assembler)

Create an instance of `StateParamMap`.
"""
function StateParamMap(
  F,U::FESpace,V_φ::FESpace,U_reg::FESpace,
  assem_U::Assembler,assem_deriv::Assembler
)
  φ₀, u₀ = interpolate(x->-sqrt((x[1]-1/2)^2+(x[2]-1/2)^2)+0.2,V_φ), zero(U)
  # TODO: Can we make F a dummy functional?
  ∂j∂u_vecdata = collect_cell_vector(U,∇(F,[u₀,φ₀],1))
  ∂j∂φ_vecdata = collect_cell_vector(U_reg,∇(F,[u₀,φ₀],2))
  ∂j∂u_vec = allocate_vector(assem_U,∂j∂u_vecdata)
  ∂j∂φ_vec = allocate_vector(assem_deriv,∂j∂φ_vecdata)
  assems = (assem_U,assem_deriv)
  spaces = (U,V_φ,U_reg)
  caches = (∂j∂u_vec,∂j∂φ_vec)
  return StateParamMap(F,spaces,assems,caches)
end

"""
    (u_to_j::StateParamMap)(uh,φh)

Evaluate the `StateParamMap` at parameters `uh` and `φh`.
"""
(u_to_j::AbstractStateParamMap)(uh,φh) = sum(u_to_j.F(uh,φh))

function (u_to_j::StateParamMap)(u::AbstractVector,φ::AbstractVector)
  U,V_φ,_ = u_to_j.spaces
  uh = FEFunction(U,u)
  φh = FEFunction(V_φ,φ)
  return u_to_j(uh,φh)
end

"""
    ChainRulesCore.rrule(u_to_j::StateParamMap,uh,φh)

Return the evaluation of a `StateParamMap` and a
a function for evaluating the pullback of `u_to_j`. This enables
compatiblity with `ChainRules.jl`
"""
function ChainRulesCore.rrule(u_to_j::StateParamMap,uh,φh)
  F = u_to_j.F
  U,V_φ,U_reg = u_to_j.spaces
  assem_U,assem_deriv = u_to_j.assems
  ∂j∂u_vec,∂j∂φ_vec = u_to_j.caches

  function u_to_j_pullback(dj)
    ## Compute ∂F/∂uh(uh,φh) and ∂F/∂φh(uh,φh)
    ∂j∂u = ∇(F,[uh,φh],1)
    ∂j∂u_vecdata = collect_cell_vector(U,∂j∂u)
    assemble_vector!(∂j∂u_vec,assem_U,∂j∂u_vecdata)
    ∂j∂φ = ∇(F,[uh,φh],2)
    ∂j∂φ_vecdata = collect_cell_vector(U_reg,∂j∂φ)
    assemble_vector!(∂j∂φ_vec,assem_deriv,∂j∂φ_vecdata)
    ∂j∂u_vec .*= dj
    ∂j∂φ_vec .*= dj
    (  NoTangent(), ∂j∂u_vec, ∂j∂φ_vec )
  end
  return u_to_j(uh,φh), u_to_j_pullback
end

function ChainRulesCore.rrule(u_to_j::StateParamMap,u::AbstractVector,φ::AbstractVector)
  U,V_φ,U_reg = u_to_j.spaces
  uh = FEFunction(U,u)
  φh = FEFunction(V_φ,φ)
  return ChainRulesCore.rrule(u_to_j,uh,φh)
end

# Backwards compat
const StateParamIntegrandWithMeasure = StateParamMap

"""
    abstract type AbstractFEStateMap

Types inheriting from this abstract type should enable the evaluation and differentiation of
the solution to an FE problem `u` that implicitly depends on an auxiliary parameter `φ`.
"""
abstract type AbstractFEStateMap end

"""
    get_state(m::AbstractFEStateMap)

Return the solution/state `u` to the FE problem.
"""
get_state(::AbstractFEStateMap) = @abstractmethod

"""
    get_spaces(m::AbstractFEStateMap)

Return a collection of FE spaces. The first four entires should correspond to
[`get_trial_space`](@ref), [`get_test_space`](@ref), [`get_aux_space`](@ref), and
[`get_deriv_space`](@ref) unless these are overloaded for a particular implementation.
"""
get_spaces(::AbstractFEStateMap) = @abstractmethod

"""
    get_assemblers(m::AbstractFEStateMap)

Return a collection of assemblers. The first two entires should correspond to
[`get_pde_assembler`](@ref) and [`get_deriv_assembler`](@ref) unless these are
overloaded for a particular implementation.
"""
get_assemblers(::AbstractFEStateMap) = @abstractmethod

"""
    get_trial_space(m::AbstractFEStateMap)

Return trial space for FE problem.
"""
get_trial_space(m::AbstractFEStateMap) = get_spaces(m)[1]

"""
    get_test_space(m::AbstractFEStateMap)

Return test space for FE problem.
"""
get_test_space(m::AbstractFEStateMap) = get_spaces(m)[2]

"""
    get_aux_space(m::AbstractFEStateMap)

Return space for auxillary parameter.
"""
get_aux_space(m::AbstractFEStateMap) = get_spaces(m)[3]

"""
    get_deriv_space(m::AbstractFEStateMap)

Return space for derivatives.
"""
get_deriv_space(m::AbstractFEStateMap) = get_spaces(m)[4]

"""
    get_pde_assembler(m::AbstractFEStateMap)

Return assembler for FE problem.
"""
get_pde_assembler(m::AbstractFEStateMap) = get_assemblers(m)[1]

"""
    get_deriv_assembler(m::AbstractFEStateMap)

Return assembler for derivatives.
"""
get_deriv_assembler(m::AbstractFEStateMap) = get_assemblers(m)[2]

"""
    (φ_to_u::AbstractFEStateMap)(φh)

Evaluate the forward problem `u` given `φ`. This should compute the
FE problem.
"""
@inline (φ_to_u::AbstractFEStateMap)(φh) = forward_solve!(φ_to_u,φh)

"""
    forward_solve!(φ_to_u::AbstractFEStateMap,φh)

Evaluate the forward problem `u` given `φ`. This should compute the
FE problem.
"""
function forward_solve!(φ_to_u::AbstractFEStateMap,φh)
  @abstractmethod
end

"""
    update_adjoint_caches!(φ_to_u::AbstractFEStateMap,uh,φh)

Update the cache for the adjoint problem. This is usually a tuple
of objects.
"""
function update_adjoint_caches!(φ_to_u::AbstractFEStateMap,uh,φh)
  @abstractmethod
end

function update_adjoint_caches!(φ_to_u::AbstractFEStateMap,u::AbstractVector,φ::AbstractVector)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return update_adjoint_caches!(φ_to_u,uh,φh)
end

"""
    adjoint_solve!(φ_to_u::AbstractFEStateMap,du::AbstractVector)

Evaluate the solution to the adjoint problem given a RHS vector `∂F∂u` denoted `du`.
This should solve the linear problem `dRduᵀ*λ = ∂F∂uᵀ`.
"""
function adjoint_solve!(φ_to_u::AbstractFEStateMap,du::AbstractVector)
  @abstractmethod
end

"""
    dRdφ(φ_to_u::AbstractFEStateMap,uh,vh,φh)

Compute the derivative with respect to `φh` of the residual R.
"""
function dRdφ(φ_to_u::AbstractFEStateMap,uh,vh,φh)
  @abstractmethod
end

function dRdφ(φ_to_u::AbstractFEStateMap,u::AbstractVector,v::AbstractVector,φ::AbstractVector)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  vh = FEFunction(get_test_space(φ_to_u),v)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return dRdφ(φ_to_u,uh,vh,φh)
end

"""
    pullback(φ_to_u::AbstractFEStateMap,uh,φh,du;updated)

Compute `∂F∂u*dudφ` at `φh` and `uh` using the adjoint method. I.e., let

`∂F∂u*dudφ = -λᵀ*dRdφ`

and solve the adjoint problem `dRduᵀ*λ = ∂F∂uᵀ` using [`adjoint_solve!`](@ref).
"""
function pullback(φ_to_u::AbstractFEStateMap,uh,φh,du;updated=false)
  dudφ_vec, assem_deriv = φ_to_u.plb_caches
  U_reg = get_deriv_space(φ_to_u)

  ## Adjoint Solve
  if !updated
    update_adjoint_caches!(φ_to_u,uh,φh)
  end
  λ  = adjoint_solve!(φ_to_u,du)
  λh = FEFunction(get_test_space(φ_to_u),λ)

  ## Compute grad
  dudφ_vecdata = collect_cell_vector(U_reg,dRdφ(φ_to_u,uh,λh,φh))
  assemble_vector!(dudφ_vec,assem_deriv,dudφ_vecdata)
  rmul!(dudφ_vec, -1)

  return (NoTangent(),dudφ_vec)
end

function pullback(φ_to_u::AbstractFEStateMap,u::AbstractVector,φ::AbstractVector,du::AbstractVector;updated=false)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return pullback(φ_to_u,uh,φh,du;updated=updated)
end

"""
    rrule(φ_to_u::AbstractFEStateMap,φh)

Return the evaluation of a `AbstractFEStateMap` and a
a function for evaluating the pullback of `φ_to_u`. This enables
compatiblity with `ChainRules.jl`
"""
function ChainRulesCore.rrule(φ_to_u::AbstractFEStateMap,φh)
  u  = forward_solve!(φ_to_u,φh)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  update_adjoint_caches!(φ_to_u,uh,φh)
  return u, du -> pullback(φ_to_u,uh,φh,du;updated=true)
end

function ChainRulesCore.rrule(φ_to_u::AbstractFEStateMap,φ::AbstractVector)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return ChainRulesCore.rrule(φ_to_u,φh)
end

function StateParamMap(F::Function,φ_to_u::AbstractFEStateMap)
  U = get_trial_space(φ_to_u)
  V = get_test_space(φ_to_u)
  V_φ = get_aux_space(φ_to_u)
  U_reg = get_deriv_space(φ_to_u)
  assem_deriv = get_deriv_assembler(φ_to_u)
  assem_U = get_pde_assembler(φ_to_u)
  StateParamMap(F,U,V_φ,U_reg,assem_U,assem_deriv)
end

"""
    struct AffineFEStateMap{A,B,C,D,E,F} <: AbstractFEStateMap

A structure to enable the forward problem and pullback for affine finite
element operators `AffineFEOperator`.

# Parameters

- `biform::A`: `Function` defining the bilinear form.
- `liform::B`: `Function` defining the linear form.
- `spaces::C`: `Tuple` of finite element spaces.
- `plb_caches::D`: A cache for the pullback operator.
- `fwd_caches::E`: A cache for the forward problem.
- `adj_caches::F`: A cache for the adjoint problem.
"""
struct AffineFEStateMap{A,B,C,D,E,F} <: AbstractFEStateMap
  biform     :: A
  liform     :: B
  spaces     :: C
  plb_caches :: D
  fwd_caches :: E
  adj_caches :: F

  @doc """
      AffineFEStateMap(
        a::Function,l::Function,
        U,V,V_φ,U_reg,φh;
        assem_U = SparseMatrixAssembler(U,V),
        assem_adjoint = SparseMatrixAssembler(V,U),
        assem_deriv = SparseMatrixAssembler(U_reg,U_reg),
        ls::LinearSolver = LUSolver(),
        adjoint_ls::LinearSolver = LUSolver()
      )

  Create an instance of `AffineFEStateMap` given the bilinear form `a` and linear
  form `l` as `Function` types, trial and test spaces `U` and `V`, the FE space `V_φ`
  for `φh`, the FE space `U_reg` for derivatives, and the measures as additional arguments.

  Optional arguments enable specification of assemblers and linear solvers.
  """
  function AffineFEStateMap(
      biform::Function,liform::Function,
      U,V,V_φ,U_reg,φh;
      assem_U = SparseMatrixAssembler(U,V),
      assem_adjoint = SparseMatrixAssembler(V,U),
      assem_deriv = SparseMatrixAssembler(U_reg,U_reg),
      ls::LinearSolver = LUSolver(),
      adjoint_ls::LinearSolver = LUSolver()
    )
    # TODO: I really want to get rid of the φh argument...

    spaces = (U,V,V_φ,U_reg)

    ## Pullback cache
    uhd = zero(U)
    vecdata = collect_cell_vector(U_reg,∇(biform,[uhd,uhd,φh],3) - ∇(liform,[uhd,φh],2))
    dudφ_vec = allocate_vector(assem_deriv,vecdata)
    plb_caches = (dudφ_vec,assem_deriv)

    ## Forward cache
    op = AffineFEOperator((u,v)->biform(u,v,φh),v->liform(v,φh),U,V,assem_U)
    K, b = get_matrix(op), get_vector(op)
    x  = allocate_in_domain(K); fill!(x,zero(eltype(x)))
    ns = numerical_setup(symbolic_setup(ls,K),K)
    fwd_caches = (ns,K,b,x,uhd,assem_U,ls)

    ## Adjoint cache
    adjoint_K  = assemble_matrix((u,v)->biform(v,u,φh),assem_adjoint,V,U)
    adjoint_x  = allocate_in_domain(adjoint_K); fill!(adjoint_x,zero(eltype(adjoint_x)))
    adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
    adj_caches = (adjoint_ns,adjoint_K,adjoint_x,assem_adjoint,adjoint_ls)

    A,B,C = typeof(biform), typeof(liform), typeof(spaces)
    D,E,F = typeof(plb_caches),typeof(fwd_caches), typeof(adj_caches)
    return new{A,B,C,D,E,F}(biform,liform,spaces,plb_caches,fwd_caches,adj_caches)
  end
end

# Getters
get_state(m::AffineFEStateMap) = FEFunction(get_trial_space(m),m.fwd_caches[4])
get_spaces(m::AffineFEStateMap) = m.spaces
get_assemblers(m::AffineFEStateMap) = (m.fwd_caches[6],m.plb_caches[2],m.adj_caches[4])

function forward_solve!(φ_to_u::AffineFEStateMap,φh::FEFunction)
  biform, liform = φ_to_u.biform, φ_to_u.liform
  U, V, _, _ = φ_to_u.spaces
  ns, K, b, x, uhd, assem_U, _ = φ_to_u.fwd_caches

  a_fwd(u,v) = biform(u,v,φh)
  l_fwd(v)   = liform(v,φh)
  assemble_matrix_and_vector!(a_fwd,l_fwd,K,b,assem_U,U,V,uhd)
  numerical_setup!(ns,K)
  solve!(x,ns,b)
  return x
end

function forward_solve!(φ_to_u::AffineFEStateMap,φ::AbstractVector)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return forward_solve!(φ_to_u,φh)
end

function dRdφ(φ_to_u::AffineFEStateMap,uh,vh,φh)
  biform, liform = φ_to_u.biform, φ_to_u.liform
  return ∇(biform,[uh,vh,φh],3) - ∇(liform,[vh,φh],2)
end

function update_adjoint_caches!(φ_to_u::AffineFEStateMap,uh,φh)
  adjoint_ns, adjoint_K, _, assem_adjoint, _ = φ_to_u.adj_caches
  U, V, _, _ = φ_to_u.spaces
  assemble_matrix!((u,v) -> φ_to_u.biform(v,u,φh),adjoint_K,assem_adjoint,V,U)
  numerical_setup!(adjoint_ns,adjoint_K)
  return φ_to_u.adj_caches
end

function adjoint_solve!(φ_to_u::AffineFEStateMap,du::AbstractVector)
  adjoint_ns, _, adjoint_x, _, _ = φ_to_u.adj_caches
  solve!(adjoint_x,adjoint_ns,du)
  return adjoint_x
end

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
        res::Function,U,V,V_φ,U_reg,φh;
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
    adj_caches = (adjoint_ns,adjoint_K,adjoint_x,assem_adjoint,adjoint_ls)

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

function forward_solve!(φ_to_u::NonlinearFEStateMap,φh::FEFunction)
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
  adjoint_ns, adjoint_K, _, assem_adjoint, _ = φ_to_u.adj_caches
  U, V, _, _ = φ_to_u.spaces
  jac(du,v) =  φ_to_u.jac(uh,du,v,φh)
  assemble_adjoint_matrix!(jac,adjoint_K,assem_adjoint,U,V)
  numerical_setup!(adjoint_ns,adjoint_K)
  return φ_to_u.adj_caches
end

function adjoint_solve!(φ_to_u::NonlinearFEStateMap,du::AbstractVector)
  adjoint_ns, _, adjoint_x, _, _ = φ_to_u.adj_caches
  solve!(adjoint_x,adjoint_ns,du)
  return adjoint_x
end

"""
    struct RepeatingAffineFEStateMap <: AbstractFEStateMap

A structure to enable the forward problem and pullback for affine finite
element operators `AffineFEOperator` with multiple linear forms but only
a single bilinear form.

# Parameters

- `biform`: A `Function` defining the bilinear form.
- `liform`: A vector of `Function` defining the linear forms.
- `spaces`: Repeated finite element spaces.
- `spaces_0`: Original finite element spaces that are being repeated.
- `plb_caches`: A cache for the pullback operator.
- `fwd_caches`: A cache for the forward problem.
- `adj_caches`: A cache for the adjoint problem.
"""
struct RepeatingAffineFEStateMap{A,B,C,D,E,F,G} <: AbstractFEStateMap
  biform     :: A
  liform     :: B
  spaces     :: C
  spaces_0   :: D
  plb_caches :: E
  fwd_caches :: F
  adj_caches :: G

  @doc """
      RepeatingAffineFEStateMap(
        nblocks::Int,a::Function,l::Vector{<:Function},
        U0,V0,V_φ,U_reg,φh;
        assem_U = SparseMatrixAssembler(U0,V0),
        assem_adjoint = SparseMatrixAssembler(V0,U0),
        assem_deriv = SparseMatrixAssembler(U_reg,U_reg),
        ls::LinearSolver = LUSolver(),
        adjoint_ls::LinearSolver = LUSolver()
      )

  Create an instance of `RepeatingAffineFEStateMap` given the number of blocks `nblocks`,
  a bilinear form `a`, a vector of linear form `l` as `Function` types, the trial and test
  spaces `U` and `V`, the FE space `V_φ` for `φh`, the FE space `U_reg` for derivatives,
  and the measures as additional arguments.

  Optional arguments enable specification of assemblers and linear solvers.

  # Note

  - The resulting `FEFunction` will be a `MultiFieldFEFunction` (or GridapDistributed equivalent)
    where each field corresponds to an entry in the vector of linear forms
  """
  function RepeatingAffineFEStateMap(
    nblocks::Int,biform::Function,liforms::Vector{<:Function},
    U0,V0,V_φ,U_reg,φh;
    assem_U = SparseMatrixAssembler(U0,V0),
    assem_adjoint = SparseMatrixAssembler(V0,U0),
    assem_deriv = SparseMatrixAssembler(U_reg,U_reg),
    ls::LinearSolver = LUSolver(),
    adjoint_ls::LinearSolver = LUSolver()
  )
    @check nblocks == length(liforms)

    spaces_0 = (U0,V0)
    assem_U0 = assem_U

    U, V = repeat_spaces(nblocks,U0,V0)
    spaces = (U,V,V_φ,U_reg)
    assem_U = SparseMatrixAssembler(
      get_local_matrix_type(assem_U0), get_local_vector_type(assem_U0),
      U, V, get_local_assembly_strategy(assem_U0)
    )

    ## Pullback cache
    uhd = zero(U0)
    contr = nblocks * ∇(biform,[uhd,uhd,φh],3)
    for liform in liforms
      contr = contr - ∇(liform,[uhd,φh],2)
    end
    dudφ_vec = allocate_vector(assem_deriv,collect_cell_vector(U_reg,contr))
    plb_caches = (dudφ_vec,assem_deriv)

    ## Forward cache
    K  = assemble_matrix((u,v) -> biform(u,v,φh),assem_U0,U0,V0)
    b  = allocate_in_range(K); fill!(b,zero(eltype(b)))
    b0 = allocate_in_range(K); fill!(b0,zero(eltype(b0)))
    x  = repeated_allocate_in_domain(nblocks,K); fill!(x,zero(eltype(x)))
    ns = numerical_setup(symbolic_setup(ls,K),K)
    fwd_caches = (ns,K,b,x,uhd,assem_U0,b0,assem_U,ls)

    ## Adjoint cache
    adjoint_K  = assemble_matrix((u,v)->biform(v,u,φh),assem_adjoint,V0,U0)
    adjoint_x  = repeated_allocate_in_domain(nblocks,adjoint_K); fill!(adjoint_x,zero(eltype(adjoint_x)))
    adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
    adj_caches = (adjoint_ns,adjoint_K,adjoint_x,assem_adjoint,adjoint_ls)

    A,B,C,D = typeof(biform), typeof(liforms), typeof(spaces), typeof(spaces_0)
    E,F,G = typeof(plb_caches), typeof(fwd_caches), typeof(adj_caches)
    return new{A,B,C,D,E,F,G}(biform,liforms,spaces,spaces_0,plb_caches,fwd_caches,adj_caches)
  end
end

const MultiFieldSpaceTypes = Union{<:MultiField.MultiFieldFESpace,<:GridapDistributed.DistributedMultiFieldFESpace}

function repeat_spaces(nblocks::Integer,U0::FESpace,V0::FESpace)
  U = MultiFieldFESpace([U0 for i in 1:nblocks];style=BlockMultiFieldStyle())
  V = MultiFieldFESpace([V0 for i in 1:nblocks];style=BlockMultiFieldStyle())
  return U,V
end

function repeat_spaces(nblocks::Integer,U0::MultiFieldSpaceTypes,V0::MultiFieldSpaceTypes)
  nfields = num_fields(U0)
  @assert nfields == num_fields(V0)
  @assert MultiFieldStyle(U0) === MultiFieldStyle(V0)

  mfs = repeat_mfs(nblocks,nfields,MultiFieldStyle(U0))
  U = MultiFieldFESpace(repeat([U0...],nblocks);style=mfs)
  V = MultiFieldFESpace(repeat([V0...],nblocks);style=mfs)
  return U,V
end

function repeat_mfs(nblocks::Integer,nfields::Integer,::ConsecutiveMultiFieldStyle)
  return BlockMultiFieldStyle(nblocks,Tuple(fill(nfields,nblocks)))
end

function repeat_mfs(nblocks::Integer,nfields::Integer,::BlockMultiFieldStyle{NB,SB,P}) where {NB,SB,P}
  @assert length(P) == nfields
  P_new = [P...]
  for iB in 2:nblocks
    P_new = vcat(P_new,(P .+ (iB-1)*nfields)...)
  end
  return BlockMultiFieldStyle(NB*nblocks, Tuple(repeat([SB...],nblocks)), Tuple(P_new))
end

repeated_blocks(V0::FESpace,x::AbstractBlockVector) = blocks(x)
repeated_blocks(V0::FESpace,xh) = xh

repeated_blocks(V0::MultiFieldSpaceTypes,x::AbstractBlockVector) = repeated_blocks(MultiFieldStyle(V0),V0,x)
repeated_blocks(::ConsecutiveMultiFieldStyle,V0,x::AbstractBlockVector) = blocks(x)

function repeated_blocks(::BlockMultiFieldStyle{NB},V0::MultiFieldSpaceTypes,x::AbstractBlockVector) where NB
  xb = blocks(x)
  @assert length(xb) % NB == 0

  nblocks = length(xb) ÷ NB
  rep_blocks = map(1:nblocks) do iB
    mortar(xb[(iB-1)*NB+1:iB*NB])
  end
  return rep_blocks
end

function repeated_blocks(V0::MultiFieldSpaceTypes,xh)
  x_blocks = repeated_blocks(MultiFieldStyle(V0),V0,get_free_dof_values(xh))
  rep_blocks = map(x_blocks) do xb
    FEFunction(V0,xb)
  end
  return rep_blocks
end

function repeated_allocate_in_domain(nblocks::Integer,M::AbstractMatrix)
  mortar(map(i -> allocate_in_domain(M), 1:nblocks))
end

function repeated_allocate_in_domain(nblocks::Integer,M::AbstractBlockMatrix)
  mortar(vcat(map(i -> blocks(allocate_in_domain(M)), 1:nblocks)...))
end

get_state(m::RepeatingAffineFEStateMap) = FEFunction(get_trial_space(m),m.fwd_caches[4])
get_spaces(m::RepeatingAffineFEStateMap) = m.spaces
get_assemblers(m::RepeatingAffineFEStateMap) = (m.fwd_caches[8],m.plb_caches[2],m.adj_caches[4])

function forward_solve!(φ_to_u::RepeatingAffineFEStateMap,φh)
  biform, liforms = φ_to_u.biform, φ_to_u.liform
  U0, V0 = φ_to_u.spaces_0
  ns, K, b, x, uhd, assem_U0, b0, _, _ = φ_to_u.fwd_caches

  a_fwd(u,v) = biform(u,v,φh)
  assemble_matrix!(a_fwd,K,assem_U0,U0,V0)
  numerical_setup!(ns,K)

  l0_fwd(v) = a_fwd(uhd,v)
  assemble_vector!(l0_fwd,b0,assem_U0,V0)
  rmul!(b0,-1)

  v = get_fe_basis(V0)
  map(repeated_blocks(U0,x),liforms) do xi, li
    copy!(b,b0)
    vecdata = collect_cell_vector(V0,li(v,φh))
    assemble_vector_add!(b,assem_U0,vecdata)
    solve!(xi,ns,b)
  end
  return x
end

function forward_solve!(φ_to_u::RepeatingAffineFEStateMap,φ::AbstractVector)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return forward_solve!(φ_to_u,φh)
end

function dRdφ(φ_to_u::RepeatingAffineFEStateMap,uh,vh,φh)
  biform, liforms = φ_to_u.biform, φ_to_u.liform
  U0, V0 = φ_to_u.spaces_0

  uh_blocks = repeated_blocks(U0,uh)
  vh_blocks = repeated_blocks(V0,vh)
  res = DomainContribution()
  for (liform,uhi,vhi) in zip(liforms,uh_blocks,vh_blocks)
    res = res + ∇(biform,[uhi,vhi,φh],3) - ∇(liform,[vhi,φh],2)
  end
  return res
end

function update_adjoint_caches!(φ_to_u::RepeatingAffineFEStateMap,uh,φh)
  adjoint_ns, adjoint_K, _, assem_adjoint, _ = φ_to_u.adj_caches
  U0, V0 = φ_to_u.spaces_0
  assemble_matrix!((u,v) -> φ_to_u.biform(v,u,φh),adjoint_K,assem_adjoint,V0,U0)
  numerical_setup!(adjoint_ns,adjoint_K)
  return φ_to_u.adj_caches
end

function adjoint_solve!(φ_to_u::RepeatingAffineFEStateMap,du::AbstractBlockVector)
  adjoint_ns, _, adjoint_x, _, _ = φ_to_u.adj_caches
  U0, V0 = φ_to_u.spaces_0
  map(repeated_blocks(U0,adjoint_x),repeated_blocks(U0,du)) do xi, dui
    solve!(xi,adjoint_ns,dui)
  end
  return adjoint_x
end

abstract type AbstractPDEConstrainedFunctionals{N} end

"""
    struct PDEConstrainedFunctionals{N,A} <: AbstractPDEConstrainedFunctionals{N}

An object that computes the objective, constraints, and their derivatives.

# Implementation

This implementation computes derivatives of a integral quantity

``F(u(\\varphi),\\varphi,\\mathrm{d}\\Omega_1,\\mathrm{d}\\Omega_2,...) =
\\Sigma_{i}\\int_{\\Omega_i} f_i(\\varphi)~\\mathrm{d}\\Omega``

with respect to an auxiliary parameter ``\\varphi`` where ``u``
is the solution to a PDE and implicitly depends on ``\\varphi``.
This requires two pieces of information:

 1) Computation of ``\\frac{\\partial F}{\\partial u}`` and
    ``\\frac{\\partial F}{\\partial \\varphi}`` (handled by [`StateParamMap `](@ref)).
 2) Computation of ``\\frac{\\partial F}{\\partial u}
    \\frac{\\partial u}{\\partial \\varphi}`` at ``\\varphi`` and ``u``
    using the adjoint method (handled by [`AbstractFEStateMap`](@ref)). I.e., let

    ``\\frac{\\partial F}{\\partial u}
    \\frac{\\partial u}{\\partial \\varphi} = -\\lambda^\\intercal
    \\frac{\\partial \\mathcal{R}}{\\partial \\varphi}``

    where ``\\mathcal{R}`` is the residual and solve the (linear) adjoint
    problem:

    ``\\frac{\\partial \\mathcal{R}}{\\partial u}^\\intercal\\lambda =
    \\frac{\\partial F}{\\partial u}^\\intercal.``

The gradient is then ``\\frac{\\partial F}{\\partial \\varphi} =
\\frac{\\partial F}{\\partial \\varphi} -
\\frac{\\partial F}{\\partial u}\\frac{\\partial u}{\\partial \\varphi}``.

# Parameters

- `J`: A `StateParamMap` corresponding to the objective.
- `C`: A vector of `StateParamMap` corresponding to the constraints.
- `dJ`: The DoFs for the objective sensitivity.
- `dC`: The DoFs for each constraint sensitivity.
- `analytic_dJ`: a `Function` for computing the analytic objective sensitivity.
- `analytic_dC`: A vector of `Function` for computing the analytic objective sensitivities.
- `state_map::A`: The state map for the problem.

# Note

- If `analytic_dJ = nothing` automatic differentiation will be used.
- If `analytic_dC[i] = nothing` automatic differentiation will be used for `C[i]`.
"""
struct PDEConstrainedFunctionals{N,A} <: AbstractPDEConstrainedFunctionals{N}
  J
  C
  dJ
  dC
  analytic_dJ
  analytic_dC
  state_map :: A

  @doc """
      PDEConstrainedFunctionals(objective::Function,constraints::Vector{<:Function},
        state_map::AbstractFEStateMap;analytic_dJ;analytic_dC)

  Create an instance of `PDEConstrainedFunctionals`. The arguments for the objective
  and constraints must follow the specification in [`StateParamMap`](@ref).
  By default we use automatic differentation for the objective and all constraints. This
  can be disabled by passing the shape derivative as a type `Function` to `analytic_dJ`
  and/or entires in `analytic_dC`.
  """
  function PDEConstrainedFunctionals(
      objective   :: Function,
      constraints :: Vector{<:Function},
      state_map   :: AbstractFEStateMap;
      analytic_dJ = nothing,
      analytic_dC = fill(nothing,length(constraints)))

    # Create StateParamMaps
    J = StateParamMap(objective,state_map)
    C = map(Ci -> StateParamMap(Ci,state_map),constraints)

    # Preallocate
    dJ = similar(J.caches[2])
    dC = map(Ci->similar(Ci.caches[2]),C)

    N = length(constraints)
    T = typeof(state_map)
    return new{N,T}(J,C,dJ,dC,analytic_dJ,analytic_dC,state_map)
  end

  function PDEConstrainedFunctionals(
      objective   :: AbstractStateParamMap,
      constraints :: Vector{<:AbstractStateParamMap},
      state_map   :: AbstractFEStateMap;
      analytic_dJ = nothing,
      analytic_dC = fill(nothing,length(constraints)))

    # Preallocate
    dJ = similar(objective.caches[2])
    dC = map(Ci->similar(Ci.caches[2]),constraints)

    N = length(constraints)
    T = typeof(state_map)
    return new{N,T}(objective,constraints,dJ,dC,analytic_dJ,analytic_dC,state_map)
  end
end

"""
    PDEConstrainedFunctionals(objective,state_map;analytic_dJ)

Create an instance of `PDEConstrainedFunctionals` when the problem has no constraints.
"""
PDEConstrainedFunctionals(J,state_map::AbstractFEStateMap;analytic_dJ=nothing) =
  PDEConstrainedFunctionals(J,typeof(J)[],state_map;analytic_dJ = analytic_dJ,analytic_dC = Nothing[])

get_state_map(m::PDEConstrainedFunctionals) = m.state_map
get_state(m::PDEConstrainedFunctionals) = get_state(get_state_map(m))

"""
    evaluate_functionals!(pcf::PDEConstrainedFunctionals,φh)

Evaluate the objective and constraints at `φh`.
"""
function evaluate_functionals!(pcf::PDEConstrainedFunctionals,φh;kwargs...)
  u  = get_state_map(pcf)(φh)
  U  = get_trial_space(get_state_map(pcf))
  uh = FEFunction(U,u)
  return pcf.J(uh,φh), map(Ci->Ci(uh,φh),pcf.C)
end

function evaluate_functionals!(pcf::PDEConstrainedFunctionals,φ::AbstractVector;kwargs...)
  V_φ = get_aux_space(get_state_map(pcf))
  φh = FEFunction(V_φ,φ)
  return evaluate_functionals!(pcf,φh;kwargs...)
end

"""
    evaluate_derivatives!(pcf::PDEConstrainedFunctionals,φh)

Evaluate the derivatives of the objective and constraints at `φh`.
"""
function evaluate_derivatives!(pcf::PDEConstrainedFunctionals,φh;kwargs...)
  _,_,dJ,dC = evaluate!(pcf,φh)
  return dJ,dC
end

function evaluate_derivatives!(pcf::PDEConstrainedFunctionals,φ::AbstractVector;kwargs...)
  V_φ = get_aux_space(get_state_map(pcf))
  φh = FEFunction(V_φ,φ)
  return evaluate_derivatives!(pcf,φh;kwargs...)
end

"""
    Fields.evaluate!(pcf::PDEConstrainedFunctionals,φh)

Evaluate the objective and constraints, and their derivatives at
`φh`.
"""
function Fields.evaluate!(pcf::PDEConstrainedFunctionals,φh;kwargs...)
  J, C, dJ, dC = pcf.J,pcf.C,pcf.dJ,pcf.dC
  analytic_dJ  = pcf.analytic_dJ
  analytic_dC  = pcf.analytic_dC
  U = get_trial_space(get_state_map(pcf))

  U_reg = get_deriv_space(get_state_map(pcf))
  deriv_assem = get_deriv_assembler(get_state_map(pcf))

  ## Foward problem
  u, u_pullback = rrule(get_state_map(pcf),φh)
  uh = FEFunction(U,u)

  function ∇!(F::AbstractStateParamMap,dF,::Nothing)
    # Automatic differentation
    j_val, j_pullback = rrule(F,uh,φh)   # Compute functional and pull back
    _, dFdu, dFdφ     = j_pullback(1)    # Compute dFdu, dFdφ
    _, dφ_adj         = u_pullback(dFdu) # Compute -dFdu*dudφ via adjoint
    copy!(dF,dφ_adj)
    dF .+= dFdφ
    return j_val
  end
  function ∇!(F::AbstractStateParamMap,dF,dF_analytic::Function)
    # Analytic shape derivative
    j_val = F(uh,φh)
    _dF(q) = dF_analytic(q,uh,φh)
    assemble_vector!(_dF,dF,deriv_assem,U_reg)
    return j_val
  end
  j = ∇!(J,dJ,analytic_dJ)
  c = map(∇!,C,dC,analytic_dC)

  return j,c,dJ,dC
end

function Fields.evaluate!(pcf::PDEConstrainedFunctionals,φ::AbstractVector;kwargs...)
  V_φ = get_aux_space(get_state_map(pcf))
  φh = FEFunction(V_φ,φ)
  return evaluate!(pcf,φh;kwargs...)
end

"""
    mutable struct EmbeddedPDEConstrainedFunctionals{N} <: AbstractPDEConstrainedFunctionals{N}

A mutable version of `PDEConstrainedFunctionals` that allows `state_map` to be
updated given new FE spaces for the forward problem. This is currently required
for body-fitted mesh methods and unfitted methods.
"""
struct EmbeddedPDEConstrainedFunctionals{N,T} <: AbstractPDEConstrainedFunctionals{N}
  dJ
  dC
  analytic_dJ
  analytic_dC
  embedded_collection

  @doc """
      EmbeddedPDEConstrainedFunctionals(objective::Function,constraints::Vector{<:Function},
        embedded_collection :: EmbeddedCollection;analytic_dJ;analytic_dC)

  Create an instance of `EmbeddedPDEConstrainedFunctionals`.
  """
  function EmbeddedPDEConstrainedFunctionals(
      embedded_collection :: EmbeddedCollection;
      analytic_dJ = nothing,
      analytic_dC = nothing)

    @assert Set((:state_map,:J,:C)) == keys(embedded_collection.objects) """
    Expected EmbeddedCollection to have objects ':state_map,:J,:C'. Ensure that you
    have updated the collection after adding new recipes.

    You have $(keys(embedded_collection.objects))

    Note:
    - We require that this EmbeddedCollection is seperate to the one used for the
      UnfittedEvolution. This is because updating the FEStateMap is more expensive than
      cutting and there are instances where evolution and reinitialisation happen
      at before recomputing the forward solution. As such, we cut an extra time
      to avoid allocating the state map more often then required.
    - For problems with no constraints `:C` must at least point to an empty list
    """
    # Preallocate
    dJ = similar(embedded_collection.J.caches[2])
    dC = map(Ci->similar(Ci.caches[2]),embedded_collection.C)

    N = length(embedded_collection.C)
    if analytic_dC isa Nothing
      analytic_dC = fill(nothing,length(N))
    end

    T = typeof(embedded_collection.state_map)
    return new{N,T}(dJ,dC,analytic_dJ,analytic_dC,embedded_collection)
  end
end

get_state_map(m::EmbeddedPDEConstrainedFunctionals) = m.embedded_collection.state_map
get_state(m::EmbeddedPDEConstrainedFunctionals) = get_state(get_state_map(m))

"""
    evaluate_functionals!(pcf::EmbeddedPDEConstrainedFunctionals,φh)

Evaluate the objective and constraints at `φh`.

!!! warning
    Taking `update_space = false` will NOT update the underlying finite element
    spaces and assemblers that depend on `φh`. This should only be used
    when you are certain that `φh` has not been updated.
"""
function evaluate_functionals!(pcf::EmbeddedPDEConstrainedFunctionals,φh;update_space::Bool=true)
  update_space && update_collection!(pcf.embedded_collection,φh)
  u  = get_state_map(pcf)(φh)
  U  = get_trial_space(get_state_map(pcf))
  uh = FEFunction(U,u)
  J = pcf.embedded_collection.J
  C = pcf.embedded_collection.C
  return J(uh,φh), map(Ci->Ci(uh,φh),C)
end

function evaluate_functionals!(pcf::EmbeddedPDEConstrainedFunctionals,φ::AbstractVector;kwargs...)
  V_φ = get_aux_space(get_state_map(pcf))
  φh  = FEFunction(V_φ,φ)
  return evaluate_functionals!(pcf,φh;kwargs...)
end

"""
    evaluate_derivatives!(pcf::EmbeddedPDEConstrainedFunctionals,φh)

Evaluate the derivatives of the objective and constraints at `φh`.

!!! warning
    Taking `update_space = false` will NOT update the underlying finite element
    spaces and assemblers that depend on `φh`. This should only be used
    when you are certain that `φh` has not been updated.
"""
function evaluate_derivatives!(pcf::EmbeddedPDEConstrainedFunctionals,φh;update_space::Bool=true)
  update_space && update_collection!(pcf.embedded_collection,φh)
  _,_,dJ,dC = evaluate!(pcf,φh)
  return dJ,dC
end

function evaluate_derivatives!(pcf::EmbeddedPDEConstrainedFunctionals,φ::AbstractVector;kwargs...)
  V_φ = get_aux_space(get_state_map(pcf))
  φh = FEFunction(V_φ,φ)
  return evaluate_derivatives!(pcf,φh;kwargs...)
end

"""
    Fields.evaluate!(pcf::EmbeddedPDEConstrainedFunctionals,φh)

Evaluate the objective and constraints, and their derivatives at
`φh`.

!!! warning
    Taking `update_space = false` will NOT update the underlying finite element
    spaces and assemblers that depend on `φh`. This should only be used
    when you are certain that `φh` has not been updated.
"""
function Fields.evaluate!(pcf::EmbeddedPDEConstrainedFunctionals,φh;update_space::Bool=true)
  update_space && update_collection!(pcf.embedded_collection,φh)

  J           = pcf.embedded_collection.J
  C           = pcf.embedded_collection.C
  dJ          = pcf.dJ
  dC          = pcf.dC
  analytic_dJ = pcf.analytic_dJ
  analytic_dC = pcf.analytic_dC
  state_map   = get_state_map(pcf)
  U           = get_trial_space(state_map)

  U_reg = get_deriv_space(state_map)
  deriv_assem = get_deriv_assembler(state_map)

  ## Foward problem
  u, u_pullback = rrule(state_map,φh)
  uh = FEFunction(U,u)

  function ∇!(F::StateParamMap,dF,::Nothing)
    # Automatic differentation
    j_val, j_pullback = rrule(F,uh,φh)   # Compute functional and pull back
    _, dFdu, dFdφ     = j_pullback(1)    # Compute dFdu, dFdφ
    _, dφ_adj         = u_pullback(dFdu) # Compute -dFdu*dudφ via adjoint
    copy!(dF,dφ_adj)
    dF .+= dFdφ
    return j_val
  end
  function ∇!(F::StateParamMap,dF,dF_analytic::Function)
    # Analytic shape derivative
    j_val = F(uh,φh)
    _dF(q) = dF_analytic(q,uh,φh)
    assemble_vector!(_dF,dF,deriv_assem,U_reg)
    return j_val
  end
  j = ∇!(J,dJ,analytic_dJ)
  c = map(∇!,C,dC,analytic_dC)

  return j,c,dJ,dC
end

function Fields.evaluate!(pcf::EmbeddedPDEConstrainedFunctionals,φ::AbstractVector;kwargs...)
  V_φ = get_aux_space(get_state_map(pcf))
  φh = FEFunction(V_φ,φ)
  return evaluate!(pcf,φh;kwargs...)
end

# IO

function Base.show(io::IO,object::StateParamMap)
  print(io,"$(nameof(typeof(object)))")
end

function Base.show(io::IO,object::AbstractFEStateMap)
  print(io,"$(nameof(typeof(object)))")
end

function Base.show(io::IO,::MIME"text/plain",f::AbstractPDEConstrainedFunctionals{N}) where N
  print(io,"$(nameof(typeof(f))):
    num_constraints: $N")
end

struct CustomPDEConstrainedFunctionals{N,A} #<: ParameterisedObjective{N,A}
  φ_to_jc :: Function
  dJ :: Vector{Float64}
  dC :: Vector{Vector{Float64}}
  analytic_dJ
  analytic_dC
  state_map :: A
  V_φ :: FESpace

    function CustomPDEConstrainedFunctionals(
      φ_to_jc :: Function,
      state_map :: AbstractFEStateMap,
      φh_bg;
    )

    V_φ = φh_bg.fe_space
    φh = interpolate(φh_bg,get_aux_space(state_map))
    φ = φh.free_values
    
    # Pre-allocaitng
    grad = Zygote.jacobian(φ_to_jc, φ)
    dJ = grad[1][1,:]
    dC = [collect(row) for row in eachrow(grad[1][2:end,:])]    

    N = length(dC)
    A = typeof(state_map)
    analytic_dJ = nothing
    analytic_dC = fill(nothing,N)

    return new{N,A}(φ_to_jc,dJ,dC,analytic_dJ,analytic_dC,state_map,V_φ)
  end
end

function Fields.evaluate!(pcf::CustomPDEConstrainedFunctionals,φh_bg)
  φ_to_jc,dJ,dC = pcf.φ_to_jc,pcf.dJ,pcf.dC

  φh = interpolate(φh_bg,get_aux_space(pcf.state_map))

  obj,grad = Zygote.withjacobian(φ_to_jc, φh.free_values)
  j = obj[1]
  c = obj[2:end]
  copy!(dJ,grad[1][1,:])
  copy!(dC,[collect(row) for row in eachrow(grad[1][2:end,:])])

  return j,c,dJ,dC
end

get_state(m::CustomPDEConstrainedFunctionals) = get_state(m.state_map)

function evaluate_functionals!(pcf::CustomPDEConstrainedFunctionals,φh::FEFunction)
  φ = φh.free_values
  return evaluate_functionals!(pcf,φ)
end

function evaluate_functionals!(pcf::CustomPDEConstrainedFunctionals,φ_bg::AbstractVector)
  φ_to_jc =  pcf.φ_to_jc
  φh_bg = FEFunction(pcf.V_φ,φ_bg)
  φh = interpolate(φh_bg,get_aux_space(pcf.state_map))
  φ = φh.free_values
  obj = φ_to_jc(φ)
  j = obj[1]
  c = obj[2:end]
  return j,c
end