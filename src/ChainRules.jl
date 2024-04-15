"""
    struct IntegrandWithMeasure{A,B<:Tuple}

A wrapper to enable serial or parallel partial differentation of an 
integral `F` using `Gridap.gradient`. This is required to allow automatic 
differentation with `DistributedMeasure`.

# Properties
- `F  :: A`: A function that returns a `DomainContribution` or `DistributedDomainContribution`.
- `dΩ :: B`: A tuple of measures.
"""
struct IntegrandWithMeasure{A,B<:Tuple}
  F  :: A
  dΩ :: B
end

"""
    (F::IntegrandWithMeasure)(args...)

Evaluate `F.F` given arguments `args`.
"""
(F::IntegrandWithMeasure)(args...) = F.F(args...,F.dΩ...)

"""
    Gridap.gradient(F::IntegrandWithMeasure,uh::Vector,K::Int)

Given an an `IntegrandWithMeasure` `F` and a vector of `FEFunctions` `uh` (excluding measures)
evaluate the partial derivative of `F.F` with respect to `uh[K]`.

# Example

Suppose `uh` and `φh` are FEFunctions with measures `dΩ` and `dΓ_N`.
Then the partial derivative of a function `J` wrt to `φh` is computed via  
````
J(u,φ,dΩ,dΓ_N) = ∫(f(u,φ))dΩ + ∫(g(u,φ))dΓ_N
J_iwm = IntegrandWithMeasure(J,(dΩ,dΓ_N))
∂J∂φh = ∇(J_iwm,[uh,φh],2)
````
where `f` and `g` are user defined. 
"""
function Gridap.gradient(F::IntegrandWithMeasure,uh::Vector{<:FEFunction},K::Int)
  @check 0 < K <= length(uh)
  _f(uk) = F.F(uh[1:K-1]...,uk,uh[K+1:end]...,F.dΩ...)
  return Gridap.gradient(_f,uh[K])
end

function Gridap.gradient(F::IntegrandWithMeasure,uh::Vector,K::Int)
  @check 0 < K <= length(uh)
  local_fields = map(local_views,uh) |> to_parray_of_arrays
  local_measures = map(local_views,F.dΩ) |> to_parray_of_arrays
  contribs = map(local_measures,local_fields) do dΩ,lf
    # TODO: Remove second term below, this is a fix for the problem discussed in 
    #  https://github.com/zjwegert/LSTO_Distributed/issues/46
    _f = u -> F.F(lf[1:K-1]...,u,lf[K+1:end]...,dΩ...) #+ ∑(∫(0)dΩ[i] for i = 1:length(dΩ))
    return Gridap.Fields.gradient(_f,lf[K])
  end
  return DistributedDomainContribution(contribs)
end

Gridap.gradient(F::IntegrandWithMeasure,uh) = Gridap.gradient(F,[uh],1)

"""
    Gridap.jacobian(F::IntegrandWithMeasure,uh::Vector,K::Int)

Given an an `IntegrandWithMeasure` `F` and a vector of `FEFunctions` or `CellField` `uh` 
(excluding measures) evaluate the Jacobian `F.F` with respect to `uh[K]`.
"""
function Gridap.jacobian(F::IntegrandWithMeasure,uh::Vector{<:Union{FEFunction,CellField}},K::Int)
  @check 0 < K <= length(uh)
  _f(uk) = F.F(uh[1:K-1]...,uk,uh[K+1:end]...,F.dΩ...)
  return Gridap.jacobian(_f,uh[K])
end

function Gridap.jacobian(F::IntegrandWithMeasure,uh::Vector,K::Int)
  @check 0 < K <= length(uh)
  local_fields = map(local_views,uh) |> to_parray_of_arrays
  local_measures = map(local_views,F.dΩ) |> to_parray_of_arrays
  contribs = map(local_measures,local_fields) do dΩ,lf
    _f = u -> F.F(lf[1:K-1]...,u,lf[K+1:end]...,dΩ...)
    return Gridap.jacobian(_f,lf[K])
  end
  return DistributedDomainContribution(contribs)
end

Gridap.jacobian(F::IntegrandWithMeasure,uh) = Gridap.jacobian(F,[uh],1)

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

"""
    struct StateParamIntegrandWithMeasure{A<:IntegrandWithMeasure,B,C,D}

A wrapper to handle partial differentation of an [`IntegrandWithMeasure`](@ref)
of a specific form (see below) in a `ChainRules.jl` compatible way with caching.

# Assumptions

We assume that we have a `IntegrandWithMeasure` of the following form:

`F(u,φ,dΩ₁,dΩ₂,...) = ∫(f(u,φ))dΩ₁ + ∫(g(u,φ))dΩ₂ + ...,`.

where `u` and `φ` are each expected to inherit from `Union{FEFunction,MultiFieldFEFunction}`
or the GridapDistributed equivalent.
"""
struct StateParamIntegrandWithMeasure{A<:IntegrandWithMeasure,B,C,D}
  F       :: A
  spaces  :: B
  assems  :: C
  caches  :: D
end

"""
    StateParamIntegrandWithMeasure(F::IntegrandWithMeasure,U::FESpace,V_φ::FESpace,
    U_reg::FESpace,assem_U::Assembler,assem_deriv::Assembler)

Create an instance of `StateParamIntegrandWithMeasure`. 
"""
function StateParamIntegrandWithMeasure(
  F::IntegrandWithMeasure,
  U::FESpace,V_φ::FESpace,U_reg::FESpace,
  assem_U::Assembler,assem_deriv::Assembler
)
  φ₀, u₀ = zero(V_φ), zero(U)
  ∂j∂u_vecdata = collect_cell_vector(U,∇(F,[u₀,φ₀],1))
  ∂j∂φ_vecdata = collect_cell_vector(U_reg,∇(F,[u₀,φ₀],2))
  ∂j∂u_vec = allocate_vector(assem_U,∂j∂u_vecdata)
  ∂j∂φ_vec = allocate_vector(assem_deriv,∂j∂φ_vecdata)
  assems = (assem_U,assem_deriv)
  spaces = (U,V_φ,U_reg)
  caches = (∂j∂u_vec,∂j∂φ_vec)
  return StateParamIntegrandWithMeasure(F,spaces,assems,caches)
end

"""
    (u_to_j::StateParamIntegrandWithMeasure)(uh,φh)

Evaluate the `StateParamIntegrandWithMeasure` at parameters `uh` and `φh`.  
"""
(u_to_j::StateParamIntegrandWithMeasure)(uh,φh) = sum(u_to_j.F(uh,φh))

function (u_to_j::StateParamIntegrandWithMeasure)(u::AbstractVector,φ::AbstractVector)
  U,V_φ,_ = u_to_j.spaces
  uh = FEFunction(U,u)
  φh = FEFunction(V_φ,φ)
  return u_to_j(uh,φh)
end

"""
    ChainRulesCore.rrule(u_to_j::StateParamIntegrandWithMeasure,uh,φh)

Return the evaluation of a `StateParamIntegrandWithMeasure` and a 
a function for evaluating the pullback of `u_to_j`. This enables
compatiblity with `ChainRules.jl`
"""
function ChainRulesCore.rrule(u_to_j::StateParamIntegrandWithMeasure,uh,φh)
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

function ChainRulesCore.rrule(u_to_j::StateParamIntegrandWithMeasure,u::AbstractVector,φ::AbstractVector)
  U,V_φ,U_reg = u_to_j.spaces
  uh = FEFunction(U,u)
  φh = FEFunction(V_φ,φ)
  return ChainRulesCore.rrule(u_to_j,uh,φh)
end

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
    get_measure(m::AbstractFEStateMap)

Return the measures associated with the FE problem.
"""
get_measure(::AbstractFEStateMap) = @abstractmethod

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

function forward_solve!(φ_to_u::AbstractFEStateMap,φ::AbstractVector)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return forward_solve!(φ_to_u,φh)
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

"""
    StateParamIntegrandWithMeasure(f::Function,φ_to_u::AbstractFEStateMap)

Create an instance of `StateParamIntegrandWithMeasure` given a `f` and
`φ_to_u`.
"""
function StateParamIntegrandWithMeasure(f::Function,φ_to_u::AbstractFEStateMap)
  dΩ = get_measure(φ_to_u)
  F  = IntegrandWithMeasure(f,dΩ)
  StateParamIntegrandWithMeasure(F,φ_to_u)
end

function StateParamIntegrandWithMeasure(F::IntegrandWithMeasure,φ_to_u::AbstractFEStateMap)
  U,V,V_φ,U_reg = φ_to_u.spaces
  assem_deriv = get_deriv_assembler(φ_to_u)
  assem_U = get_pde_assembler(φ_to_u)
  StateParamIntegrandWithMeasure(F,U,V_φ,U_reg,assem_U,assem_deriv)
end

"""
    struct AffineFEStateMap{A,B,C,D,E,F} <: AbstractFEStateMap

A structure to enable the forward problem and pullback for affine finite
element operators `AffineFEOperator`.

# Parameters

- `biform::A`: `IntegrandWithMeasure` defining the bilinear form.
- `liform::B`: `IntegrandWithMeasure` defining the linear form.
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
        U,V,V_φ,U_reg,φh,dΩ...;
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
    a::Function,l::Function,
    U,V,V_φ,U_reg,φh,dΩ...;
    assem_U = SparseMatrixAssembler(U,V),
    assem_adjoint = SparseMatrixAssembler(V,U),
    assem_deriv = SparseMatrixAssembler(U_reg,U_reg),
    ls::LinearSolver = LUSolver(),
    adjoint_ls::LinearSolver = LUSolver()
  )
    # TODO: I really want to get rid of the φh argument...

    biform = IntegrandWithMeasure(a,dΩ)
    liform = IntegrandWithMeasure(l,dΩ)
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
    fwd_caches = (ns,K,b,x,uhd,assem_U)

    ## Adjoint cache
    adjoint_K  = assemble_matrix((u,v)->biform(v,u,φh),assem_adjoint,V,U)
    adjoint_x  = allocate_in_domain(adjoint_K); fill!(adjoint_x,zero(eltype(adjoint_x)))
    adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
    adj_caches = (adjoint_ns,adjoint_K,adjoint_x,assem_adjoint)

    A,B,C = typeof(biform), typeof(liform), typeof(spaces)
    D,E,F = typeof(plb_caches),typeof(fwd_caches), typeof(adj_caches)
    return new{A,B,C,D,E,F}(biform,liform,spaces,plb_caches,fwd_caches,adj_caches)
  end
end

# Getters
get_state(m::AffineFEStateMap) = FEFunction(get_trial_space(m),m.fwd_caches[4])
get_measure(m::AffineFEStateMap) = m.biform.dΩ
get_spaces(m::AffineFEStateMap) = m.spaces
get_assemblers(m::AffineFEStateMap) = (m.fwd_caches[6],m.plb_caches[2],m.adj_caches[4])

function forward_solve!(φ_to_u::AffineFEStateMap,φh)
  biform, liform = φ_to_u.biform, φ_to_u.liform
  U, V, _, _ = φ_to_u.spaces
  ns, K, b, x, uhd, assem_U = φ_to_u.fwd_caches

  a_fwd(u,v) = biform(u,v,φh)
  l_fwd(v)   = liform(v,φh)
  assemble_matrix_and_vector!(a_fwd,l_fwd,K,b,assem_U,U,V,uhd)
  numerical_setup!(ns,K)
  solve!(x,ns,b)
  return x
end

function dRdφ(φ_to_u::AffineFEStateMap,uh,vh,φh)
  biform, liform = φ_to_u.biform, φ_to_u.liform
  return ∇(biform,[uh,vh,φh],3) - ∇(liform,[vh,φh],2)
end

function update_adjoint_caches!(φ_to_u::AffineFEStateMap,uh,φh)
  adjoint_ns, adjoint_K, _, assem_adjoint = φ_to_u.adj_caches
  U, V, _, _ = φ_to_u.spaces
  assemble_matrix!((u,v) -> φ_to_u.biform(v,u,φh),adjoint_K,assem_adjoint,V,U)
  numerical_setup!(adjoint_ns,adjoint_K)
  return φ_to_u.adj_caches
end

function adjoint_solve!(φ_to_u::AffineFEStateMap,du::AbstractVector)
  adjoint_ns, _, adjoint_x, _ = φ_to_u.adj_caches
  solve!(adjoint_x,adjoint_ns,du)
  return adjoint_x
end

"""
    struct NonlinearFEStateMap{A,B,C,D,E,F} <: AbstractFEStateMap

A structure to enable the forward problem and pullback for nonlinear finite
element operators.

# Parameters

- `res::A`: an `IntegrandWithMeasure` defining the residual of the problem.
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
        res::Function,U,V,V_φ,U_reg,φh,dΩ...;
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
    res::Function,U,V,V_φ,U_reg,φh,dΩ...;
    assem_U = SparseMatrixAssembler(U,V),
    assem_adjoint = SparseMatrixAssembler(V,U),
    assem_deriv = SparseMatrixAssembler(U_reg,U_reg),
    nls::NonlinearSolver = NewtonSolver(LUSolver();maxiter=50,rtol=1.e-8,verbose=true),
    adjoint_ls::LinearSolver = LUSolver()
  )
    res = IntegrandWithMeasure(res,dΩ)
    jac = (u,du,dv,φh) -> jacobian(res,[u,dv,φh],1)
    spaces = (U,V,V_φ,U_reg)

    ## Pullback cache
    uhd = zero(U)
    vecdata = collect_cell_vector(U_reg,∇(res,[uhd,uhd,φh],3))
    dudφ_vec = allocate_vector(assem_deriv,vecdata)
    plb_caches = (dudφ_vec,assem_deriv)

    ## Forward cache
    x = zero_free_values(U)
    _res(u,v) = res(u,v,φh)
    _jac(u,du,dv) = jac(u,du,dv,φh)
    op = get_algebraic_operator(FEOperator(_res,_jac,U,V,assem_U))
    nls_cache = instantiate_caches(x,nls,op)
    fwd_caches = (nls,nls_cache,x,assem_U)

    ## Adjoint cache
    _jac_adj(du,dv) = jac(uhd,du,dv,φh)
    adjoint_K  = assemble_adjoint_matrix(_jac_adj,assem_adjoint,U,V)
    adjoint_x  = allocate_in_domain(adjoint_K); fill!(adjoint_x,zero(eltype(adjoint_x)))
    adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
    adj_caches = (adjoint_ns,adjoint_K,adjoint_x,assem_adjoint)
    
    A, B, C = typeof(res), typeof(jac), typeof(spaces)
    D, E, F = typeof(plb_caches), typeof(fwd_caches), typeof(adj_caches)
    return new{A,B,C,D,E,F}(res,jac,spaces,plb_caches,fwd_caches,adj_caches)
  end
end

get_state(m::NonlinearFEStateMap) = FEFunction(get_trial_space(m),m.fwd_caches[3])
get_measure(m::NonlinearFEStateMap) = m.res.dΩ
get_spaces(m::NonlinearFEStateMap) = m.spaces
get_assemblers(m::NonlinearFEStateMap) = (m.fwd_caches[4],m.plb_caches[2],m.adj_caches[4])

function forward_solve!(φ_to_u::NonlinearFEStateMap,φh)
  U, V, _, _ = φ_to_u.spaces
  nls, nls_cache, x, assem_U = φ_to_u.fwd_caches

  res(u,v) = φ_to_u.res(u,v,φh)
  jac(u,du,dv) = φ_to_u.jac(u,du,dv,φh)
  op = get_algebraic_operator(FEOperator(res,jac,U,V,assem_U))
  solve!(x,nls,op,nls_cache)
  return x
end

function dRdφ(φ_to_u::NonlinearFEStateMap,uh,vh,φh)
  res = φ_to_u.res
  return ∇(res,[uh,vh,φh],3)
end

function update_adjoint_caches!(φ_to_u::NonlinearFEStateMap,uh,φh)
  adjoint_ns, adjoint_K, _, assem_adjoint = φ_to_u.adj_caches
  U, V, _, _ = φ_to_u.spaces
  jac(du,dv) =  φ_to_u.jac(uh,du,dv,φh)
  assemble_adjoint_matrix!(jac,adjoint_K,assem_adjoint,U,V)
  numerical_setup!(adjoint_ns,adjoint_K)
  return φ_to_u.adj_caches
end

function adjoint_solve!(φ_to_u::NonlinearFEStateMap,du::AbstractVector)
  adjoint_ns, _, adjoint_x, _ = φ_to_u.adj_caches
  solve!(adjoint_x,adjoint_ns,du)
  return adjoint_x
end

"""
    struct RepeatingAffineFEStateMap <: AbstractFEStateMap

A structure to enable the forward problem and pullback for affine finite
element operators `AffineFEOperator` with multiple linear forms but only
a single bilinear form.

# Parameters

- `biform`: `IntegrandWithMeasure` defining the bilinear form.
- `liform`: A vector of `IntegrandWithMeasure` defining the linear forms.
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
        U0,V0,V_φ,U_reg,φh,dΩ...;
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
    nblocks::Int,a::Function,l::Vector{<:Function},
    U0,V0,V_φ,U_reg,φh,dΩ...;
    assem_U = SparseMatrixAssembler(U0,V0),
    assem_adjoint = SparseMatrixAssembler(V0,U0),
    assem_deriv = SparseMatrixAssembler(U_reg,U_reg),
    ls::LinearSolver = LUSolver(),
    adjoint_ls::LinearSolver = LUSolver()
  )
    @check nblocks == length(l)

    spaces_0 = (U0,V0)
    assem_U0 = assem_U

    biform = IntegrandWithMeasure(a,dΩ)
    liforms = map(li -> IntegrandWithMeasure(li,dΩ),l)
    U, V = repeat_spaces(nblocks,U0,V0)
    spaces = (U,V,V_φ,U_reg)
    assem_U = SparseMatrixAssembler(
      get_local_matrix_type(assem_U0), get_local_vector_type(assem_U0),
      U, V, get_assembly_strategy(assem_U0)
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
    x  = mortar(map(i -> allocate_in_domain(K), 1:nblocks)); fill!(x,zero(eltype(x)))
    ns = numerical_setup(symbolic_setup(ls,K),K)
    fwd_caches = (ns,K,b,x,uhd,assem_U0,b0,assem_U)

    ## Adjoint cache
    adjoint_K  = assemble_matrix((u,v)->biform(v,u,φh),assem_adjoint,V0,U0)
    adjoint_x  = mortar(map(i -> allocate_in_domain(adjoint_K), 1:nblocks)); fill!(adjoint_x,zero(eltype(adjoint_x)))
    adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
    adj_caches = (adjoint_ns,adjoint_K,adjoint_x,assem_adjoint)

    A,B,C,D = typeof(biform), typeof(liforms), typeof(spaces), typeof(spaces_0)
    E,F,G = typeof(plb_caches), typeof(fwd_caches), typeof(adj_caches)
    return new{A,B,C,D,E,F,G}(biform,liforms,spaces,spaces_0,plb_caches,fwd_caches,adj_caches)
  end
end

function repeat_spaces(nblocks::Integer,U0::FESpace,V0::FESpace)
  U = MultiFieldFESpace([U0 for i in 1:nblocks];style=BlockMultiFieldStyle())
  V = MultiFieldFESpace([V0 for i in 1:nblocks];style=BlockMultiFieldStyle())
  return U,V
end

function repeat_spaces(
  nblocks::Integer,U0::T,V0::T
) where T <: Union{MultiField.MultiFieldFESpace,GridapDistributed.DistributedMultiFieldFESpace}
  nfields = num_fields(U0)
  @assert nfields == num_fields(V0)
  mfs = BlockMultiFieldStyle(nblocks,Tuple(fill(nfields,nblocks)))
  U = MultiFieldFESpace(repeat([U0...],nblocks);style=mfs)
  V = MultiFieldFESpace(repeat([V0...],nblocks);style=mfs)
  return U,V
end

get_state(m::RepeatingAffineFEStateMap) = FEFunction(get_trial_space(m),m.fwd_caches[4])
get_measure(m::RepeatingAffineFEStateMap) = m.biform.dΩ
get_spaces(m::RepeatingAffineFEStateMap) = m.spaces
get_assemblers(m::RepeatingAffineFEStateMap) = (m.fwd_caches[8],m.plb_caches[2],m.adj_caches[4])

function forward_solve!(φ_to_u::RepeatingAffineFEStateMap,φh)
  biform, liforms = φ_to_u.biform, φ_to_u.liform
  U0, V0 = φ_to_u.spaces_0
  ns, K, b, x, uhd, assem_U0, b0, _ = φ_to_u.fwd_caches

  a_fwd(u,v) = biform(u,v,φh)
  assemble_matrix!(a_fwd,K,assem_U0,U0,V0)
  numerical_setup!(ns,K)

  l0_fwd(v) = a_fwd(uhd,v)
  assemble_vector!(l0_fwd,b0,assem_U0,V0)
  rmul!(b0,-1)

  v = get_fe_basis(V0)
  map(blocks(x),liforms) do xi, li
    copy!(b,b0)
    vecdata = collect_cell_vector(V0,li(v,φh))
    assemble_vector_add!(b,assem_U0,vecdata)
    solve!(xi,ns,b)
  end
  return x
end

function dRdφ(φ_to_u::RepeatingAffineFEStateMap,uh,vh,φh)
  biform, liforms = φ_to_u.biform, φ_to_u.liform

  res = DomainContribution()
  for (liform,uhi,vhi) in zip(liforms,uh,vh)
    res = res + ∇(biform,[uhi,vhi,φh],3) - ∇(liform,[vhi,φh],2)
  end
  return res
end

function update_adjoint_caches!(φ_to_u::RepeatingAffineFEStateMap,uh,φh)
  adjoint_ns, adjoint_K, _, assem_adjoint = φ_to_u.adj_caches
  U0, V0 = φ_to_u.spaces_0
  assemble_matrix!((u,v) -> φ_to_u.biform(v,u,φh),adjoint_K,assem_adjoint,V0,U0)
  numerical_setup!(adjoint_ns,adjoint_K)
  return φ_to_u.adj_caches
end

function adjoint_solve!(φ_to_u::RepeatingAffineFEStateMap,du::AbstractBlockVector)
  adjoint_ns, _, adjoint_x, _ = φ_to_u.adj_caches
  map(blocks(adjoint_x),blocks(du)) do xi, dui
    solve!(xi,adjoint_ns,dui)
  end
  return adjoint_x
end

"""
    struct PDEConstrainedFunctionals{N,A}

An object that computes the objective, constraints, and their derivatives. 

# Implementation

This implementation computes derivatives of a integral quantity 

``F(u(\\varphi),\\varphi,\\mathrm{d}\\Omega_1,\\mathrm{d}\\Omega_2,...) = 
\\Sigma_{i}\\int_{\\Omega_i} f_i(\\varphi)~\\mathrm{d}\\Omega`` 

with respect to an auxiliary parameter ``\\varphi`` where ``u``
is the solution to a PDE and implicitly depends on ``\\varphi``. 
This requires two pieces of information:

 1) Computation of ``\\frac{\\partial F}{\\partial u}`` and 
    ``\\frac{\\partial F}{\\partial \\varphi}`` (handled by [`StateParamIntegrandWithMeasure `](@ref)).
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

- `J`: A `StateParamIntegrandWithMeasure` corresponding to the objective.
- `C`: A vector of `StateParamIntegrandWithMeasure` corresponding to the constraints.
- `dJ`: The DoFs for the objective sensitivity.
- `dC`: The DoFs for each constraint sensitivity.
- `analytic_dJ`: a `Function` for computing the analytic objective sensitivity.
- `analytic_dC`: A vector of `Function` for computing the analytic objective sensitivities.
- `state_map::A`: The state map for the problem.

# Note

- If `analytic_dJ = nothing` automatic differentiation will be used.
- If `analytic_dC[i] = nothing` automatic differentiation will be used for `C[i]`.
"""
struct PDEConstrainedFunctionals{N,A}
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
  and constraints must follow the specification in [`StateParamIntegrandWithMeasure`](@ref).
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

    # Create StateParamIntegrandWithMeasures
    J = StateParamIntegrandWithMeasure(objective,state_map)
    C = map(Ci -> StateParamIntegrandWithMeasure(Ci,state_map),constraints)
    
    # Preallocate
    dJ = similar(J.caches[2])
    dC = map(Ci->similar(Ci.caches[2]),C)

    N = length(constraints)
    T = typeof(state_map)
    return new{N,T}(J,C,dJ,dC,analytic_dJ,analytic_dC,state_map)
  end
end

"""
    PDEConstrainedFunctionals(objective,state_map;analytic_dJ)

Create an instance of `PDEConstrainedFunctionals` when the problem has no constraints.
"""
PDEConstrainedFunctionals(J::Function,state_map::AbstractFEStateMap;analytic_dJ=nothing) = 
  PDEConstrainedFunctionals(J,Function[],state_map;analytic_dJ = analytic_dJ,analytic_dC = Nothing[])

get_state(m::PDEConstrainedFunctionals) = get_state(m.state_map)

"""
    evaluate_functionals!(pcf::PDEConstrainedFunctionals,φh)

Evaluate the objective and constraints at `φh`.
"""
function evaluate_functionals!(pcf::PDEConstrainedFunctionals,φh)
  u  = pcf.state_map(φh)
  U  = get_trial_space(pcf.state_map)
  uh = FEFunction(U,u)
  return pcf.J(uh,φh), map(Ci->Ci(uh,φh),pcf.C)
end

function evaluate_functionals!(pcf::PDEConstrainedFunctionals,φ::AbstractVector)
  V_φ = get_aux_space(pcf.state_map)
  φh = FEFunction(V_φ,φ)
  return evaluate_functionals!(pcf,φh)
end

"""
    evaluate_derivatives!(pcf::PDEConstrainedFunctionals,φh)

Evaluate the derivatives of the objective and constraints at `φh`.
"""
function evaluate_derivatives!(pcf::PDEConstrainedFunctionals,φh)
  _,_,dJ,dC = evaluate!(pcf,φh)
  return dJ,dC
end

function evaluate_derivatives!(pcf::PDEConstrainedFunctionals,φ::AbstractVector)
  V_φ = get_aux_space(pcf.state_map)
  φh = FEFunction(V_φ,φ)
  return evaluate_derivatives!(pcf,φh)
end

"""
    Fields.evaluate!(pcf::PDEConstrainedFunctionals,φh)

Evaluate the objective and constraints, and their derivatives at
`φh`. 
"""
function Fields.evaluate!(pcf::PDEConstrainedFunctionals,φh)
  J, C, dJ, dC = pcf.J,pcf.C,pcf.dJ,pcf.dC
  analytic_dJ  = pcf.analytic_dJ
  analytic_dC  = pcf.analytic_dC
  U = get_trial_space(pcf.state_map)

  U_reg = get_deriv_space(pcf.state_map)
  deriv_assem = get_deriv_assembler(pcf.state_map)
  dΩ = get_measure(pcf.state_map)

  ## Foward problem
  u, u_pullback = rrule(pcf.state_map,φh)
  uh = FEFunction(U,u)

  function ∇!(F::StateParamIntegrandWithMeasure,dF,::Nothing)
    # Automatic differentation
    j_val, j_pullback = rrule(F,uh,φh)   # Compute functional and pull back
    _, dFdu, dFdφ     = j_pullback(1)    # Compute dFdu, dFdφ
    _, dφ_adj         = u_pullback(dFdu) # Compute -dFdu*dudφ via adjoint 
    copy!(dF,dφ_adj)
    dF .+= dFdφ
    return j_val
  end
  function ∇!(F::StateParamIntegrandWithMeasure,dF,dF_analytic)
    # Analytic shape derivative
    j_val = F(uh,φh)
    _dF(q) = dF_analytic(q,uh,φh,dΩ...)
    assemble_vector!(_dF,dF,deriv_assem,U_reg)
    return j_val
  end
  j = ∇!(J,dJ,analytic_dJ)
  c = map(∇!,C,dC,analytic_dC)

  return j,c,dJ,dC
end

function Fields.evaluate!(pcf::PDEConstrainedFunctionals,φ::AbstractVector)
  V_φ = get_aux_space(pcf.state_map)
  φh = FEFunction(V_φ,φ)
  return evaluate!(pcf,φh)
end

# IO

function Base.show(io::IO,object::IntegrandWithMeasure)
  print(io,"$(nameof(typeof(object)))")
end

function Base.show(io::IO,object::StateParamIntegrandWithMeasure)
  print(io,"$(nameof(typeof(object)))")
end

function Base.show(io::IO,object::AbstractFEStateMap)
  print(io,"$(nameof(typeof(object)))")
end

function Base.show(io::IO,::MIME"text/plain",f::PDEConstrainedFunctionals)
  print(io,"$(nameof(typeof(f))):
    num_constraints: $(length(f.C))")
end
