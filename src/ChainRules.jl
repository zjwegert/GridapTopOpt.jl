"""
  IntegrandWithMeasure

  Enables partial differentation of an integrand F via Gridap.gradient.
"""
struct IntegrandWithMeasure{A,B<:Tuple}
  F  :: A
  dΩ :: B
end

(F::IntegrandWithMeasure)(args...) = sum(F.F(args...,F.dΩ...))

Gridap.gradient(F::IntegrandWithMeasure,uh) = Gridap.gradient(F,[uh],1)

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
    _f = u -> F.F(lf[1:K-1]...,u,lf[K+1:end]...,dΩ...)
    return Gridap.Fields.gradient(_f,lf[K])
  end
  return DistributedDomainContribution(contribs)
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

"""
  StateParamIntegrandWithMeasure

  Assume that we have a IntegrandWithMeasure of the following form:
      F: (u,φ,[dΩ₁,dΩ₂,...]) ↦ ∫_Ω₁ f(u(φ),φ) dΩ₁ + ∫_Ω₂ g(u(φ),φ) dΩ₂ + ...,
  where u is a state field and φ is auxilary. 
  
  Assumptions:
    - The arguments to F match the weak form.
    - The argument u is the solution to an FE problem. This can be a single field or multifield.
    - There is a single auxilary field. Again, this can possibly be a MultiFieldFEFunction.
      E.g., multiple level set functions.
"""
struct StateParamIntegrandWithMeasure{A<:IntegrandWithMeasure,B<:Tuple,C<:Tuple,D<:Tuple}
  F       :: A
  spaces  :: B
  assem   :: C
  caches  :: D
end

function StateParamIntegrandWithMeasure(
    F::IntegrandWithMeasure,
    U,
    V_φ,
    U_reg,
    assem_U,
    assem_deriv)

  φ₀ = zero(V_φ); u₀ = zero(U)
  djdu_vecdata = collect_cell_vector(U,∇(F,[u₀,φ₀],1))
  djdφ_vecdata = collect_cell_vector(U_reg,∇(F,[u₀,φ₀],2))
  djdu_vec = allocate_vector(assem_U,djdu_vecdata)
  djdφ_vec = allocate_vector(assem_deriv,djdφ_vecdata)
  assems = (assem_U,assem_deriv)
  spaces = (U,V_φ,U_reg)
  caches = (djdu_vec,djdφ_vec)
  StateParamIntegrandWithMeasure(F,spaces,assems,caches)
end

function (u_to_j::StateParamIntegrandWithMeasure)(u::T1,φ::T2) where {T1<:AbstractArray,T2<:AbstractArray}
  U,V_φ,_ = u_to_j.spaces
  uh = FEFunction(U,u)
  φh = FEFunction(V_φ,φ)
  return u_to_j.F(uh,φh)
end

function ChainRulesCore.rrule(u_to_j::StateParamIntegrandWithMeasure,u::T1,φ::T2) where {T1<:AbstractArray,T2<:AbstractArray}
  F=u_to_j.F
  U,V_φ,U_reg = u_to_j.spaces
  assem_U,assem_deriv = u_to_j.assem
  djdu_vec,djdφ_vec = u_to_j.caches
  uh = FEFunction(U,u)
  φh = FEFunction(V_φ,φ)
  fields = [uh,φh];
  function u_to_j_pullback(dj)
    ## Compute ∂F/∂uh(uh) and ∂F/∂φh(φh)
    djdu = ∇(F,fields,1)
    djdu_vecdata = collect_cell_vector(U,djdu)
    assemble_vector!(djdu_vec,assem_U,djdu_vecdata)
    djdφ = ∇(F,fields,2)
    djdφ_vecdata = collect_cell_vector(U_reg,djdφ)
    assemble_vector!(djdφ_vec,assem_deriv,djdφ_vecdata)
    djdu_vec .*= dj
    djdφ_vec .*= dj
    (  NoTangent(), djdu_vec, djdφ_vec )
  end
  u_to_j(u,φ), u_to_j_pullback
end

"""
  AbstractFEStateMap
"""
abstract type AbstractFEStateMap end

# Getters
get_state(::AbstractFEStateMap) = @abstractmethod
get_measure(::AbstractFEStateMap) = @abstractmethod;
get_trial_space(::AbstractFEStateMap) = @abstractmethod;
get_test_space(::AbstractFEStateMap) = @abstractmethod;
get_aux_space(::AbstractFEStateMap) = @abstractmethod;
get_deriv_space(::AbstractFEStateMap) = @abstractmethod;
get_state_assembler(::AbstractFEStateMap) = @abstractmethod
get_deriv_assembler(::AbstractFEStateMap) = @abstractmethod

"""
  AffineFEStateMap
"""
struct AffineFEStateMap{A,B,C,D<:Tuple,E<:Tuple,F<:Tuple,G<:Tuple,H<:Tuple} <: AbstractFEStateMap
  a               ::A
  l               ::B
  res             ::C
  dΩ              ::D
  spaces          ::E
  cache           ::F
  fwd_caches      ::G
  adjoint_caches  ::H

  function AffineFEStateMap(
      a::A,l::B,res::C,
      U,V,V_φ,U_reg,φh,
      dΩ...;
      assem_U = SparseMatrixAssembler(U,V),
      assem_adjoint = SparseMatrixAssembler(V,U),
      assem_deriv = SparseMatrixAssembler(U_reg,U_reg),
      ls::LinearSolver = LUSolver(),
      adjoint_ls::LinearSolver = LUSolver()) where {A,B,C}
    
    spaces = (U,V,V_φ,U_reg)
    ## dφdu cache
    uhd = zero(U)
    vecdata = collect_cell_vector(U_reg,∇(IntegrandWithMeasure(res,dΩ),[uhd,uhd,φh],3))
    dφdu_vec = allocate_vector(assem_deriv,vecdata)

    ## K,b,x
    op = AffineFEOperator((u,v) -> a(u,v,φh,dΩ...),v -> l(v,φh,dΩ...),U,V,assem_U)
    K = get_matrix(op); b = get_vector(op); 
    x = allocate_col_vector(K)

    ## Adjoint K,b,x
    adjoint_K = assemble_matrix((u,v) -> a(v,u,φh,dΩ...),assem_adjoint,V,U)
    adjoint_x = allocate_col_vector(adjoint_K)

    ## Numerical setups
    ns = numerical_setup(symbolic_setup(ls,K),K)
    adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
    
    ## Caches and adjoint caches
    cache = (dφdu_vec,assem_deriv)
    fwd_caches = (ns,K,b,x,uhd,assem_U)
    adjoint_caches = (adjoint_ns,adjoint_K,adjoint_x,assem_adjoint)
    return new{A,B,C,typeof(dΩ),typeof(spaces),typeof(cache),typeof(fwd_caches),
      typeof(adjoint_caches)}(a,l,res,dΩ,spaces,cache,fwd_caches,adjoint_caches)
  end
end

# Getters
get_state(m::AffineFEStateMap) = FEFunction(get_trial_space(m),m.fwd_caches[4])
get_measure(m::AffineFEStateMap) = m.dΩ;
get_trial_space(m::AffineFEStateMap) = m.spaces[1];
get_test_space(m::AffineFEStateMap) = m.spaces[2];
get_aux_space(m::AffineFEStateMap) = m.spaces[3];
get_deriv_space(m::AffineFEStateMap) = m.spaces[4];
get_state_assembler(m::AffineFEStateMap) = last(m.fwd_caches)
get_deriv_assembler(m::AffineFEStateMap) = last(m.cache)

function (φ_to_u::AffineFEStateMap)(φ::T) where T <: AbstractVector
  a=φ_to_u.a
  l=φ_to_u.l
  dΩ = φ_to_u.dΩ
  U,V,V_φ,U_reg = φ_to_u.spaces
  ns,K,b,x,uhd,assem_U = φ_to_u.fwd_caches
  φh = FEFunction(V_φ,φ)
  
  ## Assemble and solve
  du = get_trial_fe_basis(U)
  dv = get_fe_basis(V);
  data = collect_cell_matrix_and_vector(U,V,a(du,dv,φh,dΩ...),l(dv,φh,dΩ...),uhd)
  assemble_matrix_and_vector!(K,b,assem_U,data)
  numerical_setup!(ns,K)
  solve!(x,ns,b)
  x
end

function ChainRulesCore.rrule(φ_to_u::AffineFEStateMap,φ::T) where T <: AbstractVector
  a=φ_to_u.a
  res = φ_to_u.res
  dΩ = φ_to_u.dΩ
  U,V,V_φ,U_reg = φ_to_u.spaces
  dφdu_vec,assem_deriv = φ_to_u.cache
  adjoint_ns,adjoint_K,λ,assem_adjoint = φ_to_u.adjoint_caches
  
  ## Forward problem
  u = φ_to_u(φ)

  ## Adjoint operator
  φh = FEFunction(V_φ,φ)
  assemble_matrix!((u,v) -> a(v,u,φh,dΩ...),adjoint_K,assem_adjoint,V,U)
  function φ_to_u_pullback(du)
    ## Adjoint Solve
    numerical_setup!(adjoint_ns,adjoint_K)
    solve!(λ,adjoint_ns,du)
    λh = FEFunction(V,λ)
    
    ## Compute grad
    uh = FEFunction(U,u)
    res_functional = IntegrandWithMeasure(res,dΩ)
    dφdu_contrib = ∇(res_functional,[uh,λh,φh],3)
    dφdu_vecdata = collect_cell_vector(U_reg,dφdu_contrib) 
    assemble_vector!(dφdu_vec,assem_deriv,dφdu_vecdata)
    dφdu_vec .*= -1;
    ( NoTangent(),dφdu_vec)
  end
  u, φ_to_u_pullback
end

"""
  NonlinearFEStateMap
"""
struct NonlinearFEStateMap{A,B<:Tuple,C<:Tuple,D<:Tuple,E<:Tuple,F<:Tuple} <: LSTO_Distributed.AbstractFEStateMap
  res             ::A
  dΩ              ::B
  spaces          ::C
  cache           ::D
  fwd_caches      ::E
  adjoint_caches  ::F

  function NonlinearFEStateMap(
      res::A,U,V,V_φ,U_reg,φh,dΩ...;
      assem_U = SparseMatrixAssembler(U,V),
      assem_adjoint = SparseMatrixAssembler(V,U),
      assem_deriv = SparseMatrixAssembler(U_reg,U_reg),
      nls = NLSolver(),
      adjoint_ls::LinearSolver = LUSolver()) where {A}
    
    spaces = (U,V,V_φ,U_reg)
    ## dφdu cache
    uhd = zero(U)
    vecdata = collect_cell_vector(U_reg,∇(IntegrandWithMeasure(res,dΩ),[uhd,uhd,φh],3))
    dφdu_vec = allocate_vector(assem_deriv,vecdata)

    ## Nonlinear FE operator and solution vector x
    op = FEOperator((u,v) -> res(u,v,φh,dΩ...),U,V,assem_U)
    op_cache = CachedFEOperator(op)
    x = zero(U)#get_free_dof_values(zero(U));

    ## Adjoint K,x
    adjoint_K = assemble_matrix((u,v) -> op.jac(uhd,v,u),assem_adjoint,V,U)
    adjoint_x = get_free_dof_values(zero(V))

    ## Numerical setups
    nls_cache = NLCache(nothing)
    adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
    
    ## Caches and adjoint caches
    cache = (dφdu_vec,assem_deriv)
    fwd_caches = (nls,nls_cache,x,op_cache,assem_U)
    adjoint_caches = (adjoint_ns,adjoint_K,adjoint_x,assem_adjoint)
    return new{A,typeof(dΩ),typeof(spaces),typeof(cache),typeof(fwd_caches),
      typeof(adjoint_caches)}(res,dΩ,spaces,cache,fwd_caches,adjoint_caches)
  end
end

# Helpers and getters
mutable struct NLCache
  cache
end

mutable struct CachedFEOperator
  op
end

get_state(m::NonlinearFEStateMap) = m.fwd_caches[3]
get_measure(m::NonlinearFEStateMap) = m.dΩ;
get_trial_space(m::NonlinearFEStateMap) = m.spaces[1];
get_test_space(m::NonlinearFEStateMap) = m.spaces[2];
get_aux_space(m::NonlinearFEStateMap) = m.spaces[3];
get_deriv_space(m::NonlinearFEStateMap) = m.spaces[4];
get_state_assembler(m::NonlinearFEStateMap) = last(m.fwd_caches)
get_deriv_assembler(m::NonlinearFEStateMap) = last(m.cache)

function (φ_to_u::NonlinearFEStateMap)(φ::T) where T <: AbstractVector
  res=φ_to_u.res
  dΩ = φ_to_u.dΩ
  U,V,V_φ,U_reg = φ_to_u.spaces
  nls,nls_cache,x,op,assem_U = φ_to_u.fwd_caches
  nl_cache = nls_cache.cache
 
  ## Update residual and jacobian, and solve
  φh = FEFunction(V_φ,φ)
  op.op = FEOperator((u,v) -> res(u,v,φh,dΩ...),U,V,assem_U) 
  x,cache = solve!(x,nls,op.op,nl_cache)
  # Update cache for next call
  nls_cache.cache = cache 
  get_free_dof_values(x)
end

function ChainRulesCore.rrule(φ_to_u::NonlinearFEStateMap,φ::T) where T <: AbstractVector
  @assert T<:Vector "Nonlinear FEs are currently not supported in parallel, please run in serial mode."
  res = φ_to_u.res
  dΩ = φ_to_u.dΩ
  U,V,V_φ,U_reg = φ_to_u.spaces
  dφdu_vec,assem_deriv = φ_to_u.cache
  _,_,_,op,_ = φ_to_u.fwd_caches
  adjoint_ns,adjoint_K,λ,assem_adjoint = φ_to_u.adjoint_caches
  
  ## Forward problem
  u = φ_to_u(φ)

  ## Adjoint operator
  uh = FEFunction(U,u)
  φh = FEFunction(V_φ,φ)
  assemble_matrix!((u,v) -> op.op.jac(uh,v,u),adjoint_K,assem_adjoint,V,U)
  function φ_to_u_pullback(du)
    ## Adjoint Solve
    #### NOTE: This breaks in parallel, need assembler to output matrix adjoint.
    numerical_setup!(adjoint_ns,adjoint(adjoint_K))
    ## DEBUG only
    K = jacobian(op.op,uh)
    println("**Debug** |adjoint_K - K| = ", norm(adjoint_K - K,Inf)) # This should be > 0
    println("**Debug** |adjoint(adjoint_K) - K| = ", norm(adjoint(adjoint_K) - K,Inf)) # This should be zero
    ##
    solve!(λ,adjoint_ns,du)
    λh = FEFunction(V,λ)
    
    ## Compute grad
    uh = FEFunction(U,u)
    res_functional = IntegrandWithMeasure(res,dΩ)
    dφdu_contrib = ∇(res_functional,[uh,λh,φh],3)
    dφdu_vecdata = collect_cell_vector(U_reg,dφdu_contrib) 
    assemble_vector!(dφdu_vec,assem_deriv,dφdu_vecdata)
    dφdu_vec .*= -1;
    ( NoTangent(),dφdu_vec)
  end
  u, φ_to_u_pullback
end

"""
  PDEConstrainedFunctionals
"""
struct PDEConstrainedFunctionals{N,A<:AbstractFEStateMap}
  J
  C
  dJ
  dC
  analytic_dJ
  analytic_dC
  state_map :: A

  function PDEConstrainedFunctionals(
      J :: Function,
      C :: Vector{<:Function},
      state_map :: T;
      analytic_dJ = nothing,
      analytic_dC = fill(nothing,length(C))) where T<:AbstractFEStateMap

    dΩ = get_measure(state_map)
    U = get_trial_space(state_map)
    V_φ = get_aux_space(state_map)
    U_reg = get_deriv_space(state_map)
    assem_U = get_state_assembler(state_map)
    assem_deriv = get_deriv_assembler(state_map)

    # Create StateParamIntegrandWithMeasures
    spiwm(f) = StateParamIntegrandWithMeasure(
      IntegrandWithMeasure(f,dΩ),U,V_φ,U_reg,assem_U,assem_deriv)
    J_spiwm = spiwm(J)
    C_spiwm = isempty(C) ? StateParamIntegrandWithMeasure[] : map(spiwm,C);

    # Preallocate
    _,djdφ_vec = J_spiwm.caches
    dJ = similar(djdφ_vec)
    dC = map(_->similar(dJ),C)

    return new{length(C),T}(
      J_spiwm,C_spiwm,dJ,dC,analytic_dJ,analytic_dC,state_map)
  end
end

PDEConstrainedFunctionals(J::Function,state_map::AbstractFEStateMap;analytic_dJ=nothing) = 
  PDEConstrainedFunctionals(J,Function[],state_map;analytic_dJ = analytic_dJ,analytic_dC = Nothing[])

get_state(m::PDEConstrainedFunctionals) = get_state(m.state_map)

function evaluate_functionals!(pcf::PDEConstrainedFunctionals,φ::T) where T <: AbstractVector
  u = pcf.state_map(φ)
  J = pcf.J; C = pcf.C
  [J(u,φ), map(C(u,φ),1:length(C))]
  return J(u,φ), map(C(u,φ),1:length(C))
end

function evaluate_derivatives!(pcf::PDEConstrainedFunctionals,φ::T) where T <: AbstractVector
  _,_,dJ,dC = _evaluate_derivatives(pcf,φ)
  return dJ,dC
end

function Gridap.evaluate!(pcf::PDEConstrainedFunctionals,φ::T) where T <: AbstractVector
  _evaluate_derivatives(pcf,φ)
end

function _evaluate_derivatives(pcf::PDEConstrainedFunctionals,φ::T) where T <: AbstractVector
  J = pcf.J; C = pcf.C
  dJ = pcf.dJ; dC = pcf.dC
  analytic_dJ = pcf.analytic_dJ; 
  analytic_dC = pcf.analytic_dC
  U = get_trial_space(pcf.state_map)
  V_φ = get_aux_space(pcf.state_map)
  U_reg = get_deriv_space(pcf.state_map)
  deriv_assem = get_deriv_assembler(pcf.state_map)
  dΩ = get_measure(pcf.state_map)

  ## Foward problem
  u, u_pullback = rrule(pcf.state_map,φ)

  function ∇!(F::StateParamIntegrandWithMeasure,dF,::Nothing)
    # Automatic differentation
    j_val, j_pullback = rrule(F,u,φ); # Compute functional and pull back
    _, dFdu, dFdφ     = j_pullback(1); # Compute dFdu, dFdφ
    _, dφ_adj         = u_pullback(dFdu); # Compute -dFdu*dudφ via adjoint 
    copy!(dF,dφ_adj)
    dF .+= dFdφ
    return j_val
  end
  function ∇!(F::StateParamIntegrandWithMeasure,dF,dF_analytic)
    # Analytic shape derivative
    j_val = F(u,φ)
    uh = FEFunction(U,u)
    φh = FEFunction(V_φ,φ)
    _dF = (q) -> dF_analytic(q,uh,φh,dΩ...)
    assemble_vector!(_dF,dF,deriv_assem,U_reg)
    dF .*= -1 # <- Take θ=-vn
    return j_val
  end
  j = ∇!(J,dJ,analytic_dJ)
  c = map(∇!,C,dC,analytic_dC)

  return j,c,dJ,dC
end

# Helpers
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
  print(io,"$(nameof(typeof(object)))")
  print(io,":")
  print(io,"\n num_constraints: $(length(object.C))")
end