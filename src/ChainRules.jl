"""
  IntegrandWithMeasure

  Enables partial differentation of an integrand F via Gridap.gradient.
"""
struct IntegrandWithMeasure{A,B<:Tuple}
  F  :: A
  dΩ :: B
end

(F::IntegrandWithMeasure)(args...) = F.F(args...,F.dΩ...)

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
  assems  :: C
  caches  :: D
end

function StateParamIntegrandWithMeasure(
  F::IntegrandWithMeasure,
  U::FESpace,V_φ::FESpace,U_reg::FESpace,
  assem_U::Assembler,assem_deriv::Assembler
)
  φ₀ = zero(V_φ); u₀ = zero(U)
  djdu_vecdata = collect_cell_vector(U,∇(F,[u₀,φ₀],1))
  djdφ_vecdata = collect_cell_vector(U_reg,∇(F,[u₀,φ₀],2))
  djdu_vec = allocate_vector(assem_U,djdu_vecdata)
  djdφ_vec = allocate_vector(assem_deriv,djdφ_vecdata)
  assems = (assem_U,assem_deriv)
  spaces = (U,V_φ,U_reg)
  caches = (djdu_vec,djdφ_vec)
  return StateParamIntegrandWithMeasure(F,spaces,assems,caches)
end

function (u_to_j::StateParamIntegrandWithMeasure)(u::T1,φ::T2) where {T1<:AbstractArray,T2<:AbstractArray}
  U,V_φ,_ = u_to_j.spaces
  uh = FEFunction(U,u)
  φh = FEFunction(V_φ,φ)
  return sum(u_to_j.F(uh,φh))
end

function ChainRulesCore.rrule(u_to_j::StateParamIntegrandWithMeasure,u::T1,φ::T2) where {T1<:AbstractArray,T2<:AbstractArray}
  F = u_to_j.F
  U,V_φ,U_reg = u_to_j.spaces
  assem_U,assem_deriv = u_to_j.assems
  djdu_vec,djdφ_vec = u_to_j.caches
  uh = FEFunction(U,u)
  φh = FEFunction(V_φ,φ)
  fields = [uh,φh]
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
  return u_to_j(u,φ), u_to_j_pullback
end

"""
  AbstractFEStateMap
"""
abstract type AbstractFEStateMap end

get_state(::AbstractFEStateMap) = @abstractmethod
get_measure(::AbstractFEStateMap) = @abstractmethod
get_spaces(::AbstractFEStateMap) = @abstractmethod
get_assemblers(::AbstractFEStateMap) = @abstractmethod

get_trial_space(m::AbstractFEStateMap) = get_spaces(m)[1]
get_test_space(m::AbstractFEStateMap) = get_spaces(m)[2]
get_aux_space(m::AbstractFEStateMap) = get_spaces(m)[3]
get_deriv_space(m::AbstractFEStateMap) = get_spaces(m)[4]

get_pde_assembler(m::AbstractFEStateMap) = get_assemblers(m)[1]
get_deriv_assembler(m::AbstractFEStateMap) = get_assemblers(m)[2]

@inline (φ_to_u::AbstractFEStateMap)(φ::AbstractVector) = forward_solve(φ_to_u,φ)

function forward_solve(φ_to_u::AbstractFEStateMap,φh)
  @abstractmethod
end

function forward_solve(φ_to_u::AbstractFEStateMap,φ::AbstractVector)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return forward_solve(φ_to_u,φh)
end

function update_adjoint_caches!(φ_to_u::AbstractFEStateMap,uh,φh)
  @abstractmethod
end

function update_adjoint_caches!(φ_to_u::AbstractFEStateMap,u::AbstractVector,φ::AbstractVector)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return update_adjoint_caches!(φ_to_u,uh,φh)
end

function adjoint_solve!(φ_to_u::AbstractFEStateMap,du::AbstractVector)
  @abstractmethod
end

function res_grad(φ_to_u::AbstractFEStateMap,uh,vh,φh)
  @abstractmethod
end

function res_grad(φ_to_u::AbstractFEStateMap,u::AbstractVector,v::AbstractVector,φ::AbstractVector)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  vh = FEFunction(get_test_space(φ_to_u),v)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return res_grad(φ_to_u,uh,vh,φh)
end

function pullback(φ_to_u::AbstractFEStateMap,uh,φh,du;updated=false)
  dφdu_vec, assem_deriv = φ_to_u.plb_caches
  U_reg = get_deriv_space(φ_to_u)

  ## Adjoint Solve
  if !updated
    update_adjoint_caches!(φ_to_u,uh,φh)
  end
  λ  = adjoint_solve!(φ_to_u,du)
  λh = FEFunction(get_test_space(φ_to_u),λ)

  ## Compute grad
  dφdu_vecdata = collect_cell_vector(U_reg,res_grad(φ_to_u,uh,λh,φh)) 
  assemble_vector!(dφdu_vec,assem_deriv,dφdu_vecdata)
  rmul!(dφdu_vec, -1)
  
  return (NoTangent(),dφdu_vec)
end

function ChainRulesCore.rrule(φ_to_u::AbstractFEStateMap,φh)
  u  = forward_solve(φ_to_u,φh)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  update_adjoint_caches!(φ_to_u,uh,φh)
  return u, du -> pullback(φ_to_u,uh,φh,du;updated=true)
end

function ChainRulesCore.rrule(φ_to_u::AbstractFEStateMap,φ::AbstractVector)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return ChainRulesCore.rrule(φ_to_u,φh)
end

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
  AffineFEStateMap
"""
struct AffineFEStateMap{A,B,C,D,E,F} <: AbstractFEStateMap
  biform     :: A
  liform     :: B
  spaces     :: C
  plb_caches :: D
  fwd_caches :: E
  adj_caches :: F

  function AffineFEStateMap(
    a::Function,l::Function,
    U,V,V_φ,U_reg,φh,dΩ...;
    assem_U = SparseMatrixAssembler(U,V),
    assem_adjoint = SparseMatrixAssembler(V,U),
    assem_deriv = SparseMatrixAssembler(U_reg,U_reg),
    ls::LinearSolver = LUSolver(),
    adjoint_ls::LinearSolver = LUSolver()
  )

    biform = IntegrandWithMeasure(a,dΩ)
    liform = IntegrandWithMeasure(l,dΩ)
    spaces = (U,V,V_φ,U_reg)

    ## Pullback cache
    uhd = zero(U)
    vecdata = collect_cell_vector(U_reg,∇(biform,[uhd,uhd,φh],3) - ∇(liform,[uhd,φh],2))
    dφdu_vec = allocate_vector(assem_deriv,vecdata)
    plb_caches = (dφdu_vec,assem_deriv)

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

function forward_solve(φ_to_u::AffineFEStateMap,φh)
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

function res_grad(φ_to_u::AffineFEStateMap,uh,vh,φh)
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
  NonlinearFEStateMap
"""
struct NonlinearFEStateMap{A,B,C,D,E,F} <: LSTO_Distributed.AbstractFEStateMap
  res        :: A
  jac        :: B
  spaces     :: C
  plb_caches :: D
  fwd_caches :: E
  adj_caches :: F

  function NonlinearFEStateMap(
    res::Function,U,V,V_φ,U_reg,φh,dΩ...;
    assem_U = SparseMatrixAssembler(U,V),
    assem_adjoint = SparseMatrixAssembler(V,U),
    assem_deriv = SparseMatrixAssembler(U_reg,U_reg),
    nls::NonlinearSolver = NewtonSolver(LUSolver();maxiter=50,rtol=1.e-10,verbose=true),
    adjoint_ls::LinearSolver = LUSolver()
  )
    res = IntegrandWithMeasure(res,dΩ)
    jac = (u,du,dv,φh) -> jacobian(res,[u,dv,φh],1)
    spaces = (U,V,V_φ,U_reg)

    ## Pullback cache
    uhd = zero(U)
    vecdata = collect_cell_vector(U_reg,∇(res,[uhd,uhd,φh],3))
    dφdu_vec = allocate_vector(assem_deriv,vecdata)
    plb_caches = (dφdu_vec,assem_deriv)

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

function forward_solve(φ_to_u::NonlinearFEStateMap,φh)
  U, V, _, _ = φ_to_u.spaces
  nls, nls_cache, x, assem_U = φ_to_u.fwd_caches

  res(u,v) = φ_to_u.res(u,v,φh)
  jac(u,du,dv) = φ_to_u.jac(u,du,dv,φh)
  op = get_algebraic_operator(FEOperator(res,jac,U,V,assem_U))
  solve!(x,nls,op,nls_cache)
  return x
end

function res_grad(φ_to_u::NonlinearFEStateMap,uh,vh,φh)
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

# Instantiate nonlinear solver caches

function instantiate_caches(x,nls::NLSolver,op::NonlinearOperator)
  Gridap.Algebra._new_nlsolve_cache(x,nls,op)
end

function instantiate_caches(x,nls::NewtonRaphsonSolver,op::NonlinearOperator)
  b = residual(op, x)
  A = jacobian(op, x)
  dx = similar(b)
  ss = symbolic_setup(nls.ls, A)
  ns = numerical_setup(ss,A)
  return Gridap.Algebra.NewtonRaphsonCache(A,b,dx,ns)
end

function instantiate_caches(x,nls::NewtonSolver,op::NonlinearOperator)
  b  = residual(op, x)
  A  = jacobian(op, x)
  dx = allocate_in_domain(A); fill!(dx,zero(eltype(dx)))
  ss = symbolic_setup(nls.ls, A)
  ns = numerical_setup(ss,A,x)
  return GridapSolvers.NonlinearSolvers.NewtonCache(A,b,dx,ns)
end

# Transpose contributions

transpose_contributions(b::DistributedDomainContribution) = 
  DistributedDomainContribution(map(transpose_contributions,local_views(b)))

function transpose_contributions(b::DomainContribution)
  c = DomainContribution()
  for (trian,array_old) in b.dict
    array_new = lazy_map(transpose,array_old)
    add_contribution!(c,trian,array_new)
  end
  return c
end

function assemble_adjoint_matrix(f::Function,a::Assembler,U::FESpace,V::FESpace)
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  contr = transpose_contributions(f(u,v))
  assemble_matrix(a,collect_cell_matrix(V,U,contr))
end

function assemble_adjoint_matrix!(f::Function,A::AbstractMatrix,a::Assembler,U::FESpace,V::FESpace)
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  contr = transpose_contributions(f(u,v))
  assemble_matrix!(A,a,collect_cell_matrix(V,U,contr))
end

function Gridap.FESpaces.assemble_matrix_and_vector!(
    a::Function,l::Function,A::AbstractMatrix,b::AbstractVector,assem::Assembler,U::FESpace,V::FESpace,uhd)
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  assemble_matrix_and_vector!(A,b,assem,collect_cell_matrix_and_vector(U,V,a(u,v),l(v),uhd))
end

"""
  PDEConstrainedFunctionals
"""
struct PDEConstrainedFunctionals{N,A}
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

    # Create StateParamIntegrandWithMeasures
    J_spiwm = StateParamIntegrandWithMeasure(J,state_map)
    if isempty(C)
      C_spiwm = StateParamIntegrandWithMeasure[]
    else
      C_spiwm = map(ci -> StateParamIntegrandWithMeasure(ci,state_map),C)
    end

    # Preallocate
    dJ = similar(J_spiwm.caches[2])
    dC = map(Ci_spiwm->similar(Ci_spiwm.caches[2]),C_spiwm)

    return new{length(C),T}(J_spiwm,C_spiwm,dJ,dC,analytic_dJ,analytic_dC,state_map)
  end
end

PDEConstrainedFunctionals(J::Function,state_map::AbstractFEStateMap;analytic_dJ=nothing) = 
  PDEConstrainedFunctionals(J,Function[],state_map;analytic_dJ = analytic_dJ,analytic_dC = Nothing[])

get_state(m::PDEConstrainedFunctionals) = get_state(m.state_map)

function evaluate_functionals!(pcf::PDEConstrainedFunctionals,φ::T) where T <: AbstractVector
  u = pcf.state_map(φ)
  J = pcf.J; C = pcf.C
  return J(u,φ), map(Ci->Ci(u,φ),C)
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
