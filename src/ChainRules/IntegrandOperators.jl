abstract type IntegrandOperator end

function gradient_cache(F::IntegrandOperator,uh,K)
  return nothing
end

function gradient!(cache,F::IntegrandOperator,uh,K)
  @abstractmethod
end

Gridap.gradient(F::IntegrandOperator,uh) = Gridap.gradient(F,[uh],1)

function Gridap.gradient(F::IntegrandOperator,uh,K)
  cache = gradient_cache(F,uh,K)
  return gradient!(cache,F,uh,K)
end

"""
    struct GenericIntegrandOperator{A,B<:Tuple} <: IntegrandOperator

  Represents a functional of the form

    F(u₁,...,uₙ,dΩ₁,...,dΩₘ) = ∫f₁(u₁,...,uₙ)dΩ₁ + ... + ∫fₘ(u₁,...,uₙ)dΩₘ

  where 
   - `u₁,u₂,...,uₙ` are `FEFunctions/FEBasis`, and
   - `dΩ₁,dΩ₂,...,dΩₘ` are `Measures`.
"""
struct GenericIntegrandOperator{A,B<:Tuple}
  F  :: A
  dΩ :: B

  function GenericIntegrandOperator(F::Function,dΩ::Tuple)
    A, B = typeof(F), typeof(dΩ)
    return new{A,B}(F,dΩ)
  end
end

(F::GenericIntegrandOperator)(args...) = F.F(args...,F.dΩ...)

function Gridap.gradient!(cache,F::GenericIntegrandOperator,uh::Vector{<:FEFunction},K::Int)
  @check 0 < K <= length(uh)
  _f(uk) = F.F(uh[1:K-1]...,uk,uh[K+1:end]...,F.dΩ...)
  return Gridap.gradient(_f,uh[K])
end

function Gridap.gradient!(cache,F::GenericIntegrandOperator,uh::Vector,K::Int)
  @check 0 < K <= length(uh)
  local_fields = map(local_views,uh) |> to_parray_of_arrays
  local_measures = map(local_views,F.dΩ) |> to_parray_of_arrays
  contribs = map(local_measures,local_fields) do dΩ,lf
    _f = u -> F.F(lf[1:K-1]...,u,lf[K+1:end]...,dΩ...)
    return Gridap.Fields.gradient(_f,lf[K])
  end
  return DistributedDomainContribution(contribs)
end

"""
    struct ParametricIntegrandOperator{A,B} <: IntegrandOperator

  Represents a functional of the form

    G(φ) = F(u(φ),φ)

  where `F` is an `IntegrandOperator` and the map `u(φ)` is defined by a `StateMap`.
"""
struct ParametricIntegrandOperator{A,B} <: IntegrandOperator
  F :: A
  state_map :: B
  function ParametricIntegrandOperator(
    F::IntegrandOperator,
    state_map::StateMap
  )
    A, B = typeof(F), typeof(state_map)
    return new{A,B}(F,state_map)
  end
end

function gradient_cache(G::ParametricIntegrandOperator,φh,K)
  @check K == 1
  U = get_trial_space(G.state_map)
  uh = zero(U)

  dFdu_cache = gradient_cache(AF.F,[uh,φh],1)
  dFdφ_cache = gradient_cache(AF.F,[uh,φh],2)
  
  dFdu = gradient!(dFdu_cache,AF.F,[uh,φh],1)
  x = allocate_vector(get_pde_assembler(G.state_map),collect_cell_vector(U,dFdu))
  return x, dFdu_cache, dFdφ_cache
end

function gradient!(cache,G::ParametricIntegrandOperator,φh,K)
  @check K == 1
  dFdu_vec, dFdu_cache, dFdφ_cache = cache
  U = get_trial_space(G.state_map)

  u, u_pullback = rrule(G.state_map,φh)
  uh = FEFunction(U,u)

  dFdu = gradient!(dFdu_cache,AF.F,[uh,φh],1)
  dFdφ = gradient!(dFdφ_cache,AF.F,[uh,φh],2)

  assemble_vector!(dFdu_vec,collect_cell_vector(U,dFdu))
  dGdφ = dFdφ + u_pullback(dFdu_vec)
  return dGdφ
end
