abstract type IntegrandOperator end

(F::IntegrandOperator)(args...) = evaluate(F,args...)

function evaluate_cache(F::IntegrandOperator,uh)
  return nothing
end

function Arrays.evaluate!(cache,F::IntegrandOperator,uh;updated=false)
  @abstractmethod
end

function Arrays.evaluate(F::IntegrandOperator,uh)
  cache = evaluate_cache(F,uh)
  return evaluate!(cache,F,uh)
end

function gradient_cache(F::IntegrandOperator,uh,K)
  return nothing
end

function gradient!(cache,F::IntegrandOperator,uh,K;updated=false)
  @abstractmethod
end

function Gridap.gradient(F::IntegrandOperator,uh,K)
  cache = gradient_cache(F,uh,K)
  return gradient!(cache,F,uh,K)
end

function evaluate_and_gradient_cache(F::IntegrandOperator,uh,K)
  eval_cache = evaluate_cache(F,uh)
  grad_cache = gradient_cache(F,uh,K)
  return eval_cache, grad_cache
end

function evaluate_and_gradient!(cache,F::IntegrandOperator,uh,K;updated=false)
  eval_cache, grad_cache = cache
  return evaluate!(eval_cache,F,uh;updated), gradient!(grad_cache,F,uh,K;updated)
end

function evaluate_and_gradient(F::IntegrandOperator,uh,K)
  cache = evaluate_and_gradient_cache(F,uh,K)
  return evaluate_and_gradient!(cache,F,uh,K)
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

function evaluate!(cache,F::GenericIntegrandOperator,uh)
  return F.F(uh...,F.dΩ...)
end

function gradient!(cache,F::GenericIntegrandOperator,uh::Vector{<:FEFunction},K::Int)
  @check 0 < K <= length(uh)
  _f(uk) = F.F(uh[1:K-1]...,uk,uh[K+1:end]...,F.dΩ...)
  return Gridap.gradient(_f,uh[K])
end

function gradient!(cache,F::GenericIntegrandOperator,uh::Vector,K::Int)
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

function evaluate_cache(G::ParametricIntegrandOperator,φh)
  uh = zero(get_trial_space(G.state_map))
  F_cache = evaluate_cache(G.F,[uh,φh])
  return uh, F_cache
end

function Arrays.evaluate!(cache,G::ParametricIntegrandOperator,φh;updated=false)
  uh, F_cache = cache
  if update
    uh = forward_solve!(G.state_map,uh,φh)
  end
  return evaluate!(F_cache,G.F,[uh,φh])
end

function gradient_cache(G::ParametricIntegrandOperator,φh,K)
  @check K == 1
  return gradient_cache(G.F,φh[1])
end

function gradient!(cache,G::ParametricIntegrandOperator,φh,K)
  @check K == 1
  return gradient!(cache,G.F,φh[1])
end

function gradient_cache(G::ParametricIntegrandOperator,φh)
  U = get_trial_space(G.state_map)
  uh = zero(U)

  dFdu_cache = gradient_cache(AF.F,[uh,φh],1)
  dFdφ_cache = gradient_cache(AF.F,[uh,φh],2)
  
  dFdu = gradient!(dFdu_cache,AF.F,[uh,φh],1)
  dFdu_vec = allocate_vector(get_pde_assembler(G.state_map),collect_cell_vector(U,dFdu))
  return uh, dFdu_vec, dFdu_cache, dFdφ_cache
end

function gradient!(cache,G::ParametricIntegrandOperator,φh;updated=false)
  uh, dFdu_vec, dFdu_cache, dFdφ_cache = cache
  U = get_trial_space(G.state_map)

  if !updated
    uh = forward_solve!(G.state_map,uh,φh)
    update_adjoint_caches!(G.state_map,uh,φh)
  end

  dFdu = gradient!(dFdu_cache,AF.F,[uh,φh],1)
  dFdφ = gradient!(dFdφ_cache,AF.F,[uh,φh],2)

  assemble_vector!(dFdu_vec,collect_cell_vector(U,dFdu))
  dGdφ = dFdφ + pullback(G.state_map,uh,φh,dFdu_vec;updated=true)
  return dGdφ
end
