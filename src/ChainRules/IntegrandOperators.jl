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
    struct FEIntegrandOperator{A,B<:Tuple}

  Represents a functional of the form

    F(u₁,...,uₙ,dΩ₁,...,dΩₘ) = ∑ ∫fᵢ(u₁,u₂,...,uₙ)dΩᵢ

  where 
   - `u₁,u₂,...,uₙ` are `FEFunctions`, and
   - `dΩ₁,dΩ₂,...,dΩₘ` are `Measures`.
"""
struct FEIntegrandOperator{A,B<:Tuple}
  F  :: A
  dΩ :: B

  function FEIntegrandOperator(F::Function,dΩ::Tuple)
    A, B = typeof(F), typeof(dΩ)
    return new{A,B}(F,dΩ)
  end
end

(F::FEIntegrandOperator)(args...) = F.F(args...,F.dΩ...)

function Gridap.gradient!(cache,F::FEIntegrandOperator,uh::Vector{<:FEFunction},K::Int)
  @check 0 < K <= length(uh)
  _f(uk) = F.F(uh[1:K-1]...,uk,uh[K+1:end]...,F.dΩ...)
  return Gridap.gradient(_f,uh[K])
end

function Gridap.gradient!(cache,F::FEIntegrandOperator,uh::Vector,K::Int)
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
    struct AlgebraicIntegrandOperator

  Algebraic view of an `IntegrandOperator` that allows working with plain 
  arrays instead of `FEFunction`s and `DomainContribution`s.
"""
struct AlgebraicIntegrandOperator{A,B,C,D}
  F      :: A
  spaces :: B
  assems :: C
  function AlgebraicIntegrandOperator(
    F::IntegrandOperator,
    spaces::Vector{<:FESpace};
    assems = map(V -> SparseMatrixAssembler(V,V),spaces)
  )

    fields = map(zero,spaces)
    caches = map(LinearIndices(spaces),spaces,assems) do k, Vk, ak
      dFduk = gradient(F,fields,k)
      allocate_vector(ak,collect_cell_vector(Vk,dFduk))
    end

    A, B = typeof(F), typeof(spaces)
    C, D = typeof(assems), typeof(caches)
    return new{A,B,C,D}(F,spaces,assems,caches)
  end
end

function AlgebraicIntegrandOperator(
  F::Function,dΩ::Tuple,spaces::Vector{<:FESpace};
  assems = map(V -> SparseMatrixAssembler(V,V),spaces)
)
  op = FEIntegrandOperator(F,dΩ)
  return AlgebraicIntegrandOperator(op,spaces;assems=assems)
end

function gradient_cache(AF::AlgebraicIntegrandOperator,uh,k)
  @check 0 < K <= length(AF.spaces)
  Vk, ak = AF.spaces[k], AF.assems[k]

  dFduk_cache = gradient_cache(AF.F,uh,k)
  dFduk = gradient!(dFduk_cache,AF.F,uh,k)
  xk = allocate_vector(ak,collect_cell_vector(Vk,dFduk))
  return xk, dFduk_cache
end

function gradient!(cache,AF::AlgebraicIntegrandOperator,uh,K)
  @check 0 < K <= length(AF.spaces)
  xk, dFduk_cache = cache

  dFduk = gradient!(dFduk_cache,AF.F,uh,K)
  assemble_vector!(xk,ak,collect_cell_vector(Vk,dFduk))
  return xk
end

"""
    struct ParametricIntegrandOperator{A,B}

    G(φ) = F(u(φ),φ)
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

function Gridap.gradient!(cache,G::ParametricIntegrandOperator,φh,K)
  @check K == 1
  dFdu_vec, dFdu_cache, dFdφ_cache = cache
  U = get_trial_space(G.state_map)

  u, u_pullback = rrule(G.state_map,φh)
  uh = FEFunction(U,u)

  dFdu = gradient!(dFdu_cache,AF.F,[uh,φh],1)
  dFdφ = gradient!(dFdφ_cache,AF.F,[uh,φh],2)

  assemble_vector!(dFdu_vec,collect_cell_vector(U,dFdu))
  dF = dFdφ + u_pullback(dFdu_vec)
  return dF
end
