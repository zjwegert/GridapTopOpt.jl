"""
    struct AlgebraicIntegrandOperator

  Algebraic view of an `IntegrandOperator` that allows working with plain 
  arrays instead of `FEFunction`s and `DomainContribution`s.
"""
struct AlgebraicIntegrandOperator{A,B,C,D} <: IntegrandOperator
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

function evaluate_cache(AF::AlgebraicIntegrandOperator,uh)
  return evaluate_cache(AF.F,uh)
end

function Arrays.evaluate!(cache,AF::AlgebraicIntegrandOperator,uh;updated=false)
  return evaluate!(cache,AF.F,uh;updated)
end

function gradient_cache(AF::AlgebraicIntegrandOperator,uh,K)
  @check 0 < K <= length(AF.spaces)
  Vk, ak = AF.spaces[K], AF.assems[K]

  dFduk_cache = gradient_cache(AF.F,uh,K)
  dFduk = gradient!(dFduk_cache,AF.F,uh,K)
  xk = allocate_vector(ak,collect_cell_vector(Vk,dFduk))
  return xk, dFduk_cache
end

function gradient!(cache,AF::AlgebraicIntegrandOperator,uh,K;updated=false)
  @check 0 < K <= length(AF.spaces)
  xk, dFduk_cache = cache

  dFduk = gradient!(dFduk_cache,AF.F,uh,K;updated)
  assemble_vector!(xk,ak,collect_cell_vector(Vk,dFduk))
  return xk
end
