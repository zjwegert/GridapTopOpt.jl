
"""
    struct AlgebraicIntegrandOperator

  Algebraic view of an `IntegrandOperator` that allows working with plain 
  arrays instead of `FEFunction`s and `DomainContribution`s.
"""
struct AlgebraicIntegrandOperator{A,B,C,D}
  F      :: A
  spaces :: B
  assems :: C
  caches :: D
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

function Gridap.gradient(AF::AlgebraicIntegrandOperator,uh,K)
  @check 0 < K <= length(AF.spaces)
  Vk, ak, xk = AF.spaces[K], AF.assems[K], AF.caches[K]
  assemble_vector!(xk,ak,collect_cell_vector(Vk,gradient(AF.F,uh,K)))
  return xk
end

function Gridap.jacobian(AF::AlgebraicIntegrandOperator,uh,K)
  @check 0 < K <= length(AF.spaces)
  Vk, ak, xk = AF.spaces[K], AF.assems[K], AF.caches[K]
  assemble_vector!(xk,ak,collect_cell_vector(Vk,jacobian(AF.F,uh,K)))
  return xk
end