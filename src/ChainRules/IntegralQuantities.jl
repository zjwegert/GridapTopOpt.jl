
"""
  Represents a functional of the form

    J(u₁,...,uₙ,dΩ₁,...,dΩₘ) = sum(F(u₁,...,uₙ,dΩ₁,...,dΩₘ))

  where `F` is an `IntegrandOperator`.
"""
struct IntegralQuantity <: IntegrandOperator
  F :: IntegrandOperator
end

(F::IntegralQuantity)(args...) = sum(F.F(args...))

function gradient_cache(F::IntegralQuantity,uh,K)
  return gradient_cache(F.F,uh,K)
end

function gradient!(cache,F::IntegralQuantity,uh,K)
  return gradient!(cache,F.F,uh,K)
end

struct OperationIntegralQuantity{T} <: IntegrandOperator
  op :: Operation{T}
  quants :: Tuple
end

(op::Operation)(quants...::IntegralQuantity...) = OperationIntegralQuantity(op,quants)

(J::OperationIntegralQuantity)(args...) = J.op(map(Jk -> Jk(args...),J.quants)...)

# We can now define the rrules depending on the operation type...
