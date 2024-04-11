
"""
  Represents a functional of the form

    J(u₁,...,uₙ,dΩ₁,...,dΩₘ) = sum(F(u₁,...,uₙ,dΩ₁,...,dΩₘ))

  where `F` is an `IntegrandOperator`.
"""
struct IntegralQuantity <: IntegrandOperator
  F :: IntegrandOperator
end

function evaluate_cache(J::IntegralQuantity,uh)
  return evaluate_cache(J.F,uh)
end

function evaluate!(cache,J::IntegralQuantity,uh)
  return sum(evaluate!(cache,J.F,uh))
end

function gradient_cache(J::IntegralQuantity,uh,K)
  return gradient_cache(J.F,uh,K)
end

function gradient!(cache,J::IntegralQuantity,uh,K;updated=false)
  return gradient!(cache,J.F,uh,K;updated)
end

function gradient_and_evaluate_cache(J::IntegralQuantity,uh,k)
  return gradient_and_evaluate_cache(J.F,uh,k)
end

function gradient_and_evaluate!(cache,J::IntegralQuantity,uh,k;updated=false)
  j, dj = gradient_and_evaluate!(cache,J.F,uh,k;updated)
  return sum(j), dj
end

struct OperationIntegralQuantity{T} <: IntegrandOperator
  op :: Operation{T}
  quants :: Tuple
end

Arrays.evaluate!(op::Operation,quants...::IntegralQuantity...) = OperationIntegralQuantity(op,quants)

function Arrays.evaluate!(cache,J::OperationIntegralQuantity,uh)
  return J.op.op(map(Jk -> evaluate!(cache,Jk,uh),J.quants)...)
end

function gradient_cache(J::OperationIntegralQuantity,uh,K)
  return gradient_cache(J.op,uh,K)
end

function gradient!(cache,J::OperationIntegralQuantity,uh,K;updated=false)
  return gradient!(cache,J.op,uh,K;updated)
end

function gradient_and_evaluate_cache(J::OperationIntegralQuantity,uh,k)
  return gradient_and_evaluate_cache(J.op,uh,k)
end

function gradient_and_evaluate!(cache,J::OperationIntegralQuantity,uh,k;updated=false)
  return gradient_and_evaluate!(cache,J.op,uh,k;updated)
end
