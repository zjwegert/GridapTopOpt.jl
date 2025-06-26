### Gridap/Zygote adjoint
Zygote.@adjoint FEFunction(U,u) = FEFunction(U,u), y->(nothing,y)

### PartitionedArrays extensions

import Base: size
size(a::PartitionedArrays.PBroadcasted)=(length(a.own_values)+length(a.ghost_values),)
size(a::PartitionedArrays.PBroadcasted{A,Nothing,C}) where {A,C}= (length(a.own_values),)

### Zygote extensions to enable compat with PartitionedArrays

function Zygote.withgradient(f, x::PVector, args...)
  @check eltype(args) <: PVector "Additional args must be PVectors"
  y, back = Zygote.pullback(f, x, args...)
  grad = back(Zygote.sensitivity(y))
  (val=y, grad)
end

# This could be Zygote.withjacobian, but the output isn't compatible with the
# default Zygote API (eltype(grad)<:Matrix), so we define a new function instead.
function _withjacobian(f, x::PVector, args...)
  @check eltype(args) <: PVector "Additional args must be PVectors"
  y, back = Zygote.pullback(Zygote._jvec∘f, x, args...)
  delta = Zygote._eyelike(y)
  grad = map(Tuple(LinearIndices(y))) do k
    back(delta[:,k])
  end
  (val=y, grad)
end

## Some new API to unify output of grads from jacobian

function tuplify(x::NTuple{N,<:Matrix}) where N
  y = map(x) do x
    Tuple(collect(row) for row in eachrow(x))
  end
end

function val_and_jacobian(f, x::AbstractVector, args...)
  val, grad = Zygote.withjacobian(f, x, args...)
  return (;val,grad=tuplify(grad))
end

function val_and_jacobian(f, x::PVector, args...)
  _withjacobian(f, x, args...)
end

val_and_gradient(f, args...) = Zygote.withgradient(f, args...)