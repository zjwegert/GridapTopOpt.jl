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
#
# TODO: I'm not super happy about creating several PVectors in the below, but
#       it is the easiest way to get a nice structure in `grad` that conforms to
#       the serial version. Perhaps adjust in future.
#
#       Note that this is almost the same as `Zygote.withjacobian` except we return
#       Tuples of Tuples of PVectors instead of Tuples of Matrices. The latter
#       isn't compatible unless we use BlockPMatrix? - maybe this is better in future...
function _withjacobian(f, x::PVector, args...)
  @check eltype(args) <: PVector "Additional args must be PVectors"
  y, back = Zygote.pullback(Zygote._jvec∘f, x, args...)
  out = map((x,args...)) do x
    T = promote_type(eltype(x), eltype(y))
    dx = Tuple([similar(x,T) for _ in eachindex(y)])
  end
  delta = Zygote._eyelike(y)
  for k in LinearIndices(y)
    grads = back(delta[:,k])
    for (dx, grad) in zip(out, grads)
      copyto!(dx[k], grad)
    end
  end
  (val=y, grad=out)
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