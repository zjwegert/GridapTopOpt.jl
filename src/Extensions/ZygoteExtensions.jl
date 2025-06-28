### PartitionedArrays extensions

import Base: size, broadcasted
using PartitionedArrays: PBroadcasted
size(a::PBroadcasted)=(length(a.own_values)+length(a.ghost_values),)
size(a::PBroadcasted{A,Nothing,C}) where {A,C}= (length(a.own_values),)

Base.:*(a::AbstractThunk,b::PVector) = unthunk(a)*b
Base.:*(b::PVector,a::AbstractThunk) = b*unthunk(a)
Base.:/(b::PVector,a::AbstractThunk) = b/unthunk(a)
LinearAlgebra.rmul!(a::PVector,v::AbstractThunk) = rmul!(a,unthunk(v))
Base.broadcasted(f, a::AbstractThunk, b::Union{PVector,PBroadcasted}) = broadcasted(f,unthunk(a),b)
Base.broadcasted(f, a::Union{PVector,PBroadcasted}, b::AbstractThunk) = broadcasted(f,a::Union{PVector,PBroadcasted},unthunk(b))

### Zygote extensions to enable compat with PartitionedArrays

# This method is compatible with PVector inputs. This could be Zygote.withjacobian,
# but the output isn't compatible with the default Zygote API (eltype(grad)<:Matrix),
# so we define a new function instead.
#
# Notes:
# - TODO: I'm not super happy about creating several PVectors in the below, but
#   it is the easiest way to get a nice structure in `grad` that conforms to
#   the serial version. Perhaps adjust in future.
#     Note that this is almost the same as `Zygote.withjacobian` except we return
#   Tuples of Tuples of PVectors instead of Tuples of Matrices. The latter
#   isn't compatible unless we use BlockPMatrix? - maybe this is better in future...
#
# - We specify `ignore_pullback` to allow us to avoid computing gradients for some terms.
#   This is useful for when we hae an analytic derivative and don't want to compute the
#   adjoint because this requires a solution to an FE problem.
function val_and_jacobian(f, args...;ignore_pullback=[])
  y, back = Zygote.pullback(Zygote._jvec∘f, args...)
  out = map(args) do x
    T = promote_type(eltype(x), eltype(y))
    dx = [similar(x,T) for _ in eachindex(y)]
  end
  delta = Zygote._eyelike(y)
  for k in LinearIndices(y)
    if k ∈ ignore_pullback
      map(dx->fill!(dx[k],0.0),out)
      continue
    end
    grads = back(delta[:,k])
    for (dx, grad) in zip(out, grads)
      copyto!(dx[k], grad)
    end
  end
  (val=y, grad=out)
end


# Equivalent `Zygote.withgradient` call that is compatible with PartitionedArrays
function val_and_gradient(f, args...)
  y, back = Zygote.pullback(f, args...)
  grad = back(Zygote.sensitivity(y))
  (val=y, grad)
end