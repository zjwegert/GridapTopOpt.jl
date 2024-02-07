
"""
  OrthogonalisationMap
"""
abstract type OrthogonalisationMap <: Gridap.Arrays.Map end

Gridap.Arrays.evaluate!(cache,::OrthogonalisationMap,::AbstractArray{<:AbstractArray},::AbstractMatrix) = @abstractmethod
Gridap.Arrays.return_cache(::OrthogonalisationMap,::AbstractArray{<:AbstractArray},::AbstractMatrix) = @abstractmethod

"""
  HPModifiedGramSchmidt

High performance modified Gram-Schmidt. Based on Algorithm 6 in this [paper](https://doi.org/10.1007/s13160-019-00356-4).
"""
struct HPModifiedGramSchmidt <: OrthogonalisationMap end

function Gridap.Arrays.return_cache(::HPModifiedGramSchmidt,A::AbstractArray{<:AbstractArray},K::AbstractMatrix)
  n = length(A)
  Z = [allocate_in_domain(K) for _ = 1:n]
  X = [allocate_in_domain(K) for _ = 1:n]
  Q = [allocate_in_domain(K) for _ = 1:n]
  R = zeros(n,n)
  return n,Z,X,Q,R
end

function Gridap.Arrays.evaluate!(cache,::HPModifiedGramSchmidt,A::AbstractArray{<:AbstractArray},K::AbstractMatrix)
  n,Z,X,Q,R = cache
  map(copy!,Z,A)
  map((x,z) -> mul!(x,K,z), X, Z)

  nullity = 0
  for i = 1:n
    R[i,i] = dot(Z[i],X[i])
    if R[i,i] < 1.e-14
      R[i,i] = zero(eltype(R))
      Ri = one(eltype(R))
      nullity += 1
    else
      R[i,i] = sqrt(R[i,i])
      Ri = R[i,i]
    end
    Q[i] .= Z[i] ./ Ri

    Rij_sum = zero(eltype(R))
    for j = i+1:n
      R[i,j] = dot(X[i],Z[j])/Ri
      Z[j] .-= R[i,j] .* Q[i]
      Rij_sum += R[i,j]
    end
    X[i] .-= (Rij_sum / Ri) .* X[i]
  end

  D = map(i -> R[i,i]^2, 1:n)
  return Z, D, nullity
  #return Z, X, Q, R, nullity
end
