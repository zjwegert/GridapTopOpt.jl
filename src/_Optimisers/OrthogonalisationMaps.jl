
"""
  OrthogonalisationMap
"""
abstract type OrthogonalisationMap <: Arrays.Map end

Fields.evaluate!(cache,::OrthogonalisationMap,::AbstractArray{<:AbstractArray},::AbstractMatrix) = @abstractmethod
Fields.return_cache(::OrthogonalisationMap,::AbstractArray{<:AbstractArray},::AbstractMatrix) = @abstractmethod

"""
  HPModifiedGramSchmidt

  High performance modified Gram-Schmidt. Based on Algorithm 6 in https://doi.org/10.1007/s13160-019-00356-4.
"""
struct HPModifiedGramSchmidt <: OrthogonalisationMap end

function Arrays.return_cache(::HPModifiedGramSchmidt,A::AbstractArray{<:AbstractArray},K::AbstractMatrix)
  n = length(A)
  Z = [allocate_in_domain(K) for _ = 1:n]
  X = [allocate_in_domain(K) for _ = 1:n]
  Q = [allocate_in_domain(K) for _ = 1:n]
  R = zeros(n,n)
  return n,Z,X,Q,R
end

function Arrays.evaluate!(cache,::HPModifiedGramSchmidt,A::AbstractArray{<:AbstractArray},K::AbstractMatrix)
  n,Z,X,Q,R = cache
  map(copy!,Z,A)

  nullity = 0
  for i = 1:n
    mul!(X[i],K,Z[i])
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

  return Z, view(R,diagind(R)), nullity
  #return Z, X, Q, R, nullity
end
