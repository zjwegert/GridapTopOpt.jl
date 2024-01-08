
"""
  OrthogonalisationMethod

  Method must return C_orthog, normsq, nullity.
"""
abstract type OrthogonalisationMap <: Gridap.Fields.Map end
Fields.evaluate!(cache,::OrthogonalisationMap,::AbstractArray{<:AbstractArray},::AbstractMatrix) = @abstractmethod
Fields.return_cache(::OrthogonalisationMap,A::AbstractArray{<:AbstractArray},K::AbstractMatrix) = @abstractmethod

"""
  HPModifiedGramSchmidt

  High performance modified Gram-Schmidt. Based on https://doi.org/10.1007/s13160-019-00356-4.
"""
struct HPModifiedGramSchmidt <: OrthogonalisationMap end

function Fields.return_cache(::HPModifiedGramSchmidt,A::AbstractArray{<:AbstractArray},K::AbstractMatrix)
  n = length(A)
  Z = [allocate_in_domain(K) for _ = 1:n]
  Q = [allocate_in_domain(K) for _ = 1:n]
  R = zeros(n)
  P = allocate_in_domain(K)
  X = [allocate_in_domain(K) for _ = 1:n]
  return n,Z,Q,R,P,X
end

function Fields.evaluate!(cache,::HPModifiedGramSchmidt,A::AbstractArray{<:AbstractArray},K::AbstractMatrix)
  n,Z,Q,R,P,X = cache

  map(copyto!,Z,A)
  map((x,z) -> mul!(x,K,Z),X,Z)

  nullity = 0
  for i = 1:n
    R[i] = dot(Z[i],X[i])
    if R[i] < 1.e-14
      R[i] = zero(eltype(R))
      Ri = one(eltype(R))
      nullity += 1
    else
      Ri = sqrt(R[i])
    end
    Q[i] .= Z[i] ./ Ri
    P .= X[i] ./ Ri

    Rij_sum = zero(eltype(R))
    for j = i+1:n
      Rij = dot(P,Z[j])
      Z[j] .-= Rij .* Q[i]
      Rij_sum += Rij
    end
    X[i] .-= Rij_sum .* P
  end

  return Z,R,nullity
end

# Verbose printing
function verbose_print(orthog,dV,dC,dC_orthog,K,P,λ,nullity,debug_code,verbose)
  if ~isnothing(verbose)
    orth_norm = zeros(length(dC),length(dC))
    for i = 1:length(dC)
      mul!(P,K,dC_orthog[i])
      for j = 1:length(dC)
        orth_norm[i,j]=dot(dC_orthog[j],P);
      end
    end
    mul!(P,K,dV)
    proj_norm = dot.(dC_orthog,(P,))
    if i_am_main(verbose)
      orth_norm[diagind(orth_norm)] .= 0; 
      println("  ↱----------------------------------------------------------↰")
      print("                   ")
      printstyled("Hilbertian Projection Method\n\n",color=:yellow,underline=true);
      println("      -->         Orthog. method: $(typeof(orthog))");
      @printf("      --> Orthogonality inf-norm: %e\n",norm(orth_norm,Inf))
          @printf("      -->    Projection inf-norm: %e\n",(norm(proj_norm,Inf)))
      println("      -->          Basis nullity: ",nullity)
      if iszero(debug_code)
        println("      --> Constraints satisfied: ∑α² ≈ 0.")
      elseif isone(debug_code)
        println("      -->            ∑α² > α_max: scaling λ")
      elseif isequal(debug_code,2)
        println("      -->            ∑α² < α_max: scaling λ")
      end
      print("  ↳----------------------------------------------------------↲\n")
    end
  end
end
