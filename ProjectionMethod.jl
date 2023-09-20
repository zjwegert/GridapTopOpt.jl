Base.@kwdef struct HilbertianProjectionMethod{M<:AbstractFloat}
    λ::M = M(0.5)
    α_min::M = M(0.1)
    α_max::M = M(1.0)
end

ip(u::T1,K::T2,v::T3) where {T1<:AbstractArray,T2<:AbstractMatrix,T3<:AbstractArray} = dot(u,K*v)

function project(V::VT,Cᵥ::Vector{M},C::P,B::T,VSpace::VS,m::HilbertianProjectionMethod{M}) where 
        {M<:AbstractFloat,VT<:PVector{Vector{M}},P<:Vector{<:PVector{Vector{M}}},T<:PSparseMatrix,VS<:DistributedFESpace}
    # Get Data 
    λ = m.λ; α_min = m.α_min; α_max = m.α_max;
    # Orthogonalisation
    C_orthog,nullity,len_sq=gram_schmidt_orthog(C,B)
    C_len = sqrt.(len_sq);
    # Check norm and nullity
    k = length(C_orthog)
    orth_norm = zeros(k,k);
    for i = Base.OneTo(k), j = Base.OneTo(k)
        orth_norm[i,j] = ip(C_orthog[i],B,C_orthog[j]) 
    end
    orth_norm[diagind(orth_norm)].=0;
    # Project V
    V,norm_PCperp_V = project_V(V,C_orthog,len_sq,B)
    p_norm = ip.([V],[B],C_orthog)
    # Calculate αᵢ
    α = copy(Cᵥ);
    compute_α!(α,C_orthog,C,len_sq,C_len,B)
    ∑α² =λ^2*dot(α,α);
    # Printing and check ∑α² and update λ if required
    # if MPI.Comm_rank(comm) == root
    #     print("\n  ↱----------------------------------------------------------↰\n")
    #     printstyled("   Orthogonalisation method: gram_schmidt_orthog  ",color=:yellow);
    #     @printf("\n      --> Orthogonality inf-norm: %e\n",norm(orth_norm,Inf))
    #     @printf("      -->    Projection inf-norm: %e\n",(norm(p_norm,Inf)))
    #     println("      -->          Basis nullity: ",nullity)
    # end
    if ∑α² ≈ zero(M)
        # MPI.Comm_rank(comm) == root ? println("      --> Constraints satisfied: ∑α² ≈ 0.") : 0;
    elseif ∑α² > α_max
       λ *= sqrt(α_max/∑α²)
       ∑α² = α_max
    #    MPI.Comm_rank(comm) == root ? @printf("      --> ∑α² > α_max, scaling λ: %e\n",λ) : 0;
    elseif ∑α² < α_min
       λ *= sqrt(α_min/∑α²)
       ∑α² = α_min
    #    MPI.Comm_rank(comm) == root ? @printf("      --> ∑α² < α_max, scaling λ: %e\n",λ) : 0;
    end
    # MPI.Comm_rank(comm) == root ? print("  ↳----------------------------------------------------------↲\n") : 0;
    α .*= λ
    # Calculate direction
    idx_non_zero = (Base.OneTo(k))[.~iszero.(C_len)]
    θₙ = sqrt(1-∑α²)*V/norm_PCperp_V + sum(α[i] * C_orthog[i] / C_len[i] for i ∈ idx_non_zero)
    consistent!(θₙ) |> fetch
    return FEFunction(VSpace,θₙ)
end

# Compute α ignoring λ for now
function compute_α!(α::Vector{M},C_orthog::P,C::P,len_sq::Vector{M},C_len::Vector{M},B::T) where {
        M<:AbstractFloat,P<:Vector{<:PVector{Vector{M}}},T<:PSparseMatrix}
    k = length(C_orthog);
    for i ∈ Base.OneTo(k)
       if iszero(len_sq[i])
          α[i] = zero(M)
       else
          for j = 1:i-1
             if ~iszero(len_sq[j])
                # αᵢ|̄vᵢ| = λCᵢ - ∑ⱼ₌₁ⁱ⁻¹[ αⱼ(̄vⱼ⋅vᵢ)/|̄vⱼ| ]
                α[i] -= α[j]*ip(C_orthog[j],B,C[i])/C_len[j]
             end
          end
          α[i] /= C_len[i] # |̄vᵢ|
       end
    end
    return nothing
end

# Form projection operator for C⟂ of V: P_C⟂_V = V - ∑[(̄vᵢ ⋅ m)/|̄vᵢ|² ̄vᵢ] where ̄vᵢ spans C and ̄vᵢ ⋅ ̄vⱼ = δᵢⱼ 
# Return norm of projected V
function project_V(V::VT,C_orthog::P,len_sq::Vector{M},B::T) where {M<:AbstractFloat,VT<:PVector{Vector{M}},
        P<:Vector{<:PVector{Vector{M}}},T<:PSparseMatrix}
    k = length(C_orthog);
    for i ∈ Base.OneTo(k)
       if ~iszero(len_sq[i])
          V -= ip(C_orthog[i],B,V)/len_sq[i]*C_orthog[i] # [(̄vᵢ ⋅ m)/|̄vᵢ|² ̄vᵢ]
       end
    end
    # Check if projected V is zero vector
    consistent!(V) |> fetch
    if sqrt(ip(V,B,V)) ≈ 0
       return V,one(M)
    else
       return V,sqrt(ip(V,B,V))
    end
end

function gram_schmidt_orthog(C::P,B::T) where {M<:AbstractFloat,P<:Vector{<:PVector{Vector{M}}},T<:PSparseMatrix}
    k = length(C);
    nullity = 0;
    len_sq= zeros(M,k);
    tmp = zeros(M,k);
    C_orthog = zero.(C)
    for i ∈ Base.OneTo(k)
        C_orthog[i] .= C[i]
        consistent!(C_orthog[i]) |> fetch
    end
    # Loop
    len_sq[1] = ip(C_orthog[1],B,C_orthog[1]); # |̄v₁|²
    for i = 2:k
        len_sq[i] = ip(C_orthog[i],B,C_orthog[i]); # |̄vᵢ|²
        for j = 1:i-1
            if iszero(len_sq[j])
                #  V: The j_th vector in the sum is linearly dependent with previous vectors so ignore it
                continue
            end
            tmp[j] = ip(C_orthog[j],B,C_orthog[i]); # vᵢ ⋅ ̄vⱼ 
        end
        for j = 1:i-1
            if iszero(len_sq[j])
                #  V: The j_th vector in the sum is linearly dependent with previous vectors so ignore it
                continue
            end
            # ̄vᵢ = vᵢ - ∑ⱼ₌₁ⁱ⁻¹ [(vᵢ ⋅ ̄vⱼ)/|̄vⱼ|² ̄vⱼ]
            C_orthog[i] = C_orthog[i] - tmp[j]*C_orthog[j]/len_sq[j]
        end

        # Check for a linear dependency
        if len_sq[i] < 10^-14
            nullity += 1
            len_sq[i] = zero(M)
            C_orthog[i] = zero(C_orthog[i])
        elseif ip(C_orthog[i],B,C_orthog[i])/len_sq[i] < 10^-14
            nullity += 1
            len_sq[i] = zero(M)
            C_orthog[i] = zero(C_orthog[i])
        else
            len_sq[i] = ip(C_orthog[i],B,C_orthog[i])
        end
    end
    for i ∈ Base.OneTo(k)
        consistent!(C_orthog[i]) |> fetch
    end
    C_orthog,nullity,len_sq
end