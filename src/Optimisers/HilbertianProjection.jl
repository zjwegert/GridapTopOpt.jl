# Orthogonalisation method
"""
  AbstractOrthogMethod

  Method must return C_orthog, normsq, nullity.
"""
abstract type AbstractOrthogMethod end
(::AbstractOrthogMethod)(::AbstractArray{<:AbstractArray},::AbstractMatrix,cache) = @abstractmethod
allocate_cache(::AbstractOrthogMethod,A::AbstractArray{<:AbstractArray}) = @abstractmethod

## HilbertianProjectionHistory
mutable struct HilbertianProjectionHistory
  it      :: Int
  const J :: Vector
  const C :: Matrix
  function HilbertianProjectionHistory(max_iters::Int,nconsts::Int)
    J = zeros(max_iters+1);
    C = zeros(max_iters+1,nconsts)
    new(-1,J,C)
  end
end

# Convienence 
Base.last(m::HilbertianProjectionHistory) = (m.it,m.J[m.it+1],m.C[m.it+1,:])

function write_history(m::HilbertianProjectionHistory,path;ranks=nothing)
  it = m.it; J = m.J; C = m.C;
  if i_am_main(ranks) 
    data = @views zip(J[1:it],eachslice(C[1:it,:],dims=2)...)
    writedlm(path,data)
  end
end

function update!(m::HilbertianProjectionHistory,J,C)
  m.it += 1
  m.J[m.it+1] = J 
  m.C[m.it+1,:] .= C
  return nothing
end

struct HilbertianProjection{T<:AbstractOrthogMethod} <: AbstractOptimiser
  λ                 :: Real
  α_min             :: Real
  α_max             :: Real
  α                 :: Vector
  max_iters         :: Int
  history           :: HilbertianProjectionHistory
  conv_criterion    :: Function
  orthog_method     :: T
  verbose
  orthog_cache
  line_search_cache
  γ_cache
  cache
  function HilbertianProjection(
      φ::AbstractVector,
      pcfs::PDEConstrainedFunctionals{N},
      stencil,vel_ext,interp,el_size,γ,γ_reinit;
      α_min = convert(eltype(φ),0.1),
      α_max = convert(eltype(φ),1.0),
      λ = convert(eltype(φ),0.5),
      orthog_method::T = HPModifiedGramSchmidt(),
      max_iters::Int = 1000,
      conv_criterion::Function = conv_cond,
      ls_max_iters::Int = 10,
      ls_δ_inc = convert(eltype(φ),1.1),
      ls_δ_dec = convert(eltype(φ),0.7),
      ls_ξ = convert(eltype(φ),0.005),
      ls_ξ_reduce_coef = convert(eltype(φ),0.1),
      ls_ξ_reduce_abs_tol = convert(eltype(φ),0.01),
      ls_γ_min = convert(eltype(φ),0.001),
      ls_γ_max = convert(eltype(φ),0.1),
      verbose=nothing) where {N,T<:AbstractOrthogMethod}

    V_φ = get_aux_space(pcfs.state_map)
    history = HilbertianProjectionHistory(max_iters,N)
    vel = get_free_dof_values(interpolate(0,V_φ))
    dC = pcfs.dC;
    α = zeros(eltype(φ),length(dC))
    orthog_cache = allocate_cache(orthog_method,dC)
    φ_tmp = zero(φ);
    cache = (φ,φ_tmp,pcfs,stencil,vel_ext,interp,vel,el_size)
    line_search_cache = (ls_max_iters,ls_δ_inc,ls_δ_dec,ls_ξ,
      ls_ξ_reduce_coef,ls_ξ_reduce_abs_tol,ls_γ_min,ls_γ_max);
    γ_cache = [γ,γ_reinit]
    new{T}(λ,α_min,α_max,α,max_iters,history,conv_criterion,
      orthog_method,verbose,orthog_cache,line_search_cache,γ_cache,cache)
  end
end

get_optimiser_history(m::HilbertianProjection) = m.history
get_level_set(m::HilbertianProjection) = FEFunction(get_aux_space(m.cache[3]),first(m.cache))

# Stopping criterion
function conv_cond(m::HilbertianProjection;coef=1/5)
  _,_,_,_,_,_,_,el_size = m.cache
  history = m.history
  it,Ji,Ci = last(history)

  return it > 10 && (all(@.(abs(Ji-history.J[it-5:it])) .< coef/maximum(el_size)*abs(Ji)) &&
    all(@. abs(Ci) < 0.001))
end

# 0th iteration
function Base.iterate(m::HilbertianProjection)
  φ,φ_tmp,pcfs,_,vel_ext,_,_,_ = m.cache
  K = vel_ext.K; verbose=m.verbose
  ## Compute FE problem and shape derivatives
  J_init,C_init,dJ,dC = Gridap.evaluate!(pcfs,φ)
  update!(m.history,J_init,C_init)
  ## Hilbertian extension-regularisation
  project!(vel_ext,dJ)
  project!(vel_ext,dC)
  ## Compute descent direction θ and store in dJ
  θ = update_descent_direction!(m,dJ,dJ,C_init,dC,K;verbose=verbose)
  return m.history,θ
end

# ith iteration
function Base.iterate(m::HilbertianProjection,θ)
  φ,φ_tmp,pcfs,stencil,vel_ext,_,vel,_ = m.cache
  ls_max_iters,δ_inc,δ_dec,ξ,ξ_reduce,
    ξ_reduce_tol,γ_min,γ_max = m.line_search_cache
  conv_criterion = m.conv_criterion
  U_reg = get_deriv_space(pcfs.state_map)
  V_φ = get_aux_space(pcfs.state_map)
  history = m.history
  it = history.it
  J_prev = iszero(it) ? 0 : history.J[it]
  γ,γ_reinit = m.γ_cache
  verbose = m.verbose
  ## Line search
  interpolate!(FEFunction(U_reg,θ),vel,V_φ)
  copy!(φ_tmp,φ);
  for _ ∈ Base.OneTo(ls_max_iters)
    ## Advect  & Reinitialise
    advect!(stencil,φ,vel,γ)
    reinit!(stencil,φ,γ_reinit)
    ## Calcuate new objective and constraints
    J_interm,C_interm = evaluate_functionals!(pcfs,φ)
    ## Reduce line search parameter if constraints close to saturation
    _ξ = all(@. abs(C_interm) < ξ_reduce_tol) ? ξ*ξ_reduce : ξ
    ## Accept/reject
    if J_interm < J_prev + _ξ*abs(J_prev) || γ <= γ_min || iszero(it)
      γ = min(δ_inc*γ,γ_max)
      ~isnothing(verbose) && i_am_main(verbose) ?
        printstyled("  Accepted iteration with γ = ",γ,"\n";color = :yellow) : 0;
      break
    else
      γ = max(δ_dec*γ,γ_min)
      copy!(φ,φ_tmp);
      ~isnothing(verbose) && i_am_main(verbose) ?
        printstyled("  Reject iteration with γ = ",γ,"\n"; color = :red) : 0;
    end
  end
  m.γ_cache[1] = γ
  ## Calculate objective, constraints, and shape derivatives after line search
  J_new,C_new,dJ,dC = Gridap.evaluate!(pcfs,φ)
  ## Hilbertian extension-regularisation
  project!(vel_ext,dJ)
  project!(vel_ext,dC)
  ## Compute descent direction θ and store in dJ
  θ = update_descent_direction!(m,dJ,dJ,C_new,dC,vel_ext.K;verbose=verbose)
  ## History
  update!(history,J_new,C_new)
  if conv_criterion(m) || it + 1 >= m.max_iters
    return nothing
  else
    return history,θ
  end
end

## Methods for descent direction
function update_descent_direction!(m::HilbertianProjection,θ,dV,C,dC,K;verbose=nothing)
  α=m.α; λ=m.λ; α_min=m.α_min; α_max=m.α_max
  orthog_cache=m.orthog_cache; orthogonalise = m.orthog_method
  # Orthogonalisation of dC
  dC_orthog,normsq,nullity = orthogonalise(dC,K,orthog_cache);
  # Project dV
  dV_norm = project_dV!(dV,dC_orthog,normsq,K)
  # Calculate αᵢ
  ∑α²,debug_code = compute_α!(α,C,dC_orthog,dC,normsq,K,λ,α_min,α_max)
  # Print debug info (if requested)
  verbose_print(orthogonalise,dV,dC,dC_orthog,K,λ,nullity,debug_code,verbose)
  # Calculate direction
  idx_non_zero = (Base.OneTo(last(orthog_cache)))[.~iszero.(normsq)]
  θ .= sqrt(1-∑α²)*dV/dV_norm
  length(idx_non_zero)>0 ? θ .+= sum(α[i] * dC_orthog[i] / sqrt(normsq[i]) for i ∈ idx_non_zero) : nothing
  return θ
end

# Form projection operator for Cperp: P_C⟂(dV) = dV - ∑[(̄vᵢ ⋅ m)/|̄vᵢ|² ̄vᵢ] 
#  where ̄vᵢ spans C and ̄vᵢ ⋅ ̄vⱼ = δᵢⱼ. Return norm of projected dV
function project_dV!(dV,dC_orthog,normsq,K)
  for i = 1:length(dC_orthog)
    if ~iszero(normsq[i])
      dV .-= dot(dC_orthog[i],K,dV)/normsq[i]*dC_orthog[i] # [(̄vᵢ ⋅ m)/|̄vᵢ|² ̄vᵢ]
    end
  end
  # Check if projected dV is zero vector
  dV_norm = sqrt(dot(dV,K,dV))
  if dV_norm ≈ zero(eltype(normsq))
    return one(eltype(normsq))
  else
    return dV_norm
  end
end

# Compute α coefficents using αᵢ|̄vᵢ| = λCᵢ - ∑ⱼ₌₁ⁱ⁻¹[ αⱼ(̄vⱼ⋅vᵢ)/|̄vⱼ| ]
function compute_α!(α,C,dC_orthog,dC,normsq,K,λ,α_min,α_max)
  copyto!(α,C);
  for i = 1:length(C)
    if iszero(normsq[i])
      α[i] = zero(eltype(α))
    else
      for j = 1:i-1
        if ~iszero(normsq[j])
          # Compute -∑ⱼ₌₁ⁱ⁻¹[ αⱼ(̄vⱼ⋅vᵢ)/|̄vⱼ| ]
          α[i] -= α[j]*dot(dC_orthog[j],K,dC[i])/sqrt(normsq[j])
        end
      end
      α[i] /= sqrt(normsq[i]) # |̄vᵢ|
    end
  end

  # Scale α according to α_min/α_max
  ∑α² =λ^2*dot(α,α);
  debug_code = 0;
  if ∑α² > α_max
      λ *= sqrt(α_max/∑α²)
      ∑α² = α_max
      debug_code = 1;
  elseif ∑α² < α_min
      λ *= sqrt(α_min/∑α²)
      ∑α² = α_min
      debug_code = 2;
  end
  α .*= λ
  return ∑α²,debug_code
end

"""
  HPModifiedGramSchmidt

  High performance modified Gram-Schmidt. Based on https://doi.org/10.1007/s13160-019-00356-4.
"""
struct HPModifiedGramSchmidt <: AbstractOrthogMethod end

function (m::HPModifiedGramSchmidt)(A::AbstractArray{<:AbstractArray},K::AbstractMatrix)
  cache = allocate_cache(m,A)
  return m(A,K,cache)
end

function (::HPModifiedGramSchmidt)(A::AbstractArray{<:AbstractArray},K::AbstractMatrix,cache)
  Z,Q,R,null_R,P,X,n = cache
  nullity = 0
  copyto!.(Z,A)
  mul!.(X,(K,),Z)
  for i = 1:n
    R[i,i]=dot(Z[i],X[i])
    # Check nullity
    if R[i,i] < 10^-14
      R[i,i] = 1;
      null_R[i,i] = 1;
      nullity += 1
    end
    R[i,i] = sqrt(R[i,i])
    Q[i] = Z[i]/R[i,i]
    P .= X[i]/R[i,i]
    for j = i+1:n
      R[i,j]= dot(P,Z[j])
      Z[j] -= R[i,j]*Q[i]
    end
    X[i] -= sum(R[i,j]*P for j = i+1:n;init=zero(Q[1]))
  end
  R[findall(null_R)] .= zero(eltype(R))

  return Z,R[diagind(R)].^2,nullity
end

function allocate_cache(::HPModifiedGramSchmidt,A::AbstractArray{<:AbstractArray})
  n = length(A);
  Z = [zero(first(A)) for _ = 1:n];
  Q = [zero(first(A)) for _ = 1:n];
  R = zeros(n,n)
  null_R = zeros(Bool,size(R))
  P = zero(first(A));
  X = [zero(first(A)) for _ = 1:n];
  return Z,Q,R,null_R,P,X,n
end

# Verbose printing
function verbose_print(orthog,dV,dC,dC_orthog,K,λ,nullity,debug_code,verbose)
  if ~isnothing(verbose)
    orth_norm = zeros(length(dC),length(dC))
    for i = 1:length(dC),j = 1:length(dC)
      orth_norm[i,j]=dot(dC_orthog[i],K,dC_orthog[j]);
    end
    proj_norm = dot.((dV,),(K,),dC_orthog)
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

# function test(n,k)
#   λ,α_min,α_max = 0.1,0.01,1.0
#   debug = true

#   C = rand(n)
#   dC = [rand(k) for _ = 1:n];
#   α = zeros(n)
#   dC[1] = dC[2] = dC[3];
#   _M = sprand(Float64,k,k,0.1)
#   K = 0.5*(_M+_M') + k*I;
#   dV = K\rand(k)

#   mgs_hp = HPModifiedGramSchmidt()
#   dC_orthog,normsq,nullity = mgs_hp(dC,K);
  
#   dV_norm = project_dV!(dV,dC_orthog,normsq,K)

#   ∑α²,debug_code = compute_α!(α,C,dC_orthog,dC,normsq,K,λ,α_min,α_max)

#   debug_print(mgs_hp,dV,dC,dC_orthog,K,λ,nullity,debug_code,debug)

# end

# test(5,10);