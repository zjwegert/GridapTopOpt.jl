# Projection map
struct HilbertianProjectionMap{A}
  orthog  :: OrthogonalisationMap
  vel_ext :: VelocityExtension{A}
  caches
  params
  function HilbertianProjectionMap(
    nC :: Int,
    orthog::OrthogonalisationMap,
    vel_ext::VelocityExtension{A};
    λ=0.5, α_min=0.1, α_max=1.0, debug=false
  ) where A <: AbstractMatrix
    θ = allocate_in_domain(vel_ext.K)
    θ_aux = allocate_in_domain(vel_ext.K)
    orth_caches = return_cache(orthog,fill(θ,nC),vel_ext.K)
    caches = (θ,θ_aux,orth_caches)
    params = (;λ,α_min,α_max,debug)
    return new{A}(orthog,vel_ext,caches,params)
  end
  function HilbertianProjectionMap(
    nC :: Int,
    orthog::OrthogonalisationMap,
    vel_ext::VelocityExtension{A};
    λ=0.5, α_min=0.1, α_max=1.0, debug=false
  ) where A <: PSparseMatrix
    θ  = allocate_in_domain(vel_ext.K)
    dC = [allocate_in_domain(vel_ext.K) for _ = 1:nC]
    θ_aux = allocate_in_domain(vel_ext.K)
    orth_caches = return_cache(orthog,dC,vel_ext.K)
    caches = (θ,θ_aux,dC,orth_caches)
    params = (;λ,α_min,α_max,debug)
    return new{A}(orthog,vel_ext,caches,params)
  end
end

function update_descent_direction!(m::HilbertianProjectionMap{<:AbstractMatrix},dV,C,dC,K)
  θ, θ_aux, orthog_cache = m.caches
  copy!(θ,dV)
  _update_descent_direction!(m,θ,C,dC,K,θ_aux,orthog_cache)
end

function update_descent_direction!(m::HilbertianProjectionMap{<:PSparseMatrix},dV,C,dC,K)
  θ, θ_aux, dC_aux, orthog_cache = m.caches
  copy!(θ,dV)
  copy!.(dC_aux,dC)
  _update_descent_direction!(m,θ,C,dC_aux,K,θ_aux,orthog_cache)
end

function _update_descent_direction!(m::HilbertianProjectionMap,θ,C,dC,K,θ_aux,orthog_cache)

  # Orthogonalisation of dC
  dC_orthog, normsq, nullity = evaluate!(orthog_cache,m.orthog,dC,K)

  # Project θ and normalize
  project_θ!(θ,dC_orthog,normsq,K,θ_aux)

  # Calculate αᵢ
  λ, α_min, α_max = m.params.λ, m.params.α_min, m.params.α_max
  α, ∑α², debug_flag = compute_α(C,dC_orthog,dC,normsq,K,θ_aux,λ,α_min,α_max)

  # Print debug info (if requested)
  debug_print(m.orthog,θ,dC,dC_orthog,K,θ_aux,nullity,debug_flag,m.params.debug)

  # Calculate direction
  θ .*= sqrt(1-∑α²)
  for (i,n) in enumerate(normsq)
    if !iszero(n)
      θ .+= (α[i]/sqrt(n)) .* dC_orthog[i] 
    end
  end
  return θ
end

# Form projection operator for Cperp: P_C⟂(θ) = θ - ∑[(̄vᵢ ⋅ m)/|̄vᵢ|² ̄vᵢ] 
#  where ̄vᵢ spans C and ̄vᵢ ⋅ ̄vⱼ = δᵢⱼ. Returns P_C⟂(θ)/norm(P_C⟂(θ))
function project_θ!(θ,dC_orthog,normsq,K,θ_aux)
  mul!(θ_aux,K,θ)
  for (i,n) in enumerate(normsq)
    if !iszero(n)
      αi = dot(dC_orthog[i],θ_aux)
      θ .-= (αi/n) .* dC_orthog[i] # [(̄vᵢ ⋅ m)/|̄vᵢ|² ̄vᵢ]
    end
  end
  # Check if projected dV is zero vector
  mul!(θ_aux,K,θ)
  θ_norm = sqrt(dot(θ,θ_aux))
  if !iszero(θ_norm)
    θ ./= θ_norm
  end
end

# Compute α coefficents using αᵢ|̄vᵢ| = λCᵢ - ∑ⱼ₌₁ⁱ⁻¹[ αⱼ(̄vⱼ⋅vᵢ)/|̄vⱼ| ]
function compute_α(C,dC_orthog,dC,normsq,K,P,λ,α_min,α_max)
  α = copy(C)
  for i = 1:length(C)
    if iszero(normsq[i])
      α[i] = zero(eltype(α))
    else
      mul!(P,K,dC[i])
      for j = 1:i-1
        if !iszero(normsq[j])
          # Compute -∑ⱼ₌₁ⁱ⁻¹[ αⱼ(̄vⱼ⋅vᵢ)/|̄vⱼ| ]
          α[i] -= α[j]*dot(dC_orthog[j],P)/sqrt(normsq[j])
        end
      end
      α[i] /= sqrt(normsq[i]) # |̄vᵢ|
    end
  end

  # Scale α according to α_min/α_max
  ∑α² = λ^2*dot(α,α)
  debug_flag = 0
  if ∑α² > α_max
    λ *= sqrt(α_max/∑α²)
    ∑α² = α_max
    debug_flag = 1
  elseif ∑α² < α_min
    λ *= sqrt(α_min/∑α²)
    ∑α² = α_min
    debug_flag = 2
  end
  α .*= λ
  return α, ∑α², debug_flag
end

# Optimiser
"""
  struct HilbertianProjection{T,N} <: Optimiser end
"""
struct HilbertianProjection{T,N} <: Optimiser
  problem   :: PDEConstrainedFunctionals{N}
  stencil   :: AdvectionStencil
  vel_ext   :: VelocityExtension
  projector :: HilbertianProjectionMap
  history   :: OptimiserHistory{Float64}
  converged :: Function
  params    :: NamedTuple
  φ0 # TODO: Remove me please
  function HilbertianProjection(
    problem :: PDEConstrainedFunctionals{N},
    stencil :: AdvectionStencil,
    vel_ext :: VelocityExtension,
    φ0;
    orthog = HPModifiedGramSchmidt(),
    λ=0.5, α_min=0.1, α_max=1.0, γ=0.1, γ_reinit=0.5,
    ls_max_iters = 10, ls_δ_inc = 1.1, ls_δ_dec = 0.7,
    ls_ξ = 0.0025, ls_ξ_reduce_coef = 0.1, ls_ξ_reduce_abs_tol = 0.01,
    ls_γ_min = 0.001, ls_γ_max = 0.1,
    maxiter = 1000, verbose=false, constraint_names = map(i -> Symbol("C_$i"),1:N),
    converged::Function = default_hp_converged, debug = false
  ) where {N}

    constraint_names = map(Symbol,constraint_names)
    al_keys = [:J,constraint_names...,:γ]
    al_bundles = Dict(:C => constraint_names)
    history = OptimiserHistory(Float64,al_keys,al_bundles,maxiter,verbose)

    projector = HilbertianProjectionMap(N,orthog,vel_ext;λ,α_min,α_max,debug)
    params = (;debug,γ,γ_reinit,ls_max_iters,ls_δ_inc,ls_δ_dec,ls_ξ,
               ls_ξ_reduce_coef,ls_ξ_reduce_abs_tol,ls_γ_min,ls_γ_max)
    T = typeof(orthog)
    new{T,N}(problem,stencil,vel_ext,projector,history,converged,params,φ0)
  end
end

get_history(m::HilbertianProjection) = m.history

function converged(m::HilbertianProjection)
  return m.converged(m)
end

function default_hp_converged(
  m::HilbertianProjection;
  J_tol = 0.2*maximum(m.stencil.params.Δ),
  C_tol = 0.001
)
  h  = m.history
  it = get_last_iteration(h)
  if it < 10
    return false
  end

  Ji, Ci = h[:J,it], h[:C,it]
  J_prev = h[:J,it-5:it]
  A = all(J -> abs(Ji - J)/abs(Ji) < J_tol, J_prev)
  B = all(C -> abs(C) < C_tol, Ci)
  return (it > 10) && A && B
end

# 0th iteration
function Base.iterate(m::HilbertianProjection)
  history, params = m.history, m.params
  φh = m.φ0

  ## Compute FE problem and shape derivatives
  J, C, dJ, dC = Gridap.evaluate!(m.problem,φh)
  uh  = get_state(m.problem)
  vel = copy(get_free_dof_values(φh))
  φ_tmp = copy(vel)

  ## Hilbertian extension-regularisation
  project!(m.vel_ext,dJ)
  project!(m.vel_ext,dC)
  θ = update_descent_direction!(m.projector,dJ,C,dC,m.vel_ext.K)
  
  # Update history and build state
  push!(history,(J,C...,params.γ))
  state = (0,J,C,θ,dJ,dC,uh,φh,vel,φ_tmp,params.γ) # TODO: it changed to 0 here.
  vars  = params.debug ? (0,uh,φh,state) : (0,uh,φh)
  return vars, state
end

# ith iteration
function Base.iterate(m::HilbertianProjection,state)
  it, J, C, θ, dJ, dC, uh, φh, vel, φ_tmp, γ = state
  history, params = m.history, m.params

  if finished(m)
    return nothing
  end
  
  ## Line search
  U_reg = get_deriv_space(m.problem.state_map)
  V_φ   = get_aux_space(m.problem.state_map)
  interpolate!(FEFunction(U_reg,θ),vel,V_φ)
  
  ls_max_iters,δ_inc,δ_dec = params.ls_max_iters,params.ls_δ_inc,params.ls_δ_dec
  ξ, ξ_reduce, ξ_reduce_tol = params.ls_ξ, params.ls_ξ_reduce_coef, params.ls_ξ_reduce_abs_tol
  γ_min, γ_max = params.ls_γ_min,params.ls_γ_max
  
  ls_it = 0; done = false
  φ = get_free_dof_values(φh); copy!(φ_tmp,φ)
  while !done && (ls_it <= ls_max_iters)
    # Advect  & Reinitialise
    advect!(m.stencil,φ,vel,γ)
    reinit!(m.stencil,φ,params.γ_reinit)

    # Calcuate new objective and constraints
    J_interm, C_interm = evaluate_functionals!(m.problem,φh)

    # Reduce line search parameter if constraints close to saturation
    _ξ = all(Ci -> abs(Ci) < ξ_reduce_tol, C_interm) ? ξ*ξ_reduce : ξ

    # Accept/reject
    if (J_interm < J + _ξ*abs(J)) || (γ <= γ_min)
      γ = min(δ_inc*γ, γ_max)
      done = true
      print_msg(history,"  Accepted iteration with γ = $(γ) \n";color=:yellow)
    else
      γ = max(δ_dec*γ, γ_min)
      copy!(φ,φ_tmp)
      print_msg(history,"  Reject iteration with γ = $(γ) \n";color=:red)
    end
  end

  ## Calculate objective, constraints, and shape derivatives after line search
  J, C, dJ, dC = Gridap.evaluate!(m.problem,φh)
  uh = get_state(m.problem)

  ## Hilbertian extension-regularisation
  project!(m.vel_ext,dJ)
  project!(m.vel_ext,dC)
  θ = update_descent_direction!(m.projector,dJ,C,dC,m.vel_ext.K)

  ## Update history and build state
  push!(history,(J,C...,γ))
  state = (it+1, J, C, θ, dJ, dC, uh, φh, vel, φ_tmp, γ)
  vars  = params.debug ? (it,uh,φh,state) : (it,uh,φh)
  return vars, state
end

function debug_print(orthog,dV,dC,dC_orthog,K,P,nullity,debug_code,debug)
  if debug
    orth_norm = 0.0
    for i = 1:length(dC)
      mul!(P,K,dC_orthog[i])
      for j = i+1:length(dC)
        orth_norm = max(orth_norm,abs(dot(dC_orthog[j],P)))
      end
    end
    mul!(P,K,dV)
    proj_norm = maximum(map(dCi -> dot(dCi,P),dC_orthog))
    println("  ↱----------------------------------------------------------↰")
    print("                   ")
    printstyled("Hilbertian Projection Method\n\n",color=:yellow,underline=true);
    println("      -->         Orthog. method: $(typeof(orthog))");
    @printf("      --> Orthogonality inf-norm: %e\n",orth_norm)
    @printf("      -->    Projection inf-norm: %e\n",proj_norm)
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
