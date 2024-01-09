
struct HilbertianProjection{T,N} <: Optimiser
  problem   :: PDEConstrainedFunctionals{N}
  stencil   :: AdvectionStencil
  vel_ext   :: VelocityExtension
  orthog    :: OrthogonalisationMethod
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
    ls_ξ = 0.005, ls_ξ_reduce_coef = 0.1, ls_ξ_reduce_abs_tol = 0.01,
    ls_γ_min = 0.001, ls_γ_max = 0.1,
    maxiter = 1000, verbose=false, constraint_names = map(i -> Symbol("C_$i"),1:N),
    converged::Function = default_hp_converged, debug = false
  ) where {N}

    al_keys = [:J,constraint_names...]
    al_bundles = Dict(:C => constraint_names)
    history = OptimiserHistory(Float64,al_keys,al_bundles,maxiter,verbose)

    params = (;debug,λ,α_min,α_max,γ,γ_reinit,ls_max_iters,ls_δ_inc,ls_δ_dec,ls_ξ,
      ls_ξ_reduce_coef,ls_ξ_reduce_abs_tol,ls_γ_min,ls_γ_max)
    T = typeof(orthog)
    new{T,N}(problem,stencil,vel_ext,orthog,history,converged,params,φ0)
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

  ## Compute descent direction θ and store in dJ
  θ = update_descent_direction!(m,dJ,dJ,C,dC,m.vel_ext.K;verbose=true)
  
  # Update history and build state
  push!(history,(J,C...))
  state = (1,J,C,θ,dJ,dC,uh,φh,vel,φ_tmp,params.γ)
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

  ## Hilbertian extension-regularisation
  project!(m.vel_ext,dJ)
  project!(m.vel_ext,dC)

  ## Compute descent direction θ and store in dJ
  θ = update_descent_direction!(m,dJ,dJ,C,dC,m.vel_ext.K) # TODO: Wtf so dJ = θ? Then why treat them separately? 

  ## Update history and build state
  push!(history,(J,C...))
  state = (it+1, J, C, θ, dJ, dC, uh, φh, vel, φ_tmp, γ)
  vars  = params.debug ? (it,uh,φh,state) : (it,uh,φh)
  return vars, state
end

## Methods for descent direction
function update_descent_direction!(m::HilbertianProjection,θ::PVector,dV,C,dC,K;verbose=nothing)
  # Change ghosts of dV and dC (This will change in future as allocates TODO)
  dV = GridapDistributed.change_ghost(dV,axes(K,2))
  dC = GridapDistributed.change_ghost.(dC,(axes(K,2),))
  _update_descent_direction!(m::HilbertianProjection,θ,dV,C,dC,K,verbose)
end

function update_descent_direction!(m::HilbertianProjection,θ::Vector,dV,C,dC,K;verbose=nothing)
  _update_descent_direction!(m::HilbertianProjection,θ,dV,C,dC,K,verbose)
end

function _update_descent_direction!(m::HilbertianProjection,θ,dV,C,dC,K,verbose)
  orthogonalise = m.orthog
  orthog_cache = allocate_cache(orthogonalise,dC,K)
  _,_,_,_,P,_,_ = orthog_cache

  # Orthogonalisation of dC
  dC_orthog, normsq, nullity = orthogonalise(dC,K,orthog_cache)

  # Project dV
  dV_norm = project_dV!(dV,dC_orthog,normsq,K,P)

  # Calculate αᵢ
  λ, α_min, α_max = m.params.λ, m.params.α_min, m.params.α_max
  α, ∑α²,debug_code = compute_α(C,dC_orthog,dC,normsq,K,P,λ,α_min,α_max)

  # Print debug info (if requested)
  verbose_print(orthogonalise,dV,dC,dC_orthog,K,P,λ,nullity,debug_code,verbose)

  # Calculate direction
  θ .= sqrt(1-∑α²)*dV/dV_norm
  for (i,n) in enumerate(normsq)
    if !iszero(n)
      θ .+= (α[i]/sqrt(n)) .* dC_orthog[i] 
    end
  end
  return θ
end

# Form projection operator for Cperp: P_C⟂(dV) = dV - ∑[(̄vᵢ ⋅ m)/|̄vᵢ|² ̄vᵢ] 
#  where ̄vᵢ spans C and ̄vᵢ ⋅ ̄vⱼ = δᵢⱼ. Return norm of projected dV
function project_dV!(dV,dC_orthog,normsq,K,P)
  mul!(P,K,dV)
  for (i,n) in enumerate(normsq)
    if !iszero(n)
      αi = dot(dC_orthog[i],P)
      dV .-= (αi/n) .* dC_orthog[i] # [(̄vᵢ ⋅ m)/|̄vᵢ|² ̄vᵢ]
    end
  end
  # Check if projected dV is zero vector
  mul!(P,K,dV)
  dV_norm = sqrt(dot(dV,P))
  if iszero(dV_norm)
    return one(eltype(normsq))
  else
    return dV_norm
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
