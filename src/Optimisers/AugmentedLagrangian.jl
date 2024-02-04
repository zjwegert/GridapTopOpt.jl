"""
  struct AugmentedLagrangian{N,O} <: Optimiser end
"""
struct AugmentedLagrangian{N,O} <: Optimiser
  problem           :: PDEConstrainedFunctionals{N}
  stencil           :: AdvectionStencil{O}
  vel_ext           :: VelocityExtension
  history           :: OptimiserHistory{Float64}
  converged         :: Function
  has_oscillations  :: Function
  params            :: NamedTuple
  φ0 # TODO: Please remove me
  function AugmentedLagrangian(
    problem :: PDEConstrainedFunctionals{N},
    stencil :: AdvectionStencil{O},
    vel_ext :: VelocityExtension,
    φ0;
    Λ_max = 5.0, ζ = 1.1, update_mod = 5, γ = 0.1, γ_reinit = 0.5,
    maxiter = 1000, verbose=false, constraint_names = map(i -> Symbol("C_$i"),1:N),
    converged::Function = default_al_converged, debug = false,
    has_oscillations::Function = default_has_oscillations
  ) where {N,O}

    constraint_names = map(Symbol,constraint_names)
    al_keys = [:L,:J,constraint_names...,:γ]
    al_bundles = Dict(:C => constraint_names)
    history = OptimiserHistory(Float64,al_keys,al_bundles,maxiter,verbose)

    params = (;Λ_max,ζ,update_mod,γ,γ_reinit,debug)
    new{N,O}(problem,stencil,vel_ext,history,converged,has_oscillations,params,φ0)
  end
end

get_history(m::AugmentedLagrangian) = m.history

function default_has_oscillations(m::AugmentedLagrangian;tol=1.e-4,reps=4)
  h  = m.history
  it = get_last_iteration(h)
  if it < 10
    return false
  end

  L = h[:L]
  return all(k -> abs(L[it+1] - L[k+1]) < tol*L[it+1], it .- (2:2:2reps))
  # return all(k -> abs(L[it+1] - L[k+1]) < tol, it .- (2:2:2reps))
end

function converged(m::AugmentedLagrangian)
  return m.converged(m)
end

function default_al_converged(
  m::AugmentedLagrangian;
  L_tol = 0.01*maximum(m.stencil.params.Δ), # 0.2*maximum(m.stencil.params.Δ),
  C_tol = 0.001
)
  h  = m.history
  it = get_last_iteration(h)
  if it < 10
    return false
  end

  Li, Ci = h[:L,it], h[:C,it]
  L_prev = h[:L,it-5:it-1]
  A = all(L -> abs(Li - L)/abs(Li) < L_tol, L_prev)
  B = all(C -> abs(C) < C_tol,Ci)
  return A && B
end

function Base.iterate(m::AugmentedLagrangian)
  φh, history, params = m.φ0, m.history, m.params

  ## Reinitialise as SDF
  reinit!(m.stencil,φh,params.γ_reinit)

  ## Compute FE problem and shape derivatives
  J, C, dJ, dC = evaluate!(m.problem,φh)
  uh  = get_state(m.problem)
  vel = copy(get_free_dof_values(φh))

  ## Compute initial lagrangian
  λ = zeros(eltype(J),length(C))
  Λ = convert(Vector{eltype(J)},map(Ci -> 0.1*abs(J)/abs(Ci)^1.5,C))
  L = J
  for (λi,Λi,Ci) in zip(λ,Λ,C)
    L += -λi*Ci + 0.5*Λi*Ci^2
  end

  ## Compute dL and it's projection
  dL = copy(dJ)
  for (λi,Λi,Ci,dCi) in zip(λ,Λ,C,dC)
    dL .+= -λi*dCi .+ Λi*Ci*dCi
  end
  project!(m.vel_ext,dL)

  # Update history and build state
  push!(history,(L,J,C...,params.γ))
  state = (;it=1,L,J,C,dL,dJ,dC,uh,φh,vel,λ,Λ,params.γ)
  vars  = params.debug ? (0,uh,φh,state) : (0,uh,φh)
  return vars, state
end

function Base.iterate(m::AugmentedLagrangian,state)
  it, L, J, C, dL, dJ, dC, uh, φh, vel, λ, Λ, γ = state
  params, history = m.params, m.history
  update_mod, ζ, Λ_max, γ_reinit = params.update_mod, params.ζ, params.Λ_max, params.γ_reinit

  if finished(m)
    return nothing
  end

  ## Advect & Reinitialise
  if (γ > 0.001) && m.has_oscillations(m)
    γ *= 3/4
    print_msg(m.history,"   Oscillations detected, reducing γ to $(γ)\n",color=:yellow)
  end

  U_reg = get_deriv_space(m.problem.state_map)
  V_φ = get_aux_space(m.problem.state_map)
  interpolate!(FEFunction(U_reg,dL),vel,V_φ)
  advect!(m.stencil,φh,vel,γ)
  reinit!(m.stencil,φh,γ_reinit)

  ## Calculate objective, constraints, and shape derivatives
  J, C, dJ, dC = evaluate!(m.problem,φh)
  uh = get_state(m.problem)
  L  = J
  for (λi,Λi,Ci) in zip(λ,Λ,C)
    L += -λi*Ci + 0.5*Λi*Ci^2
  end

  ## Augmented Lagrangian method
  λ .= λ .- Λ .* C
  if iszero(it % update_mod) 
    Λ .= @.(min(Λ*ζ,Λ_max))
  end

  ## Compute dL and it's projection
  copy!(dL,dJ)
  for (λi,Λi,Ci,dCi) in zip(λ,Λ,C,dC)
    dL .+= -λi*dCi .+ Λi*Ci*dCi
  end
  project!(m.vel_ext,dL)

  ## Update history and build state
  push!(history,(L,J,C...,γ))
  state = (it+1,L,J,C,dL,dJ,dC,uh,φh,vel,λ,Λ,γ)
  vars  = params.debug ? (it,uh,φh,state) : (it,uh,φh)
  return vars, state
end
