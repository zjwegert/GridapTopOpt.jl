"""
    struct AugmentedLagrangian <: Optimiser

An augmented Lagrangian method based on Nocedal and Wright, 2006
([link](https://doi.org/10.1007/978-0-387-40065-5)). Note that
this method will function as a Lagrangian method if no constraints
are defined in `problem::AbstractPDEConstrainedFunctionals`.

# Parameters

- `problem::AbstractPDEConstrainedFunctionals`: The objective and constraint setup.
- `ls_evolver::LevelSetEvolution`: Solver for the evolution and reinitisation equations.
- `vel_ext::VelocityExtension`: The velocity-extension method for extending
  shape sensitivities onto the computational domain.
- `history::OptimiserHistory{Float64}`: Historical information for optimisation problem.
- `converged::Function`: A function to check optimiser convergence.
- `has_oscillations::Function`: A function to check for oscillations.
- `params::NamedTuple`: Optimisation parameters.

The `has_oscillations` function has been added to avoid oscillations in the
iteration history. By default this uses a mean zero crossing algorithm as implemented
in ChaosTools. Oscillations checking can be disabled by taking `has_oscillations = (args...) -> false`.
"""
struct AugmentedLagrangian <: Optimiser
  problem           :: AbstractPDEConstrainedFunctionals
  ls_evolver        :: LevelSetEvolution
  vel_ext           :: VelocityExtension
  history           :: OptimiserHistory{Float64}
  converged         :: Function
  has_oscillations  :: Function
  params            :: NamedTuple
  φ0 # TODO: Please remove me

  @doc """
      AugmentedLagrangian(
        problem    :: AbstractPDEConstrainedFunctionals{N},
        ls_evolver :: LevelSetEvolution,
        vel_ext    :: VelocityExtension,
        φ0;
        Λ_max = 10^10, ζ = 1.1, update_mod = 5, γ = 0.1, γ_reinit = 0.5, os_γ_mult = 0.75,
        Λ_update_tol = 0.01,maxiter = 1000, verbose=false, constraint_names = map(i -> Symbol("C_\$i"),1:N),
        converged::Function = default_al_converged, debug = false,
        has_oscillations::Function = default_has_oscillations
      ) where {N,O}

  Create an instance of `AugmentedLagrangian` with several adjustable defaults.

  # Required

  - `problem::AbstractPDEConstrainedFunctionals`: The objective and constraint setup.
  - `ls_evolver::LevelSetEvolution`: Solver for the evolution and reinitisation equations.
  - `vel_ext::VelocityExtension`: The velocity-extension method for extending
    shape sensitivities onto the computational domain.
  - `φ0`: An initial level-set function defined as a FEFunction or GridapDistributed equivilent.

  # Optional defaults

  - `γ = 0.1`: Initial coeffient on the time step size for solving the Hamilton-Jacobi evolution equation.
  - `γ_reinit = 0.5`: Coeffient on the time step size for solving the reinitisation equation.
  - `ζ = 1.1`: Increase multiplier on Λ every `update_mod` iterations.
  - `Λ_max = 10^10`: Maximum value on any entry in Λ.
  - `update_mod = 5`: Number of iterations before increasing `Λ`.
  - `reinit_mod = 1`: How often we solve reinitialisation equation.
  - `maxiter = 1000`: Maximum number of algorithm iterations.
  - `verbose=false`: Verbosity flag.
  - `constraint_names = map(i -> Symbol("C_\$i"),1:N)`: Constraint names for history output.
  - `has_oscillations::Function = default_has_oscillations`: Function to check for oscillations
    in the history.
  - `initial_parameters::Function = default_al_init_params`: Function to generate initial λ, Λ.
    This can be replaced to inject different λ and Λ, for example.
  - `os_γ_mult = 0.75`: Decrease multiplier for `γ` when `has_oscillations` returns true
  - `Λ_update_tol = 0.01`: Tolerance of constraint satisfaction for updating Λ. In our testing, this
    is usually set to 0.01. Some problems, may perform better with a stricter tolerance (e.g.,
    0.001 or 0.0 to always update).
  - `converged::Function = default_hp_converged`: Convergence criteria.
  - `debug = false`: Debug flag.
  """
  function AugmentedLagrangian(
    problem    :: AbstractPDEConstrainedFunctionals{N},
    ls_evolver :: LevelSetEvolution,
    vel_ext    :: VelocityExtension,
    φ0;
    Λ_max = 10^10, ζ = 1.1, update_mod = 5, reinit_mod = 1, γ = 0.1, γ_reinit = 0.5,
    os_γ_mult = 0.75, Λ_update_tol = 0.01, maxiter = 1000, verbose=false, constraint_names = map(i -> Symbol("C_$i"),1:N),
    converged::Function = default_al_converged, debug = false,
    has_oscillations::Function = default_has_oscillations,
    initial_parameters::Function = default_al_init_params
  ) where N

    constraint_names = map(Symbol,constraint_names)
    λ_names = map(i -> Symbol("λ$i"),1:N)
    Λ_names = map(i -> Symbol("Λ$i"),1:N)
    al_keys = [:L,:J,constraint_names...,:γ,λ_names...,Λ_names...]
    al_bundles = Dict(:C => constraint_names, :λ => λ_names, :Λ => Λ_names)
    history = OptimiserHistory(Float64,al_keys,al_bundles,maxiter,verbose)

    params = (;Λ_max,ζ,update_mod,reinit_mod,γ,γ_reinit,os_γ_mult,Λ_update_tol,debug,initial_parameters)
    new(problem,ls_evolver,vel_ext,history,converged,has_oscillations,params,φ0)
  end
end

get_history(m::AugmentedLagrangian) = m.history

function default_has_oscillations(m::AugmentedLagrangian,os_it;itlength=25,
    itstart=2itlength)
  h  = m.history
  it = get_last_iteration(h)
  if it < itstart || it < os_it + itlength + 1
    return false
  end

  L = h[:L]
  return ~isnan(_zerocrossing_period(L[it-itlength+1:it+1]))
end

function default_al_init_params(J,C)
  λ = zeros(eltype(J),length(C))
  Λ = @. 0.1*abs(J)/abs(C)^1.5

  return λ,Λ
end

function converged(m::AugmentedLagrangian)
  return m.converged(m)
end

function default_al_converged(
  m::AugmentedLagrangian;
  L_tol = 0.01*maximum(get_dof_Δ(m.ls_evolver))/(length(get_dof_Δ(m.ls_evolver))-1),
  C_tol = 0.01
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
  V_φ = get_aux_space(get_state_map(m.problem))

  ## Reinitialise as SDF
  reinit!(m.ls_evolver,φh,params.γ_reinit)

  ## Compute FE problem and shape derivatives
  J, C, dJ, dC = evaluate!(m.problem,φh)
  uh  = get_state(m.problem)

  ## Compute initial lagrangian
  λ,Λ = params.initial_parameters(J,C)
  L = J
  for (λi,Λi,Ci) in zip(λ,Λ,C)
    L += -λi*Ci + 0.5*Λi*Ci^2
  end

  ## Compute dL and it's projection
  dL = copy(dJ)
  for (λi,Λi,Ci,dCi) in zip(λ,Λ,C,dC)
    dL .+= -λi*dCi .+ Λi*Ci*dCi
  end
  project!(m.vel_ext,FEFunction(dL,V_φ),V_φ)

  # Update history and build state
  push!(history,(L,J,C...,params.γ,λ...,Λ...))
  state = (;it=1,L,J,C,dL,dJ,dC,uh,φh,λ,Λ,params.γ,os_it=-1)
  vars  = params.debug ? (0,uh,φh,state) : (0,uh,φh)
  return vars, state
end

function Base.iterate(m::AugmentedLagrangian,state)
  it, L, J, C, dL, dJ, dC, uh, φh, λ, Λ, γ, os_it = state
  params, history = m.params, m.history
  Λ_max,ζ,update_mod,reinit_mod,_,γ_reinit,os_γ_mult,Λ_update_tol,_,_ = params

  ## Periodicially call GC
  iszero(it % 50) && GC.gc();

  ## Check stopping criteria
  if finished(m)
    return nothing
  end

  ## Advect & Reinitialise
  if (γ > 0.001) && m.has_oscillations(m,os_it)
    os_it = it + 1
    γ    *= os_γ_mult
    print_msg(m.history,"   Oscillations detected, reducing γ to $(γ)\n",color=:yellow)
  end

  V_φ = get_aux_space(get_state_map(m.problem))
  evolve!(m.ls_evolver,φh,dL,γ)
  iszero(it % reinit_mod) && reinit!(m.ls_evolver,φh,γ_reinit)

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
    for i = 1:length(C)
      if abs(C[i])>Λ_update_tol
        Λ[i] = min(Λ[i]*ζ,Λ_max)
      end
    end
  end

  ## Compute dL and it's projection
  copy!(dL,dJ)
  for (λi,Λi,Ci,dCi) in zip(λ,Λ,C,dC)
    dL .+= -λi*dCi .+ Λi*Ci*dCi
  end
  project!(m.vel_ext,FEFunction(dL,V_φ),V_φ)

  ## Update history and build state
  push!(history,(L,J,C...,γ,λ...,Λ...))
  state = (;it=it+1,L,J,C,dL,dJ,dC,uh,φh,λ,Λ,γ,os_it)
  vars  = params.debug ? (it,uh,φh,state) : (it,uh,φh)
  return vars, state
end
