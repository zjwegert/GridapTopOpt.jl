abstract type AbstractOptimiser end

# Return tuple of first iteration state
function Base.iterate(::T) where T <: AbstractOptimiser
  @notimplemented
end

# Return tuple of next iteration state given current state
function Base.iterate(::T,state) where T <: AbstractOptimiser
  @notimplemented
end

# Getters
get_optimiser_history(::AbstractOptimiser) = @notimplemented
get_level_set(::AbstractOptimiser) = @notimplemented

## Augmented Lagrangian optimiser
mutable struct AugmentedLagrangianHistory
  # I'm not the biggest fan of this but it solves the problem of not
  #  having the explicit iteration count stored in a useful way.
  #  It also means we return a single item in iterator loop.
  it      :: Int
  const J :: Vector
  const C :: Matrix
  const L :: Vector
  function AugmentedLagrangianHistory(max_iters::Int,nconsts::Int)
    J = zeros(max_iters+1);
    C = zeros(max_iters+1,nconsts)
    L = zero(J)
    new(-1,J,C,L)
  end
end

# Convienence 
Base.last(m::AugmentedLagrangianHistory) = (m.it,m.J[m.it+1],m.C[m.it+1,:],m.L[m.it+1])

function write_history(m::AugmentedLagrangianHistory,path;ranks=nothing)
  it = m.it; J = m.J; C = m.C; L = m.L
  if i_am_main(ranks) 
    if length(C) > 0
      data = @views zip(J[1:it],eachslice(C[1:it,:],dims=2)...,L[1:it])
    else
      data = @views zip(J[1:it],L[1:it])
    end
    writedlm(path,data)
  end
end

function update!(m::AugmentedLagrangianHistory,J,C,L)
  m.it += 1
  m.J[m.it+1] = J 
  length(C)>0 ? m.C[m.it+1,:] .= C : nothing 
  m.L[m.it+1] = L
  return nothing
end

struct AugmentedLagrangian <: AbstractOptimiser
  λ               :: Vector
  Λ               :: Vector
  Λ_max           :: Real
  ζ               :: Real
  update_mod      :: Int
  max_iters       :: Int
  history         :: AugmentedLagrangianHistory
  conv_criterion  :: Function
  γ_cache
  cache
  function AugmentedLagrangian(
      φ::AbstractVector,
      pcfs::PDEConstrainedFunctionals{N},
      stencil,vel_ext,interp,el_size,γ,γ_reinit;
      λ::Vector=zeros(N),
      Λ::Vector=zeros(N),
      Λ_max = 5,
      ζ = 1.1,
      update_mod::Int = 5,
      max_iters::Int = 1000,
      conv_criterion::Function = conv_cond) where {N}

    V_φ = get_aux_space(pcfs.state_map)
    history = AugmentedLagrangianHistory(max_iters,N)
    vel = get_free_dof_values(interpolate(0,V_φ))
    cache = (φ,pcfs,stencil,vel_ext,interp,vel,el_size)
    γ_cache = [γ,γ_reinit]
    new(λ,Λ,Λ_max,ζ,update_mod,max_iters,history,conv_criterion,γ_cache,cache)
  end
end

get_optimiser_history(m::AugmentedLagrangian) = m.history
get_level_set(m::AugmentedLagrangian) = FEFunction(get_aux_space(m.cache[2]),first(m.cache))

# Initialise AGM parameters
function initialise!(m::AugmentedLagrangian,J_init::Real,C_init::Vector)
  λ = m.λ; Λ = m.Λ;
  λ .= 0.0
  Λ .= @. 0.1*abs(J_init)/abs(C_init)^1.5
  return λ,Λ
end

# Update AGM parameters
function update!(m::AugmentedLagrangian,iter::Int,C_new::Vector)
  λ = m.λ; Λ = m.Λ; ζ = m.ζ;
  Λ_max = m.Λ_max;
  λ .= λ - Λ.*C_new;
  iszero(iter % m.update_mod) ? Λ .= @.(min(Λ*ζ,Λ_max)) : 0;
  return λ,Λ
end

# Stopping criterion
function conv_cond(m::AugmentedLagrangian;coef=1/5)
  _,_,_,_,_,_,el_size = m.cache
  history = m.history
  it,Ji,Ci,Li = last(history)

  return it > 10 && (all(@.(abs(Li-history.L[it-5:it])) .< coef/maximum(el_size)*abs(Li)) &&
    all(@. abs(Ci) < 0.001))
end

# 0th iteration
function Base.iterate(m::AugmentedLagrangian)
  φ,pcfs,_,vel_ext,_,_,_ = m.cache
  ## Compute FE problem and shape derivatives
  J_init,C_init,dJ,dC = Gridap.evaluate!(pcfs,φ)

  ## Compute initial values
  λ,Λ = initialise!(m,J_init,C_init)
  L_init = J_init
  length(C_init)>0 ? L_init += sum(@.(-λ*C_init + Λ/2*C_init^2)) : nothing
  update!(m.history,J_init,C_init,L_init)

  ## Compute dL and projectzero(dJ)
  dL = dJ
  length(C_init)>0 ? dL += sum(-λ[i]*dC[i] + Λ[i]*C_init[i]*dC[i] for i ∈ eachindex(λ)) : nothing
  # Because project! takes a linear form on the RHS this should
  #   be the same as projecting each shape derivative then computing dL
  project!(vel_ext,dL)

  return m.history,dL
end

# ith iteration
function Base.iterate(m::AugmentedLagrangian,dL)
  φ,pcfs,stencil,vel_ext,_,vel,_ = m.cache
  conv_criterion = m.conv_criterion
  U_reg = get_deriv_space(pcfs.state_map)
  V_φ = get_aux_space(pcfs.state_map)
  history = m.history
  it = history.it

  ## Advect & Reinitialise
  if it > 10 && m.γ_cache[1]>0.001 && all(isapprox.(history.L[it + 1],history.L[it + 1 .- [2,4,6]];rtol=10^-4))
    m.γ_cache[1] *= 3/4
    printstyled("   Oscillations detected, reducing γ to $(m.γ_cache[1])\n",color=:yellow)
  end
  γ,γ_reinit = m.γ_cache
  interpolate!(FEFunction(U_reg,dL),vel,V_φ)
  advect!(stencil,φ,vel,γ)
  reinit!(stencil,φ,γ_reinit)

  ## Calculate objective, constraints, and shape derivatives
  J_new,C_new,dJ,dC = Gridap.evaluate!(pcfs,φ)
  L_new = J_new
  length(C_new)>0 ? L_new += sum(@. -m.λ*C_new + m.Λ/2*C_new^2) : nothing
  
  ## Augmented Lagrangian method
  λ,Λ = update!(m,it,C_new)
  dL = dJ
  length(C_new)>0 ? dL += sum(-λ[i]*dC[i] + Λ[i]*C_new[i]*dC[i] for i ∈ eachindex(λ)) : nothing
  project!(vel_ext,dL)

  ## History
  update!(history,J_new,C_new,L_new)

  if conv_criterion(m) || it >= m.max_iters
    return nothing
  else
    return history,dL
  end
end