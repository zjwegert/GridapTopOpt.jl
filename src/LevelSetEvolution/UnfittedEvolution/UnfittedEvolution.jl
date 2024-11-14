"""
    abstract type Evolver

Your own unfitted level-set evolution method can be created by implementing 
concrete functionality for `solve!`.
"""
abstract type Evolver end

"""
    solve!(::Evolver,φ,args...)

Evolve the level set function φ according to an Evolution method.
"""
function solve!(::Evolver,φ,args...)
  @abstractmethod
end

"""
    abstract type Reinitialiser

Your own unfitted level-set reinitialisation method can be created by implementing 
concrete functionality for `solve!`.
"""
abstract type Reinitialiser end

"""
    solve!(::Reinitialiser,φ,args...)

Reinitialise the level set function φ according to an Reinitialiser method.
"""
function solve!(::Reinitialiser,φ,args...)
  @abstractmethod
end

"""
    struct UnfittedFEEvolution{A<:Evolver,B<:Reinitialiser} <: LevelSetEvolution

Wrapper for unfitted evolution and reinitialisation methods.
"""
struct UnfittedFEEvolution{A<:Evolver,B<:Reinitialiser} <: LevelSetEvolution
  evolver::A
  reinitialiser::B
end

function evolve!(s::UnfittedFEEvolution,φ::T,vel::M,γ,args...) where 
    {T<:AbstractVector,M<:AbstractVector}
  V_φ = get_space(s.evolver)
  φh = FEFunction(V_φ,φ)
  velh = FEFunction(V_φ,vel)
  evolve!(s,φh,velh,γ,args...)
end

function evolve!(s::UnfittedFEEvolution,φ::T,velh,γ,args...) where T<:AbstractVector
  V_φ = get_space(s.evolver)
  φh = FEFunction(V_φ,φ)
  evolve!(s,φh,velh,γ,args...)
end

function evolve!(s::UnfittedFEEvolution,φh,vel::T,γ,args...) where T<:AbstractVector
  V_φ = get_space(s.evolver)
  velh = FEFunction(V_φ,vel)
  evolve!(s,φh,velh,γ,args...)
end

function evolve!(s::UnfittedFEEvolution,φh,velh,γ,args...)
  cache = get_cache(s.evolver)
  solve!(s.evolver,φh,velh,γ,cache)
  return get_free_dof_values(φh)
end

function reinit!(s::UnfittedFEEvolution,φ::T,args...) where T<:AbstractVector
  V_φ = get_space(s.reinitialiser)
  φh = FEFunction(V_φ,φ)
  reinit!(s,φh,args...)
end

function reinit!(s::UnfittedFEEvolution,φh,args...)
  cache = get_cache(s.reinitialiser)
  solve!(s.reinitialiser,φh,cache)
  return get_free_dof_values(φh)
end

function get_dof_Δ(s::UnfittedFEEvolution)
  V_φ = get_space(s.evolver)
  _,h,_ = get_params(s.evolver)
  return h/get_order(V_φ)
end

include("CutFEMEvolve.jl")
include("StabilisedReinit.jl")
include("MutableRungeKutta.jl")