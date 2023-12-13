"""
  AbstractOptimiser

  Your own optimiser can be implemented by implementing 
    concrete functionality of the below.
"""
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

include("./AugmentedLagrangian.jl")
include("./HilbertianProjection.jl")