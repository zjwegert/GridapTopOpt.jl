"""
  AbstractOptimiser

  Your own optimiser can be implemented by implementing 
    concrete functionality of the below.
"""
abstract type AbstractOptimiser end

# Return tuple of first iteration state
function Base.iterate(::T) where T <: AbstractOptimiser
  @abstractmethod
end

# Return tuple of next iteration state given current state
function Base.iterate(::T,state) where T <: AbstractOptimiser
  @abstractmethod
end

get_optimiser_history(::AbstractOptimiser) = @abstractmethod
get_level_set(::AbstractOptimiser) = @abstractmethod

print_history(opt::AbstractOptimiser) = print_history(get_optimiser_history(opt))
print_history(opt::AbstractOptimiser,it::Int) = print_history(get_optimiser_history(opt),it)
print_msg(opt::AbstractOptimiser,msg::String;kwargs...) = print_msg(get_optimiser_history(opt),msg;kwargs...)


# Optimiser history

abstract type AbstractOptimiserHistory end

function Base.length(::AbstractOptimizerHistory)
  @abstractmethod
end

function Base.getindex(::AbstractOptimizerHistory)
  @abstractmethod
end

function Base.push!(::AbstractOptimizerHistory)
  @abstractmethod
end

function get_verbose_level(::AbstractOptimizerHistory)
  @abstractmethod
end

function update!(h::AbstractOptimizerHistory,args...)
  push!(h,args...)
  if get_verbose_level(h) > SOLVER_VERBOSE_NONE
    print_history(h,length(h))
  end
end

function Base.display(h::AbstractOptimizerHistory)
  for (i,quants) in enumerate(h)
    it = i - 1
    println("Iteration: $it | $(join([string(k,"=",v) for (k,v) in quants],", "))")
  end
end

function print_history(h::AbstractOptimizerHistory,it::Int)
  quants = h[it+1]
  println("Iteration: $it | $(join([string(k)*": "*string(v) for (k,v) in quants],", "))")
end

function print_history(h::AbstractOptimizerHistory)
  for it in 0:length(h)
    print_history(h,it)
  end
end

function print_msg(h::AbstractOptimizerHistory,msg::String;kwargs...)
  if get_verbose_level(h) > SOLVER_VERBOSE_NONE
    printstyled(msg;kwargs...)
  end
end

include("./AugmentedLagrangian.jl")
include("./HilbertianProjection.jl")