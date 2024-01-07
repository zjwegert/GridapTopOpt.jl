"""
  AbstractOptimiser

  Your own optimiser can be implemented by implementing 
    concrete functionality of the below.
"""
abstract type AbstractOptimiser end

# Return tuple of first iteration state
function Base.iterate(::AbstractOptimiser)
  @abstractmethod
end

# Return tuple of next iteration state given current state
function Base.iterate(::AbstractOptimiser,state)
  @abstractmethod
end

function finished(m::AbstractOptimiser) :: Bool
  h = get_history(m)

  it = length(h)
  A = converged(m)
  B = it + 1 >= h.maxiter
  return A || B
end

get_history(::AbstractOptimiser) :: OptimiserHistory = @abstractmethod

function print_msg(opt::AbstractOptimiser,msg::String;kwargs...) 
  print_msg(get_optimiser_history(opt),msg;kwargs...)
end

# Optimiser history slice

struct OptimiserHistorySlice{T}
  it :: Int
  h  :: OptimiserHistory{T}
end

function Base.getproperty(s::OptimiserHistorySlice,k::Symbol)
  h = s.h
  if haskey(h.values,k) || haskey(h.bundles,k)
    return getindex(h,s.it,k)
  else
    return getfield(s,k)
  end
end

function Base.propertynames(s::OptimiserHistorySlice, private::Bool=false)
  (fieldnames(typeof(s))...,s.h.keys...,keys(s.h.bundles)...)
end

function Base.getindex(s::OptimiserHistorySlice,k::Symbol)
  return getindex(s.h,s.it,k)
end

function setindex!(::OptimiserHistorySlice,args...)
  @error "OptimiserHistorySlice is read-only!"
end

function Base.display(s::OptimiserHistorySlice)
  content = join([string(k,"=",s[k]) for k in keys(s)],", ")
  println("Iteration: $(s.it) | $(content)")
end

# Optimiser history

mutable struct OptimiserHistory{T}
  niter   :: Int
  keys    :: Vector{Symbol}
  values  :: Dict{Symbol,Vector{T}}
  bundles :: Dict{Symbol,Vector{Symbol}}
  verbose :: SolverVerboseLevel
  maxiter :: Int
end

function OptimiserHistory(
  T::Type{<:Real},
  keys::Vector{Symbol},
  bundles::Dict{Symbol,Vector{Symbol}}=Dict{Symbol,Vector{Symbol}}(),
  maxiter::Int=100,
  verbose::SolverVerboseLevel=SOLVER_VERBOSE_NONE
)
  values = Dict{Symbol,Vector{T}}()
  for k in keys
    values[k] = zeros(T,maxiter+1)
  end
  return OptimiserHistory{T}(-1,keys,values,bundles,verbose,maxiter)
end

Base.length(h::OptimizerHistory) = h.niter

function Base.getindex(h::OptimiserHistory,it::Int,k::Symbol)
  @assert (it <= h.niter)
  if haskey(h.values,k)
    return h.values[k][it+1]
  elseif haskey(h.bundles,k)
    return Tuple([h.values[ki][it+1] for ki in h.bundles[k]])
  else
    @error "Key $(k) not found in OptimiserHistory"
  end
end

function Base.getindex(h::OptimiserHistory,it::Int)
  @assert it <= h.niter
  return OptimiserHistorySlice(it,h)
end

function Base.getproperty(h::OptimiserHistory,k::Symbol)
  if haskey(h.values,k)
    return view(h.values[k],1:h.niter+1)
  else
    return getfield(h,k)
  end
end

function Base.propertynames(h::OptimiserHistory, private::Bool=false)
  (fieldnames(typeof(x))...,h.keys...)
end

function Base.setindex!(h::OptimizerHistory{T},val::T,it::Int,k::Symbol)
  @inbounds h.values[k][it+1] = val
end

function Base.setindex!(h::OptimizerHistory{T},val::Tuple,it::Int,k::Symbol)
  kk = h.bundles[k]
  setindex!(h,NamedTuple(zip(kk,val)),it)
end

function Base.setindex!(h::OptimizerHistory,vals::NamedTuple,it::Int)
  for (k,v) in vals
    setindex!(h,v,it,k)
  end
end

function Base.push!(h::OptimizerHistory,vals::NamedTuple)
  h.niter += 1
  setindex!(h,vals,h.niter)
  if get_verbose_level(h) > SOLVER_VERBOSE_LOW
    display(h[length(h)])
  end
end

function Base.push!(h::OptimiserHistory,keys::Tuple,vals::Tuple)
  @assert length(vals) == length(keys)
  push!(h,NamedTuple(zip(keys,vals)))
end

Base.push!(h::OptimiserHistory,vals::Tuple) = push!(h,h.keys,vals)

get_last_iteration(h::OptimizerHistory) = h.niter

function get_verbose_level(h::OptimizerHistory)
  return h.verbose
end

function Base.display(h::OptimizerHistory)
  for s in h
    display(s)
  end
end

function print_msg(h::OptimizerHistory,msg::String;kwargs...)
  if get_verbose_level(h) > SOLVER_VERBOSE_NONE
    printstyled(msg;kwargs...)
  end
end

include("./AugmentedLagrangian.jl")
include("./HilbertianProjection.jl")