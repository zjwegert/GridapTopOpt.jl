"""
  abstract type Optimiser end

  Your own optimiser can be implemented by implementing 
    concrete functionality of the below.
"""
abstract type Optimiser end

# TODO: Add quality of life Base.show for each optimiser which displays extra info
function Base.show(io::IO,object::Optimiser)
  print(io,typeof(object))
end

Base.IteratorEltype(::Type{<:Optimiser}) = Base.EltypeUnknown()
Base.IteratorSize(::Type{<:Optimiser}) = Base.SizeUnknown()

# Return tuple of first iteration state
function Base.iterate(::Optimiser)
  @abstractmethod
end

# Return tuple of next iteration state given current state
function Base.iterate(::Optimiser,state)
  @abstractmethod
end

get_history(::Optimiser) :: OptimiserHistory = @abstractmethod

function converged(::Optimiser)
  @abstractmethod
end

function finished(m::Optimiser) :: Bool
  h = get_history(m)
  it = get_last_iteration(h)

  A = converged(m)
  B = (it >= h.maxiter)
  return A || B
end

function print_msg(opt::Optimiser,msg::String;kwargs...) 
  print_msg(get_optimiser_history(opt),msg;kwargs...)
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
  maxiter = 200,
  verbose = SOLVER_VERBOSE_NONE
)
  values = Dict{Symbol,Vector{T}}()
  for k in keys
    values[k] = zeros(T,maxiter+1)
  end
  verbose = SolverVerboseLevel(verbose)
  return OptimiserHistory{T}(-1,keys,values,bundles,verbose,maxiter)
end

function reset!(h::OptimiserHistory)
  h.niter = -1
  for k in keys(h.values)
    fill!(h.values[k],0)
  end
end

Base.keys(h::OptimiserHistory) = h.keys
Base.length(h::OptimiserHistory) = h.niter

Base.iterate(h::OptimiserHistory) = iterate(h,-1)
Base.iterate(h::OptimiserHistory,it::Int) = (it < h.niter) ? (h[it+1],it+1) : nothing

function Base.getindex(h::OptimiserHistory,k::Symbol,it::Union{<:Integer,<:UnitRange{<:Integer}})
  if haskey(h.values,k)
    return h.values[k][it.+1]
  elseif haskey(h.bundles,k)
    return Tuple([h.values[ki][it.+1] for ki in h.bundles[k]])
  else
    @error "Key $(k) not found in OptimiserHistory"
  end
end

function Base.getindex(h::OptimiserHistory,k::Symbol)
  return getindex(h,k,0:h.niter)
end

function Base.getindex(h::OptimiserHistory,it::Int)
  @assert it <= h.niter
  return OptimiserHistorySlice(it,h)
end

function Base.setindex!(h::OptimiserHistory{T},val::T,k::Symbol,it::Integer) where T
  @inbounds h.values[k][it+1] = val
end

function Base.setindex!(h::OptimiserHistory{T},vals::Tuple,k::Symbol,it::Integer) where T
  kk = h.bundles[k]
  setindex!(h,kk,vals,it)
end

function Base.setindex!(h::OptimiserHistory,vals::Tuple,keys::Tuple,it::Integer)
  for (k,v) in zip(keys,vals)
    setindex!(h,v,k,it)
  end
end

function Base.push!(h::OptimiserHistory,vals,keys)
  @assert length(vals) == length(keys)
  h.niter += 1
  setindex!(h,vals,keys,h.niter)
  if get_verbose_level(h) > SOLVER_VERBOSE_NONE
    display(h[length(h)])
  end
end

Base.push!(h::OptimiserHistory,vals::NamedTuple) = push!(h,values(vals),keys(vals))
Base.push!(h::OptimiserHistory,vals::Tuple) = push!(h,vals,Tuple(h.keys))

get_last_iteration(h::OptimiserHistory) = h.niter

function get_verbose_level(h::OptimiserHistory)
  return h.verbose
end

function Base.display(h::OptimiserHistory)
  println("OptimiserHistory with $(h.niter+1) iterations")
  for s in h
    display(s)
  end
end

function Base.write(io::IO,h::OptimiserHistory)
  content = join([k for k in keys(first(h))],", ")
  for s in h
    content *= "\n"*join([@sprintf("%.4e",s[k]) for k in keys(s)],", ")
  end
  Base.write(io,content)
end

function write_history(path::String,h::OptimiserHistory)
  open(path,"w") do f
    write(f,h)
  end  
end

function print_msg(h::OptimiserHistory,msg::String;kwargs...)
  if get_verbose_level(h) > SOLVER_VERBOSE_NONE
    printstyled(msg;kwargs...)
  end
end

# Optimiser history slice

struct OptimiserHistorySlice{T}
  it :: Int
  h  :: OptimiserHistory{T}
end

Base.keys(s::OptimiserHistorySlice) = keys(s.h)

function Base.getindex(s::OptimiserHistorySlice,k::Symbol)
  return getindex(s.h,k,s.it)
end

function Base.setindex!(::OptimiserHistorySlice,args...)
  @error "OptimiserHistorySlice is read-only!"
end

function Base.display(s::OptimiserHistorySlice)
  content = join([string(k,"=",@sprintf("%.4e",s[k])) for k in keys(s)],", ")
  println("Iteration: $(@sprintf("%3i",s.it)) | $(content)")
end

include("OrthogonalisationMaps.jl")
include("AugmentedLagrangian.jl")
include("HilbertianProjection.jl")