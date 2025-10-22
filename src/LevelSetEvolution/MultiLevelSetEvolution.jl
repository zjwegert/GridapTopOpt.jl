struct SeparateCache end
struct ReuseCache end

"""
    struct MultiLevelSetEvolution{A,B,C} <: AbstractLevelSetEvolution

A wrapper to hold a level-set evolver and reinitialiser for problems with
multiple level-set functions.
"""
struct MultiLevelSetEvolution{
    A<:Union{SeparateCache,ReuseCache},
    B<:Vector{<:Evolver},
    C<:Vector{<:Reinitialiser},
    D<:MultiFieldSpaceTypes
} <: AbstractLevelSetEvolution
  evolver       :: B
  reinitialiser :: C
  mf_space      :: D
  @doc"""
      MultiLevelSetEvolution(evolvers,reinitialisers,mf_space;reuse_cache=true)

  Create an instance of `MultiLevelSetEvolution` with the vector of level-set
  evolvers `evolvers`, the vector of reinitialisers `reinitialisers`, and the
  multi-field space `mf_space`.

  Optional argument `reuse_cache` indicates whether to reuse cache between
  evolvers/reinitialisers.
  """
  function MultiLevelSetEvolution(evolvers,reinitialisers,mf_space;reuse_cache=true)
    cache_type = reuse_cache ? ReuseCache() : SeparateCache()
    @check length(evolvers) == length(reinitialisers) "Number of evolvers and reinitialisers must be the same"
    A,B,C,D=typeof(cache_type),typeof(evolvers),typeof(reinitialisers),typeof(mf_space)
    new{A,B,C,D}(evolvers,reinitialisers,mf_space)
  end
end

function evolve!(s::MultiLevelSetEvolution,φ::AbstractVector,dφ::AbstractVector,args...)
  φh = FEFunction(s.mf_space,φ)
  dφh = FEFunction(s.mf_space,dφ)
  evolve!(s,φh,dφh,args...)
end

function evolve!(s::MultiLevelSetEvolution,φ::AbstractVector,dφ::AbstractVector,γ,::Nothing)
  φh = FEFunction(s.mf_space,φ)
  dφh = FEFunction(s.mf_space,dφ)
  evolve!(s,φh,dφh,γ,[nothing for _ in 1:length(φh)])
end

function evolve!(s::MultiLevelSetEvolution,φh,dφh,γ,::Nothing)
  evolve!(s,φh,dφh,γ,[nothing for _ in 1:length(φh)])
end

function evolve!(s::MultiLevelSetEvolution{SeparateCache},φh,dφh,γ)
  evolvers = get_evolver(s)
  N = length(φh)
  @check N == length(evolvers) "Number of level-set functions and evolvers must be the same"
  map(1:N) do i
    evolve!(evolvers[i],φh[i],dφh[i],γ)
  end |> tuple_of_arrays
end

function evolve!(s::MultiLevelSetEvolution{ReuseCache},φh,dφh,γ)
  evolvers = get_evolver(s)
  N = length(φh)
  φ1, cache1 = evolve!(evolvers[1],φh[1],dφh[1],γ,cache[1])
  φi, cachei = map(2:N) do i
    evolve!(evolvers[i],φh[i],dφh[i],γ,cache1)
  end |> tuple_of_arrays
  return vcat([φ1],φi), vcat([cache1],cachei)
end

function evolve!(s::MultiLevelSetEvolution,φh,dφh,γ,cache)
  evolvers = get_evolver(s)
  N = length(φh)
  map(1:N) do i
    evolve!(evolvers[i],φh[i],dφh[i],γ,cache[i])
  end |> tuple_of_arrays
end

function reinit!(s::MultiLevelSetEvolution,φ::AbstractVector,args...)
  φh = FEFunction(s.mf_space,φ)
  reinit!(s,φh,args...)
end

function reinit!(s::MultiLevelSetEvolution{SeparateCache},φh)
  reinitialisers = get_reinitialiser(s)
  N = length(φh)
  @check N == length(reinitialisers) "Number of level-set functions and reinitialisers must be the same"
  map(1:N) do i
    reinit!(reinitialisers[i],φh[i])
  end |> tuple_of_arrays
end

function reinit!(s::MultiLevelSetEvolution{ReuseCache},φh)
  reinitialisers = get_reinitialiser(s)
  N = length(φh)
  φ1, cache1 = reinit!(reinitialisers[1],φh[1])
  φi, cachei = map(2:N) do i
    reinit!(reinitialisers[i],φh[i],cache1)
  end |> tuple_of_arrays
  return vcat([φ1],φi), vcat([cache1],cachei)
end

function reinit!(s::MultiLevelSetEvolution,φh,cache)
  reinitialisers = get_reinitialiser(s)
  N = length(φh)
  map(1:N) do i
    reinit!(reinitialisers[i],φh[i],cache[i])
  end |> tuple_of_arrays
end

function get_evolver(s::MultiLevelSetEvolution)
  s.evolver
end

function get_reinitialiser(s::MultiLevelSetEvolution)
  s.reinitialiser
end

function get_min_dof_spacing(s::MultiLevelSetEvolution)
  minimum(map(get_min_dof_spacing,get_evolver(s)))
end

function get_ls_space(s::MultiLevelSetEvolution)
  s.mf_space
end