
struct MergeMap{T,A} <: Map
  i_to_values::A
  i_to_nvals::Vector{Int16}
  function MergeMap(i_to_values::AbstractVector{TV}) where TV
    i_to_nvals = collect(Int16,lazy_map(length,i_to_values))
    T = ifelse(TV <: Array, eltype(TV), TV)
    A = typeof(i_to_values)
    new{T,A}(i_to_values,i_to_nvals)
  end
end

Arrays.return_type(::MergeMap{T}) where T = Vector{T}

function Arrays.return_cache(k::MergeMap{T},I) where T
  s = sum((k.i_to_nvals[i] for i in I))
  r = CachedArray(zeros(T,s))
  c = array_cache(k.i_to_values)
  return r, c
end

function Arrays.evaluate!(cache,k::MergeMap{T},I) where T
  cr, c = cache
  s = sum((k.i_to_nvals[i] for i in I))
  setsize!(r,s)
  r = cr.array

  o = 1
  for i in I
    vals = getindex!(c,k.i_to_values,i)
    for v in vals
      r[o] = v
      o += 1
    end
  end

  r
end
