## Abstract methods
function simplexify!(::PolytopeType,cache,graph,vertices)
  @abstractmethod
end

function allocate_simplexify_cache(::PolytopeType,parent_graph,parent_verts,ls_maps)
  @abstractmethod
end

## General methods
# Simplexify leaves of tree
function simplexify_leaves!(p::PolytopeType,data,scache,tree::TouchedBinaryNode)
  function simplexify_leaf!(data,node)
    _graph, _vertices, inout, counters, normal = node.data # data to simplexify
    graph = @views _graph[1:counters[1]]
    vertices = @views _vertices[1:counters[1]]
    simplexify!(p,scache,graph,vertices)
    _copy_simplex_cache_to_dest!(p,data,scache,vertices,normal,inout)
  end

  map_leaves!(simplexify_leaf!,data,tree)
  nothing
end

# Allocate cache for simplexified polytope data
function allocate_simplexify_data(
  ::PolytopeType{Dc,Dp},
  parent_graph::Vector{Vector{Ti}},
  parent_verts::Vector{VectorValue{Dp,Tp}},
  ls_maps
) where {Dc,Dp,Ti,Tp}
  _T = _get_map_eltype(ls_maps)
  N = length(ls_maps)
  num_verts = length(parent_verts)
  simplex_counter = zeros(Ti,1)
  if Dc == 3
    num_possible_simplices = 2 * num_verts * 2^N
  elseif Dc == Dp  # Cell: single tree of depth N
    num_possible_simplices = max(1, num_verts - 4 + 2 * 2^N)
  else             # Facet: N trees each of depth N-1, accumulated
    num_possible_simplices = N * max(1, num_verts - 4 + 2^N)
  end
  return [zeros(VectorValue{Dp,_T}, Dc + 1) for _ in 1:num_possible_simplices], # vertices
         [zero(VectorValue{Dp,_T}) for _ in 1:num_possible_simplices],          # normal
         [zeros(Int8, N) for _ in 1:num_possible_simplices],                    # inout
         simplex_counter
end

# Copy data from simplexifier cache to data cache
function _copy_simplex_cache_to_dest!(::PolytopeType,dest,cache,parent_vertices,parent_normal,parent_inout)
  vertices, normal, inout, simplex_i = dest
  T = first(cache); c = last(cache)
  j = first(simplex_i)
  for i in Base.OneTo(first(c))
    Ti = T[i]
    for k in eachindex(Ti)
      vertices[j+i][k] = parent_vertices[Ti[k]]
    end
    normal[j+i] = first(parent_normal)
    copyto!(inout[j+i],parent_inout)
  end
  simplex_i[1] += first(c)
  nothing
end

### Geometric-specific methods
## Polyhedra
function simplexify!(::PolytopeType{3},cache,graph,vertices)
  T, vstart, stack, istouch, counters = cache
  # fill!(T,zero(eltype(T)))
  fill!(vstart,zero(eltype(vstart)))
  fill!(stack,zero(eltype(stack)))
  fill!(counters,zero(eltype(counters)))
  for i in eachindex(istouch)
    fill!(istouch[i],false)
  end
  for v in 1:length(vertices)
    _isactive(graph,v) || continue
    vstart[v] == UNSET || continue
    vstart[v] = v
    num_stack = 1
    stack[num_stack] = v
    while num_stack > 0
      vcurrent = stack[num_stack]
      num_stack -= 1
      for vneig in graph[vcurrent]
        if vstart[vneig] == UNSET
          vstart[vneig] = v
          num_stack += 1
          stack[num_stack] = vneig
        end
      end
    end
  end
  num_simplices = 0
  for v in 1:length(vertices)
    _isactive(graph,v) || continue
    for i in 1:length(graph[v])
      !istouch[v][i] || continue
      istouch[v][i] = true
      vcurrent = v
      vnext = graph[v][i]
      while vnext != v
        inext = findfirst( isequal(vcurrent), graph[vnext] )
        !isnothing(inext) || break
        inext = ( inext % length( graph[vnext] ) ) + 1
        istouch[vnext][inext] = true
        vcurrent = vnext
        vnext = graph[vnext][inext]
        @assert vcurrent ≠ vnext
        vcurrent ≠ vnext || break
        if v != vstart[v] && v != vcurrent && v != vnext
          num_simplices += 1
          T[num_simplices][1] = vstart[v]
          T[num_simplices][2] = v
          T[num_simplices][3] = vcurrent
          T[num_simplices][4] = vnext
        end
      end
    end
  end
  counters[1] = num_simplices
  nothing
end

# Allocate simplexify cache
function allocate_simplexify_cache(
  ::PolytopeType{3,3},
  parent_graph::Vector{Vector{Ti}},
  parent_verts::Vector{VectorValue{3,Tp}},
  ls_maps
) where {Ti<:Integer,Tp}
  num_verts = length(parent_verts)
  T = [zeros(Ti,4) for _ in 1:4*num_verts]
  vstart = fill(UNSET,4*num_verts)
  stack = zeros(Ti,4*num_verts)
  istouch = map( i -> falses(length(first(parent_graph))), 1:4*num_verts )
  counters = zeros(Ti,1) # Num simplicies
  return T, vstart, stack, istouch, counters
end

## Polygon
function simplexify!(::PolytopeType{2},cache,graph,vertices)
  _T, _e_to_v, counters = cache
  fill!(counters,zero(eltype(counters)))
  for i in eachindex(_T)
    fill!(_T[i],zero(eltype(_T[i])))
  end
  for i in eachindex(_e_to_v)
    fill!(_e_to_v[i],zero(eltype(_e_to_v[i])))
  end
  T = @views _T[1:length(graph)]
  e_to_v = @views _e_to_v[1:length(graph)]
  generate_facet_to_vertices!(e_to_v,graph)
  if length(e_to_v) > 0
    v0 = e_to_v[1][1]
    num_simplices = 0
    for verts in e_to_v
      if v0 ∉ verts
        num_simplices += 1
        T[num_simplices][1] = v0
        T[num_simplices][2] = verts[1]
        T[num_simplices][3] = verts[2]
      end
    end
    counters[1] = num_simplices
  end
  nothing
end

function generate_facet_to_vertices!(T,graph)
  n = length(graph)
  for v in eachindex(graph)
    vnext = ifelse(v == n, 1, v+1)
    @check vnext ∈ graph[v]
    T[v][1] = v
    T[v][2] = vnext
  end
  nothing
end

# Allocate simplexify cache
function allocate_simplexify_cache(
  ::PolytopeType{2,Dp},
  parent_graph::Vector{Vector{Ti}},
  parent_verts::Vector{VectorValue{Dp,Tp}},
  ls_maps
) where {Dp,Ti<:Integer,Tp}
  num_verts = length(parent_verts)
  T = [zeros(Ti,3) for _ in 1:3*num_verts]
  e_to_v = [zeros(Ti,2) for _ in 1:3*num_verts]
  counters = zeros(Ti,1)
  return T, e_to_v, counters
end

## Line
function simplexify!(::PolytopeType{1},cache,graph,vertices)
  nothing
end

function allocate_simplexify_cache(
  ::PolytopeType{1,Dp},
  parent_graph::Vector{Vector{Ti}},
  parent_verts::Vector{VectorValue{Dp,Tp}},
  ls_maps
) where {Dp,Ti<:Integer,Tp}
  return nothing
end

function _copy_simplex_cache_to_dest!(::PolytopeType{1},dest,cache,parent_vertices,parent_normal,parent_inout)
  vertices, normal, inout, simplex_i = dest
  j = simplex_i[1] += 1
  copyto!(vertices[j], parent_vertices)
  normal[j] = first(parent_normal)
  copyto!(inout[j], parent_inout)
  nothing
end