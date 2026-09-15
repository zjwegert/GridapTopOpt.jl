## Polytope Type
# Needed so that we can appropriately dispatch facets and cells of different dimensions.
# Dp = Dimensions of point
# Dc = Dimensions of cell
#
# I.e.,
# (Dc,Dp) = (2,2): Polytope is a polygon in R² (cell)
# (Dc,Dp) = (1,2): Polytope is a line in R² (facet)
# (Dc,Dp) = (3,3): Polytope is a polyhedra in R³ (cell)
# (Dc,Dp) = (2,3): Polytope is a polygon in R³ (facet)
struct PolytopeType{Dc,Dp} end

PolytopeType(Dc,Dp) = PolytopeType{Dc,Dp}()

## Abstract methods
# Splitter
function split!(::PolytopeType,cache,graph,vertices,vertex_values)
  @abstractmethod
end

function allocate_splitter_cache(::PolytopeType,parent_graph,parent_verts,ls_maps)
  @abstractmethod
end

function allocate_polytope_data(::PolytopeType,parent_graph,parent_verts,ls_maps)
  @abstractmethod
end

function copy_data_to_node!(::PolytopeType,node::TouchedBinaryNode,cached_data)
  @abstractmethod
end

## Polyhedra
function split!(::PolytopeType{3},cache,graph,vertices,vertex_values)
  @check !any(iszero,vertex_values)
  interpolate_values(v1,v2,w1,w2) = v1 + w1/(w1+w2)*(v2-v1)
  is_in(v) = v < 0

  all(is_in, vertex_values) && return :in
  all(!is_in, vertex_values) && return :out

  # Here we switch the ordering as this gets switched in post-processing
  out_old_to_new, out_new_to_old, out_graph, in_old_to_new, in_new_to_old, in_graph, _vertices, counters = cache

  # Reset caches
  fill!(out_old_to_new,UNSET)
  fill!(out_new_to_old,UNSET)
  fill!(in_old_to_new,UNSET)
  fill!(in_new_to_old,UNSET)
  for i in eachindex(out_graph)
    fill!(out_graph[i],zero(eltype(out_graph[i])))
    fill!(in_graph[i],zero(eltype(in_graph[i])))
  end
  for i in eachindex(vertices)
    _vertices[i] = vertices[i]
    in_graph[i] .= graph[i]
    out_graph[i] .= graph[i]
  end

  num_vertices = length(vertices)
  for v in eachindex(graph)
    _isactive(graph,v) || continue
    vertex_values[v] < 0 && continue
    for (i,vneig) in enumerate(graph[v])
      w, wneig = vertex_values[v], vertex_values[vneig]
      w*wneig > 0 && continue

      vertex = interpolate_values(_vertices[v],_vertices[vneig],abs(w),abs(wneig))
      num_vertices += 1
      _vertices[num_vertices] = vertex

      in_graph[v][i] = num_vertices
      in_graph[num_vertices][1] = v
      in_graph[num_vertices][2:end] .= UNSET

      ineig = findfirst(isequal(v), graph[vneig])
      out_graph[vneig][ineig] = num_vertices
      out_graph[num_vertices][1] = vneig
      out_graph[num_vertices][2:end] .= UNSET
    end
  end

  _split_postprocess!(@views(out_old_to_new[1:num_vertices]),@views(out_new_to_old[1:num_vertices]),
    @views(out_graph[1:num_vertices]),length(vertices),vertex_values,(>))
  _split_postprocess!(@views(in_old_to_new[1:num_vertices]),@views(in_new_to_old[1:num_vertices]),
    @views(in_graph[1:num_vertices]),length(vertices),vertex_values,(<))

  num_in = send_zeros_to_back!(in_graph)
  num_out = send_zeros_to_back!(out_graph)

  counters[1] = num_out
  counters[2] = num_in
  counters[3] = num_vertices

  return :both
end

_isactive(graph,vertex) = !isempty( graph[vertex] )

_f(i, old_to_new) = ifelse(i ∈ (OPEN,UNSET), i, @views old_to_new[i])

function _compact!(old_to_new,new_to_old,graph)
  n_old = length(graph)
  fill!(old_to_new,UNSET)
  fill!(new_to_old,UNSET)
  new = 0
  for old in 1:n_old
    iszero(graph[old]) && continue
    new += 1
    old_to_new[old] = new
  end
  nb = maximum(old_to_new)
  find_inverse_index_map!(@views(new_to_old[1:nb]), old_to_new)

  for v in 1:n_old
    if v ∈ @views(new_to_old[1:nb])
      map!(Base.Fix2(_f,old_to_new),graph[v],graph[v])
    end
  end

  return new_to_old
end

function next_vertex(edge_graph::AbstractVector,num_vertices::Integer,vstart::Integer)
  vcurrent = vstart
  vnext = first( edge_graph[vcurrent] )
  while vnext ≤ num_vertices && vnext ∉ (UNSET,OPEN)
    i = findfirst( isequal(vcurrent), edge_graph[vnext] )
    vcurrent = vnext
    inext = ( i % length( edge_graph[vcurrent] ) ) + 1
    vnext = edge_graph[vcurrent][ inext ]
  end
  vnext
end

# Some of the below are adapted from STLCutters
function _split_postprocess!(old_to_new,new_to_old,graph,n_verts,values,(≶))
  _complete_graph!(graph,n_verts)
  _disconnect_graph!(graph,n_verts,values,(≶))
  _compact!(old_to_new,new_to_old,graph)
  nothing
end

function _complete_graph!(edge_graph,num_vertices::Integer)
  for v in num_vertices+1:length(edge_graph)
    vnext = next_vertex(edge_graph,num_vertices,v)
    vnext ∉ (UNSET,OPEN) || continue
    edge_graph[v][end] = vnext
    edge_graph[vnext][2] = v
  end
end

function _disconnect_graph!(edge_graph::AbstractVector{<:AbstractVector{T}},num_vertices,distances,(≶)::Function) where T<:Integer
  for i in 1:num_vertices
    if distances[i] ≶ 0
      for j = 1:length(edge_graph[i])
        edge_graph[i][j] = zero(T)
      end
    end
  end
  nothing
end

function send_zeros_to_back!(graph::Vector{Vector{T}}) where T<:Integer
  count = 0
  for i in eachindex(graph)
    if !iszero(graph[i])
      count += 1
      copyto!(graph[count],graph[i])
    end
  end
  nz = count
  while count < length(graph)
    count += 1
    fill!(graph[count], 0)
  end
  return nz
end

function send_zeros_to_back!(v::Vector{T}) where T<:Integer
  count = 0
  for i in eachindex(v)
    if !iszero(v[i])
      count += 1
      v[count] = v[i]
    end
  end
  nz = count
  while count < length(v)
    count += 1
    v[count] = 0
  end
  return nz
end

# Allocate splitter cache
function allocate_splitter_cache(
  ::PolytopeType{3,3},
  parent_graph::Vector{Vector{Ti}},
  parent_verts::Vector{VectorValue{3,Tp}},
  ls_maps
) where {Ti<:Integer,Tp}
  _T = _get_map_eltype(ls_maps)
  num_verts = length(parent_verts)
  out_old_to_new = zeros(Ti, 4num_verts)
  out_new_to_old = zeros(Ti, 4num_verts)
  out_graph = [zeros(Ti,3) for _ in 1:4num_verts]
  in_old_to_new = zeros(Ti, 4num_verts)
  in_new_to_old = zeros(Ti, 4num_verts)
  in_graph = [zeros(Ti,3) for _ in 1:4num_verts]
  vertices = zeros(VectorValue{3,_T}, 4num_verts)
  counters = zeros(Ti,3)
  return (;out_old_to_new, out_new_to_old, out_graph,
    in_old_to_new, in_new_to_old, in_graph, vertices, counters)
end

# Allocate tree-data cache
function allocate_polytope_data(
  ::PolytopeType{3,3},
  parent_graph::Vector{Vector{Ti}},
  parent_verts::Vector{VectorValue{3,Tp}},
  ls_maps
) where {Ti<:Integer,Tp}
  _T = _get_map_eltype(ls_maps)
  N = length(ls_maps)
  num_verts = length(parent_verts)
  graph = [zeros(Ti,3) for _ in 1:4num_verts]
  vertices = zeros(VectorValue{3,_T}, 4num_verts)
  inout = zeros(Int8, N)
  counters = zeros(Ti,2) # num_vertices, num_inout
  normal= zeros(VectorValue{3,_T}, 1) # added here to unify cache, does nothing. Minor overhead
  return (;graph, vertices, inout, counters, normal)
end

# Copy data from splitter cache to tree-data cache
function copy_data_to_node!(::PolytopeType{3},node::TouchedBinaryNode,cached_data)
  # Src
  _, in_new_to_old, in_graph, _, out_new_to_old, out_graph, vertices, counters = cached_data
  _, _, sinout, scounters = node.data
  sinout_i = last(scounters)
  # Dst (IN)
  dgraph, dvertices, dinout, dcounters = node.left.data
  fill!(dcounters,zero(eltype(dcounters)))
  copy_polyhedra_data_to_node(
    dgraph, dvertices, dinout, dcounters,
    :in, in_new_to_old, in_graph, vertices, counters,
    sinout, sinout_i
  )
  node.left.touched = true
  # Dst (OUT)
  dgraph, dvertices, dinout, dcounters = node.right.data
  fill!(dcounters,zero(eltype(dcounters)))
  copy_polyhedra_data_to_node(
    dgraph, dvertices, dinout, dcounters,
    :out, out_new_to_old, out_graph, vertices, counters,
    sinout, sinout_i
  )
  node.right.touched = true
  nothing
end

function copy_polyhedra_data_to_node(
  dgraph, dvertices, dinout, dcounters,         # Dst
  sstate, snodes, sgraph, svertices, scounters, # Src split data
  sinout, sinout_i                              # Src inout
)
  # Update counters
  snum_in, snum_out, snum_vertices = scounters
  snum_nodes = ifelse(sstate == :in, snum_in, snum_out)
  nv_i = dcounters[1] = snum_nodes
  # Update IN/OUT
  copyto!(dinout, sinout)
  io_i = dcounters[2] = sinout_i + 1
  dinout[io_i] = ifelse(sstate == :in, IN, OUT)
  # Update graph
  for i in 1:nv_i
    dgraph[i] .= sgraph[i]
  end
  # Update vertices
  for (i,v) in enumerate(@views snodes[1:nv_i])
    dvertices[i] = @views svertices[1:snum_vertices][v]
  end
  nothing
end

## Polygon
function split!(::PolytopeType{2},cache,graph,vertices,vertex_values)
  @check !any(iszero,vertex_values)
  interpolate_values(v1,v2,w1,w2) = v1 + (w1/(w1+w2))*(v2-v1)
  is_in(v) = v < 0

  all(is_in, vertex_values) && return :in
  all(!is_in, vertex_values) && return :out

  in_nodes, in_graph, out_nodes, out_graph, _vertices, counters = cache

  # Reset caches
  fill!(in_nodes,zero(eltype(in_nodes)))
  fill!(out_nodes,zero(eltype(out_nodes)))
  for i in eachindex(out_graph)
    fill!(out_graph[i],zero(eltype(out_graph[i])))
    fill!(in_graph[i],zero(eltype(in_graph[i])))
  end
  for i in eachindex(vertices)
    _vertices[i] = vertices[i]
  end

  # Loop over graph and find intersections
  start = 1
  current, next = start, first(graph[start])
  num_in = 0
  num_out = 0
  num_vertices = length(vertices)
  while true
    v_current, v_next = _vertices[current], _vertices[next]
    w_current, w_next = vertex_values[current], vertex_values[next]

    if is_in(w_current)
      num_in += 1
      in_nodes[num_in] = current
    else
      num_out += 1
      out_nodes[num_out] = current
    end

    if w_current*w_next < 0
      vertex = interpolate_values(v_current,v_next,abs(w_current),abs(w_next))

      num_vertices += 1
      _vertices[num_vertices] = vertex
      num_in += 1; num_out += 1;
      in_nodes[num_in] = num_vertices
      out_nodes[num_out] = num_vertices
    end

    current, next = next, first(graph[next])
    isequal(current,start) && break
  end
  compute_graph!(in_graph,num_in)
  compute_graph!(out_graph,num_out)
  # Update counters
  counters[1] = num_in
  counters[2] = num_out
  counters[3] = num_vertices
  return :both
end

@inline function compute_graph!(graph::Vector{Vector{Int32}},num)
  for i in Base.OneTo(num)
    inext = i == num ? 1 : i+1
    iprev = i == 1 ? num : i-1
    @views graph[i][1] = iprev
    @views graph[i][2] = inext
  end
end

# Allocate splitter cache
function allocate_splitter_cache(
  ::PolytopeType{2},
  parent_graph::Vector{Vector{Ti}},
  parent_verts::Vector{VectorValue{D,Tp}},
  ls_maps
) where {D,Ti<:Integer,Tp}
  _T = _get_map_eltype(ls_maps)
  num_verts = length(parent_verts)
  in_nodes = zeros(Ti, 3*num_verts)
  in_graph = [zeros(Ti,2) for _ in 1:3*num_verts]
  out_nodes = zeros(Ti, 3*num_verts)
  out_graph = [zeros(Ti,2) for _ in 1:3*num_verts]
  vertices = zeros(VectorValue{D,_T}, 3*num_verts)
  counters = zeros(Ti,3) #num_in, num_out num_vertices
  return (;in_nodes, in_graph, out_nodes, out_graph, vertices, counters)
end

# Allocate tree-data cache
function allocate_polytope_data(
  ::PolytopeType{2},
  parent_graph::Vector{Vector{Ti}},
  parent_verts::Vector{VectorValue{D,Tp}},
  ls_maps
) where {D,Ti<:Integer,Tp}
  _T = _get_map_eltype(ls_maps)
  N = length(ls_maps)
  num_verts = length(parent_verts)
  graph = [zeros(Ti,2) for _ in 1:3*num_verts]
  vertices = zeros(VectorValue{D,_T}, 3*num_verts)
  inout = zeros(Int8, N)
  counters = zeros(Ti,2) # num_vertices, num_inout
  normal = zeros(VectorValue{D,_T}, 1) # This is only used for facets
  return (;graph, vertices, inout, counters, normal)
end

# Copy data from splitter cache to tree-data cache
function copy_data_to_node!(::PolytopeType{2},node::TouchedBinaryNode,cached_data)
  # Src
  in_nodes, in_graph, out_nodes, out_graph, vertices, counters = cached_data
  _, _, sinout, scounters = node.data
  sinout_i = last(scounters)
  # Dst (IN)
  dgraph, dvertices, dinout, dcounters = node.left.data
  fill!(dcounters,zero(eltype(dcounters)))
  copy_polyhedra_data_to_node( # Reuse method for polyhedra
    dgraph, dvertices, dinout, dcounters,
    :in, in_nodes, in_graph, vertices, counters,
    sinout, sinout_i
  )
  node.left.touched = true
  # Dst (OUT)
  dgraph, dvertices, dinout, dcounters = node.right.data
  fill!(dcounters,zero(eltype(dcounters)))
  copy_polyhedra_data_to_node(
    dgraph, dvertices, dinout, dcounters,
    :out, out_nodes, out_graph, vertices, counters,
    sinout, sinout_i
  )
  node.right.touched = true
  nothing
end

## Line
function split!(::PolytopeType{1},cache,graph,vertices,vertex_values)
  @check length(vertices) == length(vertex_values) == 2
  @check length(cache[end]) == 3
  @check !any(iszero,vertex_values)
  interpolate_values(v1,v2,w1,w2) = v1 + (w1/(w1+w2))*(v2-v1)
  is_in(v) = v < 0

  all(is_in, vertex_values) && return :in
  all(!is_in, vertex_values) && return :out

  in_nodes, out_nodes, _vertices = cache
  for i in eachindex(vertices)
    _vertices[i] = vertices[i]
  end

  vertex = interpolate_values(vertices[1],vertices[2],abs(vertex_values[1]),abs(vertex_values[2]))
  _vertices[end] = vertex

  in_nodes[1] = ifelse(is_in(vertex_values[1]),1,3)
  in_nodes[2] = ifelse(is_in(vertex_values[1]),3,2)
  out_nodes[1] = ifelse(is_in(vertex_values[1]),3,1)
  out_nodes[2] = ifelse(is_in(vertex_values[1]),2,3)

  return :both
end

# Allocate splitter cache
function allocate_splitter_cache(
  ::PolytopeType{1,Dp},
  parent_graph::Vector{Vector{Ti}},
  parent_verts::Vector{VectorValue{Dp,Tp}},
  ls_maps
) where {Dp,Ti<:Integer,Tp}
  _T = _get_map_eltype(ls_maps)
  in_nodes = zeros(Ti,2)
  out_nodes = zeros(Ti,2)
  vertices = zeros(VectorValue{Dp,_T},3)
  return (;in_nodes, out_nodes, vertices)
end

# Allocate tree-data cache
function allocate_polytope_data(
  ::PolytopeType{1,Dp},
  parent_graph::Vector{Vector{Ti}},
  parent_verts::Vector{VectorValue{Dp,Tp}},
  ls_maps
) where {Dp,Ti<:Integer,Tp}
  graph = [[Ti(1)],[Ti(2)]]
  _T = _get_map_eltype(ls_maps)
  N = length(ls_maps)
  vertices = zeros(VectorValue{Dp,_T}, 2)
  inoutcut = zeros(Int8, N)
  counters = zeros(Ti, 2) # num_vertices, inout_count
  normal = zeros(VectorValue{Dp,_T}, 1) # This is only used for facets
  return (;graph, vertices, inoutcut, counters, normal)
end

# Copy data from splitter cache to tree-data cache
function copy_data_to_node!(::PolytopeType{1},node::TouchedBinaryNode,cached_data)
  # Src
  in_nodes, out_nodes, vertices = cached_data
  _, _, sinout, scounters = node.data
  sinout_i = last(scounters)
  # Dst (IN)
  _, dvertices, dinout, dcounters = node.left.data
  fill!(dcounters,zero(eltype(dcounters)))
  copy_line_data_to_node!(dvertices, dinout, dcounters, :in , in_nodes, vertices, sinout, sinout_i)
  node.left.touched = true
  # Dst (OUT)
  _, dvertices, dinout, dcounters = node.right.data
  fill!(dcounters,zero(eltype(dcounters)))
  copy_line_data_to_node!(dvertices, dinout, dcounters,:out , out_nodes, vertices, sinout, sinout_i)
  node.right.touched = true
  nothing
end

function copy_line_data_to_node!(
  dvertices, dinout, dcounters, # Dst
  sstate, snodes, svertices,    # Src split data
  sinout, sinout_i              # Src inout
)
  dcounters[1] = length(snodes)
  # Update IN/OUT
  copyto!(dinout, sinout)
  io_i = dcounters[2] = sinout_i + 1
  dinout[io_i] = ifelse(sstate == :in, IN, OUT)
  # Update vertices
  for (i,v) in enumerate(snodes)
    dvertices[i] = @views svertices[v]
  end
  nothing
end