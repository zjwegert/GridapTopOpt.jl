# Cutter map
abstract type PolytopalCutterMap{D,N,Nc} <: Map end

### Cell cutter map

struct PolytopalCellCutterMap{D,N,Nc} <: PolytopalCutterMap{D,N,Nc} end

PolytopalCellCutterMap(D,N,Nc) = PolytopalCellCutterMap{D,N,Nc}();

function Arrays.return_cache(
  ::PolytopalCellCutterMap{D,A},
  pvertices::Vector{VectorValue{D,Tp}},
  pgraph::Vector{Vector{Ti}},
  cell_ref_to_phys_map::Field,
  ls_maps::Field...
) where {D,A,Tp,Ti}
  _T = _get_map_eltype(ls_maps)
  p2φs_cache = map(Base.Fix2(return_cache,zero(Point{D,_T})),ls_maps)
  r2p_cache = return_cache(cell_ref_to_phys_map,zero(Point{D,_T}))
  φ_val_cache = zeros(_T,2*length(pvertices))
  phys_vert_cache = zeros(Point{D,_T},2*length(pvertices))
  ## Cell caches
  cell_data = allocate_polytope_data(PolytopeType(D,D),pgraph,pvertices,ls_maps);
  cell_tree = preallocate_tree(cell_data,A);
  # splitter cache
  cell_splitter_cache = allocate_splitter_cache(PolytopeType(D,D),pgraph,pvertices,ls_maps);
  # simplexify data/cache
  simplexified_cell_data = allocate_simplexify_data(PolytopeType(D,D),pgraph,pvertices,ls_maps);
  simplexify_cell_cache = allocate_simplexify_cache(PolytopeType(D,D),pgraph,pvertices,ls_maps);
  ## Facet cache
  facet_data = allocate_polytope_data(PolytopeType(D-1,D),pgraph,pvertices,ls_maps);
  facet_tree = preallocate_tree(facet_data,A-1);
  # splitter cache
  facet_splitter_cache = allocate_splitter_cache(PolytopeType(D-1,D),pgraph,pvertices,ls_maps);
  # simplexify data/cache
  simplexified_facet_data = allocate_simplexify_data(PolytopeType(D-1,D),pgraph,pvertices,ls_maps)
  simplexify_facet_cache = allocate_simplexify_cache(PolytopeType(D-1,D),pgraph,pvertices,ls_maps)
  # Return caches
  ls_map_caches = (p2φs_cache, φ_val_cache)
  ref_to_phys_map_caches = (r2p_cache, phys_vert_cache)
  cell_caches = (cell_tree, cell_splitter_cache, simplexify_cell_cache)
  facet_caches = (facet_tree, facet_splitter_cache, simplexify_facet_cache)
  return simplexified_cell_data, simplexified_facet_data, cell_caches, facet_caches, ls_map_caches, ref_to_phys_map_caches
end

function _reset!(::PolytopalCellCutterMap{D},cache) where D
  simplexified_cell_data, simplexified_facet_data, cell_caches, facet_caches, _, _ = cache
  cell_tree, _, _ = cell_caches
  facet_tree, _, _ = facet_caches
  # Reset tree data, we don't fill with zero as we reset caches inside the splitting/simplexifying functions
  #  where needed
  reset_touched!(cell_tree)
  reset_touched!(facet_tree)
  # Reset cell/facet counters
  simplexified_cell_data[end][1] = 0
  simplexified_facet_data[end][1] = 0
  nothing
end

# No cells cut, return cache
function Arrays.evaluate!(
  cache,
  k::PolytopalCellCutterMap{D,N,0},
  pvertices::Vector{VectorValue{D,Tp}},
  pgraph::Vector{Vector{Ti}},
  cell_ref_to_phys_map::Field,
  ls_maps::Field...
) where {D,N,Tp,Ti}
  return cache
end

function Arrays.evaluate!(
  cache,
  k::PolytopalCellCutterMap{D,A},
  pvertices::Vector{VectorValue{D,Tp}},
  pgraph::Vector{Vector{Ti}},
  cell_ref_to_phys_map::Field,
  ls_maps::Field...
) where {D,A,Tp,Ti}
  # Reset cache
  _reset!(k,cache)
  # Initialise cell tree
  simplexified_cell_data, simplexified_facet_data, cell_caches, facet_caches, ls_map_caches, ref_to_phys_map_caches = cache
  cell_tree, _, _ = cell_caches
  copy_polytope_data_to_cell_tree!(cell_tree,pvertices,pgraph)
  # Cells
  recursively_cut_and_simplexify_cells!(k,simplexified_cell_data,ls_maps,cell_caches,ls_map_caches)
  # Facets
  recursively_cut_and_simplexify_facet!(k,simplexified_facet_data,ls_maps,cell_ref_to_phys_map,
    cell_caches,facet_caches,ls_map_caches,ref_to_phys_map_caches)
  return cache
end

function get_cell_rvertices(cache)
  simplexified_cell_data, _, _, _, _, _ = cache
  num_cells = simplexified_cell_data[end][1]
  return @views simplexified_cell_data[1][1:num_cells]
end

function get_subcell_inout(cache)
  simplexified_cell_data, _, _, _, _, _ = cache
  num_cells = simplexified_cell_data[end][1]
  return @views simplexified_cell_data[3][1:num_cells]
end

function get_facet_rvertices(cache)
  _, simplexified_facet_data, _, _, _, _ = cache
  num_facets = simplexified_facet_data[end][1]
  return @views simplexified_facet_data[1][1:num_facets]
end

function get_subfacet_normals(cache)
  _, simplexified_facet_data, _, _, _, _ = cache
  num_facets = simplexified_facet_data[end][1]
  return @views simplexified_facet_data[2][1:num_facets]
end

function get_subfacets_cutinout(cache)
  _, simplexified_facet_data, _, _, _, _ = cache
  num_facets = simplexified_facet_data[end][1]
  return @views simplexified_facet_data[3][1:num_facets]
end

### Facet cutter map

struct PolytopalFacetCutterMap{D,N,Nc} <: PolytopalCutterMap{D,N,Nc} end

PolytopalFacetCutterMap(D,N,Nc) = PolytopalFacetCutterMap{D,N,Nc}();

function Arrays.return_cache(
  ::PolytopalFacetCutterMap{Dp,A},
  pvertices::Vector{VectorValue{Dp,Tp}},
  pgraph::Vector{Vector{Ti}},
  ls_maps::Field...,
) where {Dp,A,Tp,Ti}
  _T = _get_map_eltype(ls_maps)
  p2φs_cache = map(Base.Fix2(return_cache,zero(Point{Dp,_T})),ls_maps)
  φ_val_cache = zeros(_T,2*length(pvertices))
  ## Cell caches
  cell_data = allocate_polytope_data(PolytopeType(Dp,Dp),pgraph,pvertices,ls_maps);
  cell_tree = preallocate_tree(cell_data,A);
  # splitter cache
  cell_splitter_cache = allocate_splitter_cache(PolytopeType(Dp,Dp),pgraph,pvertices,ls_maps);
  # simplexify data/cache
  simplexified_cell_data = allocate_simplexify_data(PolytopeType(Dp,Dp),pgraph,pvertices,ls_maps);
  simplexify_cell_cache = allocate_simplexify_cache(PolytopeType(Dp,Dp),pgraph,pvertices,ls_maps);
  # Return caches
  ls_map_caches = (p2φs_cache, φ_val_cache)
  cell_caches = (cell_tree, cell_splitter_cache, simplexify_cell_cache)
  return simplexified_cell_data, cell_caches, ls_map_caches
end

function _reset!(::PolytopalFacetCutterMap{D},cache) where D
  simplexified_cell_data, cell_caches, _ = cache
  cell_tree, _, _ = cell_caches
  # Reset tree data, we don't fill with zero as we reset caches inside the splitting/simplexifying functions
  reset_touched!(cell_tree)
  # Reset cell counters
  simplexified_cell_data[end][1] = 0
  nothing
end

# No cells cut, return cache
function Arrays.evaluate!(
  cache,
  k::PolytopalFacetCutterMap{D,N,0},
  pvertices::Vector{VectorValue{D,Tp}},
  pgraph::Vector{Vector{Ti}},
  ls_maps::Field...,
) where {D,N,Tp,Ti}
  return cache
end

function Arrays.evaluate!(
  cache,
  k::PolytopalFacetCutterMap{D,A},
  pvertices::Vector{VectorValue{D,Tp}},
  pgraph::Vector{Vector{Ti}},
  ls_maps::Field...,
) where {D,A,Tp,Ti}
  # Reset cache
  _reset!(k,cache)
  # Initialise cell tree
  simplexified_cell_data, cell_caches, ls_map_caches = cache
  cell_tree, _, _ = cell_caches
  copy_polytope_data_to_cell_tree!(cell_tree,pvertices,pgraph)
  # Cells
  recursively_cut_and_simplexify_cells!(k,simplexified_cell_data,ls_maps,cell_caches,ls_map_caches)
  return cache
end

function get_fcell_rvertices(cache)
  simplexified_cell_data, _, _ = cache
  num_cells = simplexified_cell_data[end][1]
  return @views simplexified_cell_data[1][1:num_cells]
end

function get_fsubcell_inout(cache)
  simplexified_cell_data, _, _ = cache
  num_cells = simplexified_cell_data[end][1]
  return @views simplexified_cell_data[3][1:num_cells]
end

### Recursively cut and simplexify

function recursively_cut_and_simplexify_cells!(
  k::PolytopalCutterMap{D},
  simplexified_cell_data,
  ls_maps::Tuple{Vararg{<:Field}},
  cell_caches,
  ls_map_caches
) where D
  cell_tree, cell_splitter_cache, simplexify_cell_cache = cell_caches
  p2φs_cache, φ_val_cache = ls_map_caches
  recursively_cut!(k,PolytopeType(D,D), cell_tree, ls_maps, p2φs_cache, cell_splitter_cache, φ_val_cache)
  simplexify_leaves!(PolytopeType(D,D),simplexified_cell_data,simplexify_cell_cache,cell_tree)
  return nothing
end

function recursively_cut_and_simplexify_facet!(
  k::PolytopalCutterMap{D,N},
  simplexified_facet_data,
  ls_maps::Tuple{Vararg{<:Field}},
  cell_ref_to_phys_map,
  cell_caches,
  facet_caches,
  ls_map_caches,
  ref_to_phys_map_caches
) where {D,N}
  cell_tree, cell_splitter_cache, _ = cell_caches
  facet_tree, facet_splitter_cache, simplexify_facet_cache = facet_caches
  p2φs_cache, φ_val_cache = ls_map_caches
  r2p_cache, phys_vert_cache = ref_to_phys_map_caches
  for i in Base.OneTo(N)
    reset_touched!(facet_tree)
    _φ_map = ls_maps[i]
    _p2φ_cache = p2φs_cache[i];
    # Split polytope
    graph, vertices, inout, counters = cell_tree.data
    verts_i = first(counters);
    _vertices, _graph, _vtx_vals = _prepare_data_views(verts_i,vertices,graph,φ_val_cache,_φ_map,_p2φ_cache);
    state = split!(PolytopeType(D,D),cell_splitter_cache,_graph,_vertices,_vtx_vals)
    (state == :in || state == :out) && continue
    # Get new vertices and reorder
    sp_verts = cell_splitter_cache.vertices
    sp_counter = cell_splitter_cache.counters
    n_v_l = sp_counter[end-1]
    n_v_r = sp_counter[end]
    max_vert_i = max(n_v_l,n_v_r)
    new_vertices = @views sp_verts[verts_i+1:max_vert_i]
    reorder_vertices!(new_vertices)
    # Initialise tree root and get normal
    build_facet_tree_root!(PolytopeType(D-1,D),facet_tree,new_vertices)
    n = facet_tree.data.normal
    _compute_normal!(PolytopeType(D,D),n,_vertices,new_vertices,_φ_map,cell_ref_to_phys_map,
      _p2φ_cache,r2p_cache,φ_val_cache,phys_vert_cache);
    # Recursively cut facets
    recursively_cut!(k, PolytopeType(D-1,D), facet_tree, ls_maps,
      p2φs_cache, facet_splitter_cache, φ_val_cache, i)
    _set_leaves_normal!(n,facet_tree)
    # Simplexify
    simplexify_leaves!(PolytopeType(D-1,D),simplexified_facet_data,simplexify_facet_cache,facet_tree)
  end
  return nothing
end

function recursively_cut!(k::PolytopalCutterMap{D,N}, p::PolytopeType, node,
    remaining_ls_maps, remaining_p2φs_cache, splitter_cache, φ_val_cache, skip=0) where {D,N}
  isempty(remaining_ls_maps) && return
  N_rem = length(remaining_ls_maps)
  graph, vertices, inout, counters = node.data
  # Skip a level-set, this is used for facet cutting
  if skip>0 && skip == N - N_rem + 1
    io_i = counters[end] += 1
    inout[io_i] = CUT # Technically not needed as CUT = 0
    recursively_cut!(k, p, node, @views(remaining_ls_maps[2:end]), @views(remaining_p2φs_cache[2:end]),
      splitter_cache, φ_val_cache, skip)
    return nothing
  end
  verts_i = first(counters);
  # Get data views
  φ_map = first(remaining_ls_maps)
  p2φs_cache = first(remaining_p2φs_cache)
  _vertices, _graph, _vtx_vals = _prepare_data_views(verts_i,vertices,graph,φ_val_cache,φ_map,p2φs_cache)
  # Attempt to split
  state = split!(p,splitter_cache,_graph,_vertices,_vtx_vals)
  if state == :in || state == :out
    # Check in/out of remaining level-sets
    io_i = counters[end] += 1
    inout[io_i] = (state == :in) ? IN : OUT
    recursively_cut!(k, p, node, @views(remaining_ls_maps[2:end]), @views(remaining_p2φs_cache[2:end]),
      splitter_cache, φ_val_cache, skip)
    return nothing
  end
  # Update and continue recursion
  copy_data_to_node!(p,node,splitter_cache)
  recursively_cut!(k, p, node.left, @views(remaining_ls_maps[2:end]), @views(remaining_p2φs_cache[2:end]),
    splitter_cache, φ_val_cache, skip)
  recursively_cut!(k, p, node.right, @views(remaining_ls_maps[2:end]), @views(remaining_p2φs_cache[2:end]),
    splitter_cache, φ_val_cache, skip)
  return nothing
end

### Cutter helpers

function _prepare_data_views(i,vertices,graph,φ_vals,φ_map,p2φs_cache)
  _vertices = @views vertices[1:i]
  # Get vertex values
  map!(φ_vals,_vertices) do vertex
    evaluate!(p2φs_cache,φ_map,vertex)
  end
  _vtx_vals = @views φ_vals[1:i]
  if !isnothing(graph)
    _graph = @views graph[1:i]
    return _vertices, _graph, _vtx_vals
  else
    return _vertices, graph, _vtx_vals
  end
end

function _compute_normal!(
  ::PolytopeType{Dc,Dp},
  normal,
  pvertices,
  fvertices,
  φ_map,
  cell_ref_to_phys_map,
  p2φs_cache,
  r2p_cache,
  φ_val_cache,
  phys_verts_cache
) where {Dc,Dp}
  # Prepare caches
  _phys_verts_cache = @views phys_verts_cache[1:length(fvertices)]
  map!(phys_verts_cache,fvertices) do vertex
    evaluate!(r2p_cache,cell_ref_to_phys_map,vertex)
  end
  _φ_val_cache = @views φ_val_cache[1:length(pvertices)]
  map!(_φ_val_cache,pvertices) do vertex
    evaluate!(p2φs_cache,φ_map,vertex)
  end
  # Compute normal
  in_vtx = findfirst(v->v<0,_φ_val_cache)
  out_vtx = findfirst(v->v>0,_φ_val_cache)
  in_to_out_orientation = evaluate!(r2p_cache,cell_ref_to_phys_map,pvertices[out_vtx])
  in_to_out_orientation -= evaluate!(r2p_cache,cell_ref_to_phys_map,pvertices[in_vtx])
  n = _facet_normal(Val(Dp), _phys_verts_cache)
  if (n ⋅ in_to_out_orientation) < 0
    n = -n
  end
  normal[1] = n
  nothing
end

function _facet_normal(::Val{2}, subfacet_vertices_phys)
  p1, p2 = @views subfacet_vertices_phys[1:2]
  compute_normal_vector(p2-p1)
end

function _facet_normal(::Val{3}, subfacet_vertices_phys)
  p1, p2, p3 = @views subfacet_vertices_phys[1:3]
  compute_normal_vector(p2-p1,p3-p1)
end

function compute_normal_vector(u1)
  v = LevelSetCutters._orthogonal_vector(u1)
  m = sqrt(inner(v,v))
  if m < eps()
    return zero(v)
  else
    return v/m
  end
end

function compute_normal_vector(u1,u2)
  v = LevelSetCutters._orthogonal_vector(u1,u2)
  m = sqrt(inner(v,v))
  if m < eps()
    return zero(v)
  else
    return v/m
  end
end

### Cache helpers

function _get_map_eltype(ls_maps::Tuple{Vararg{<:LinearCombinationField}})
  T = eltype.(getfield.(ls_maps,:values))
  promote_type(T...)
end

function _get_map_eltype(ls_maps::Tuple{Vararg{<:OperationField{<:LinearCombinationField}}})
  T = eltype.(eltype.(getfield.(first.(getfield.(ls_maps,:fields)),:values)))
  promote_type(T...)
end

function _get_map_eltype(::Vector{Vector{T}}) where T
  T
end

# Preallocate tree data
preallocate_tree(cell_data, ::PolytopalCutterMap{D,N}) where {D,N} = preallocate_tree(cell_data, N)

# Copy data to tree
function copy_polytope_data_to_cell_tree!(tree,vertices,graph)
  for i in eachindex(graph)
    tree.data.graph[i] .= graph[i]
  end
  copyto!(tree.data.vertices, vertices)
  tree.data.counters[1] = length(vertices)
  tree.data.counters[2] = 0
  tree.touched = true
  nothing
end

function build_facet_tree_root!(::PolytopeType{1,2},tree,vertices)
  copyto!(tree.data.vertices, vertices)
  tree.data.counters[1] = length(vertices)
  tree.data.counters[2] = 0
  tree.touched = true
  nothing
end

function build_facet_tree_root!(::PolytopeType{2,3},tree,vertices)
  copyto!(tree.data.vertices, vertices)
  graph = tree.data.graph
  for i in eachindex(vertices)
    inext = i == length(vertices) ? 1 : i+1
    iprev = i == 1 ? length(vertices) : i-1
    graph[i][1] = iprev
    graph[i][2] = inext
  end
  tree.data.counters[1] = length(vertices)
  tree.data.counters[2] = 0
  tree.touched = true
  nothing
end

function _set_leaves_normal!(n::Vector{<:VectorValue},facet_tree)
  function _set_data(data,node)
    node.data.normal[1] = data[1]
  end
  map_leaves!(_set_data,n,facet_tree)
  nothing
end

### Reorder vertices
function reorder_vertices!(vertices::AbstractVector{<:Point{3}})
  @check length(vertices) >= 3
  loc0 = vertices[1]                       # local origin
  locx = vertices[2] - loc0                # local X axis
  normal = cross(locx, vertices[3] - loc0) # vector orthogonal to polygon plane
  locy = cross(normal, locx)                   # local Y axis
  locx /= norm(locx)                           # Normalise
  locy /= norm(locy)

  # Compute centroid
  n = length(vertices)
  centroid_x = 0.0
  centroid_y = 0.0
  for p in vertices
    rel_pos = p - loc0
    local_x = dot(rel_pos, locx)
    local_y = dot(rel_pos, locy)
    centroid_x += local_x
    centroid_y += local_y
  end
  centroid_x /= n
  centroid_y /= n

  # Use merge sort by angle
  merge_sort_by_local_angle!(vertices, 1, n, loc0, locx, locy, centroid_x, centroid_y)
  vertices
end

function merge_sort_by_local_angle!(vertices, left, right, loc0, locx, locy, centroid_x, centroid_y)
  left >= right && return

  mid = (left + right) ÷ 2
  merge_sort_by_local_angle!(vertices, left, mid, loc0, locx, locy, centroid_x, centroid_y)
  merge_sort_by_local_angle!(vertices, mid + 1, right, loc0, locx, locy, centroid_x, centroid_y)
  merge_by_local_angle!(vertices, left, mid, right, loc0, locx, locy, centroid_x, centroid_y)
end

function merge_by_local_angle!(vertices, left, mid, right, loc0, locx, locy, centroid_x, centroid_y)
  i, j = left, mid + 1
  function compute_angle(v)
    rel_pos = v - loc0
    local_x = dot(rel_pos, locx)
    local_y = dot(rel_pos, locy)
    mod(atan(local_y - centroid_y, local_x - centroid_x), 2π)
  end

  while i <= mid && j <= right
    # Compute angles
    angle_i = compute_angle(vertices[i])
    angle_j = compute_angle(vertices[j])

    if angle_i <= angle_j
      i += 1
    else
      # Rotate elements: move vertices[j] to position i
      temp = vertices[j]
      for k in j:-1:i+1
        vertices[k] = vertices[k-1]
      end
      vertices[i] = temp
      i += 1
      mid += 1
      j += 1
    end
  end
end

function reorder_vertices!(vertices::AbstractVector{<:Point{2}})
  vertices
end