using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.ReferenceFEs,
  Gridap.CellData, Gridap.Fields, Gridap.Arrays, Gridap.Helpers, Gridap.FESpaces
using Gridap.ReferenceFEs: get_graph, isactive
using Gridap.Fields: OperationField, LinearCombinationField
using Gridap.FESpaces: get_vector_type
using Gridap.Geometry: TriangulationView

using GridapDistributed
using GridapDistributed: DistributedTriangulation, DistributedDiscreteModel, DistributedCellField, DistributedMultiFieldCellField

using GridapEmbedded, GridapEmbedded.LevelSetCutters, GridapEmbedded.Interfaces, GridapEmbedded.Distributed
using GridapEmbedded.LevelSetCutters: _compute_bgcell_to_inoutcut, _find_unique_leaves, extract_dualized_cell_values,
  SubCellTriangulation, SubFacetTriangulation, compute_cell_maps
using GridapEmbedded.Interfaces: compute_inoutboundary, compute_inoutcut, INTERFACE, compute_subcell_to_inout, CutInOrOut
import GridapEmbedded.CSG: Node, Leaf, get_tree, replace_data
import GridapEmbedded.LevelSetCutters: cut, cut_facets, compute_bgfacet_to_inoutcut, compute_bgcell_to_inoutcut,
  similar_geometry, compatible_geometries, CUT, _compute_ls_to_bgcell_to_inoutcut, DifferentiableTriangulation
using GridapEmbedded.Distributed: DistributedDiscreteGeometry, DistributedEmbeddedDiscretization
using PartitionedArrays

using AbstractTrees

include("TouchedBinaryNode.jl")

include("Splitters.jl")

include("Simplexifiers.jl")

include("PolytopalCutterMap.jl")

"""
    struct DiscreteGeometryFromFEFunction{D,T} <: CSG.Geometry

A discrete geometry defined by a level-set function represented as an `FEFunction`.

# Constructors

    DiscreteGeometryFromFEFunction(φh::FEFunction,point_to_coords::AbstractVector;name::String="")
    DiscreteGeometryFromFEFunction(φh::FEFunction,model::DiscreteModel;name::String="")

"""
struct DiscreteGeometryFromFEFunction{D,T} <: GridapEmbedded.CSG.Geometry
  tree::Node
  point_to_coords::Vector{Point{D,T}}
end

get_tree(a::DiscreteGeometryFromFEFunction) = a.tree

similar_geometry(a::DiscreteGeometryFromFEFunction,tree::Node) = DiscreteGeometryFromFEFunction(tree,a.point_to_coords)

function compatible_geometries(a::DiscreteGeometryFromFEFunction,b::DiscreteGeometryFromFEFunction)
  @assert a.point_to_coords === b.point_to_coords || a.point_to_coords == b.point_to_coords
  (a,b)
end

function DiscreteGeometryFromFEFunction(
  φh::CellField,model::DiscreteModel;name::String="")
  data = (φh,name,nothing)
  tree = Leaf(data)
  point_to_coords = collect1d(get_node_coordinates(model))
  DiscreteGeometryFromFEFunction(tree,point_to_coords)
end


"""
    struct PolytopalLevelSetCutter <: Cutter end

A polytopal cutter for geometries defined with `DiscreteGeometryFromFEFunction`.

# Usage

- cut(::PolytopalLevelSetCutter, model::DiscreteModel,geom::DiscreteGeometryFromFEFunction)
- cut_facets(::PolytopalLevelSetCutter, model::DiscreteModel,geom::DiscreteGeometryFromFEFunction)
- compute_bgcell_to_inoutcut(::PolytopalLevelSetCutter, model::DiscreteModel,geom::DiscreteGeometryFromFEFunction)
- compute_bgfacet_to_inoutcut(::PolytopalLevelSetCutter, model::DiscreteModel,geom::DiscreteGeometryFromFEFunction)

"""
struct PolytopalLevelSetCutter <: GridapEmbedded.Interfaces.Cutter end

function cut(model::DiscreteModel,geom::DiscreteGeometryFromFEFunction)
  cut(PolytopalLevelSetCutter(),model,geom)
end

function cut(::PolytopalLevelSetCutter,model::DiscreteModel,geom::DiscreteGeometryFromFEFunction{D,T}) where {D,T}
  tree = get_tree(geom)
  φh_s, oid_to_ls = _find_unique_leaves(tree);
  cell_map_ls = get_data.(φh_s);
  cell_values = map(Base.Fix1(extract_dualized_cell_values,model),φh_s)

  num_ls = length(φh_s)
  p0 = zero(Point{D,T})

  ## Cell maps and caches
  cell_r2p = get_cell_map(model)
  cell_r2p_cache = array_cache(cell_r2p)
  cell_val_cache = array_cache(first(cell_values))
  ## Some cell info
  topo = get_grid_topology(model)
  polys = get_polytopes(topo)
  p_ref = first(polys)
  @assert isone(length(polys)) "Only models with a single reference polytope are currently supported."
  @assert is_simplex(p_ref) "Only simplex background meshes are currently supported."
  cell_nodes = Geometry.get_faces(topo,D,0)
  Tn = eltype(eltype(cell_nodes))
  expected_cell_simplex = ReferenceFEs.simplex_polytope(Val(D))
  expected_subfacet_simplex = ReferenceFEs.simplex_polytope(Val(D-1))
  simplex_num_verts = num_vertices(expected_cell_simplex)
  facet_simplex_num_verts = num_vertices(expected_subfacet_simplex)
  ## More caches
  cell_r2p_simplex_map_cache = return_cache(first(cell_r2p),[p0 for _ in 1:simplex_num_verts])
  cell_r2p_fsimplex_map_cache = return_cache(first(cell_r2p),[p0 for _ in 1:facet_simplex_num_verts])
  ## Compute in/out for bgcells
  cell_iscut = [zeros(Bool,num_cells(model)) for _ in Base.OneTo(num_ls)];
  ls_to_bgcell_to_inoutcut = [zeros(Int8,num_cells(model)) for _ in Base.OneTo(num_ls)];
  compute_ls_to_bgcell_inoutcut_and_iscut!(ls_to_bgcell_to_inoutcut,cell_iscut,cell_values,cell_val_cache)

  ## Upper bounds
  cut_bgcells = findall(x->x>0,sum(cell_iscut))
  num_cut = length(cut_bgcells)
  nv = num_vertices(p_ref)
  max_subcells = num_cut * ifelse(D==3, 2*simplex_num_verts, simplex_num_verts) * 2^num_ls
  max_cutnodes = num_cut * num_ls * max(1, nv - 4 + 2^num_ls) * facet_simplex_num_verts
  restricted_topo,_ = Geometry.restrict(topo,findall(sum(cell_iscut) .> 0))
  max_nodes = num_faces(restricted_topo,0) + max_cutnodes
  ## Preallocate SubCellData
  cell_to_bgcell = Vector{eltype(cut_bgcells)}(undef,max_subcells)
  cell_point_to_coords = Vector{Point{D,T}}(undef,nv*max_subcells)
  cell_point_to_rcoords = Vector{Point{D,T}}(undef,nv*max_subcells)
  ## Preallocate SubFacetData
  facet_to_bgcell = Vector{eltype(cut_bgcells)}(undef,max_nodes)
  facet_to_normal = Vector{VectorValue{D,T}}(undef,max_nodes)
  facet_point_to_coords = Vector{Point{D,T}}(undef,max_nodes)
  facet_point_to_rcoords = Vector{Point{D,T}}(undef,max_nodes)
  ## Preallocate IN/OUT/CUT data
  ls_to_subcell_to_inoutcut = zeros(Int8,num_ls,max_subcells)
  ls_to_subfacet_to_inoutcut = zeros(Int8,num_ls,max_nodes)

  # Setup cutter
  bg_vertices = get_cell_ref_coordinates(model)
  bg_ctypes = get_cell_type(model)
  bgcell_to_polys = expand_cell_data(get_polytopes(model),bg_ctypes)
  bgcell_to_graph = lazy_map(pref_to_gen_polytope_graph,bgcell_to_polys);
  cut_bgcell_graph = lazy_map(Reindex(bgcell_to_graph),cut_bgcells);
  cut_bgcell_vertices = lazy_map(Reindex(bg_vertices),cut_bgcells);
  cut_bgcell_to_ls_maps = map(a->lazy_map(Reindex(a),cut_bgcells),cell_map_ls);
  cell_ref_to_phys_map = lazy_map(Reindex(cell_r2p),cut_bgcells)
  pcutter = lazy_map(PolytopalCellCutterMap{D,num_ls,num_cut}(),cut_bgcell_vertices,cut_bgcell_graph,
    cell_ref_to_phys_map,cut_bgcell_to_ls_maps...);
  pcutter_cache = array_cache(pcutter);

  subcell_count = 0
  subcell_node_count = 0
  subfacet_count = 0
  subfacet_node_count = 0
  for (i,cell) in enumerate(cut_bgcells)
    cache = getindex!(pcutter_cache,pcutter,i)
    cell_rvertices = get_cell_rvertices(cache)
    subcell_inout = get_subcell_inout(cache)
    facet_rvertices = get_facet_rvertices(cache)
    subfacet_normals = get_subfacet_normals(cache)
    subfacets_cutinout = get_subfacets_cutinout(cache)
    num_new_cells = length(cell_rvertices)
    num_new_facets = length(facet_rvertices)
    num_new_cell_nodes = num_new_cells*simplex_num_verts
    num_new_facets_nodes = num_new_facets*facet_simplex_num_verts
    # Store cut/in/out data
    my_setindex!(ls_to_subcell_to_inoutcut,subcell_inout,subcell_count+1,subcell_count+num_new_cells)
    my_setindex!(ls_to_subfacet_to_inoutcut,subfacets_cutinout,subfacet_count+1,subfacet_count+num_new_facets)
    # cell/facet to bgcell
    my_setindex!(cell_to_bgcell,cell,subcell_count+1,subcell_count+num_new_cells)
    my_setindex!(facet_to_bgcell,cell,subfacet_count+1,subfacet_count+num_new_facets)
    # cell/facet points to coords/rcoords
    cell_map = getindex!(cell_r2p_cache,cell_r2p,cell)
    set_points_from_coords!(cell_point_to_coords,cell_point_to_rcoords,
      cell_rvertices,subcell_node_count,simplex_num_verts,cell_map,cell_r2p_simplex_map_cache)
    set_points_from_coords!(facet_point_to_coords,facet_point_to_rcoords,
      facet_rvertices,subfacet_node_count,facet_simplex_num_verts,cell_map,cell_r2p_fsimplex_map_cache)
    # facet to normal
    my_setindex!(facet_to_normal,subfacet_normals,subfacet_count+1,subfacet_count+num_new_facets)

    subcell_count += num_new_cells
    subcell_node_count += num_new_cell_nodes
    subfacet_count += num_new_facets
    subfacet_node_count += num_new_facets_nodes
  end

  # Resize
  _ls_to_subcell_to_inoutcut = [ls_to_subcell_to_inoutcut[i,1:subcell_count] for i in 1:size(ls_to_subcell_to_inoutcut,1)]
  _ls_to_subfacet_to_inoutcut = [ls_to_subfacet_to_inoutcut[i,1:subfacet_count] for i in 1:size(ls_to_subfacet_to_inoutcut,1)]
  resize!(cell_to_bgcell,subcell_count)
  resize!(cell_point_to_coords,subcell_node_count)
  resize!(cell_point_to_rcoords,subcell_node_count)
  resize!(facet_to_bgcell,subfacet_count)
  resize!(facet_to_normal,subfacet_count)
  resize!(facet_point_to_coords,subfacet_node_count)
  resize!(facet_point_to_rcoords,subfacet_node_count)

  cell_to_bgcell = convert(Vector{Tn},cell_to_bgcell)
  facet_to_bgcell = convert(Vector{Tn},facet_to_bgcell)

  # Cell/facet to points
  cell_to_points_data = Base.OneTo(Tn(subcell_node_count+simplex_num_verts))
  cell_to_points_prts = convert(StepRange{Tn,Tn},1:simplex_num_verts:subcell_node_count+simplex_num_verts)
  facet_to_points_data = Base.OneTo(Tn(subfacet_node_count+facet_simplex_num_verts))
  facet_to_points_prts = convert(StepRange{Tn,Tn},1:facet_simplex_num_verts:subfacet_node_count+facet_simplex_num_verts)
  cell_to_points = Table(collect(cell_to_points_data),collect(cell_to_points_prts))
  facet_to_points = Table(collect(facet_to_points_data),collect(facet_to_points_prts))

  # Subcell/subfacet data
  subcells = LevelSetCutters.SubCellData(cell_to_points, cell_to_bgcell, cell_point_to_coords, cell_point_to_rcoords)
  subfacets = LevelSetCutters.SubFacetData(facet_to_points, facet_to_normal, facet_to_bgcell, facet_point_to_coords, facet_point_to_rcoords)

  EmbeddedDiscretization(
    model,
    ls_to_bgcell_to_inoutcut,
    subcells,
    _ls_to_subcell_to_inoutcut,
    subfacets,
    _ls_to_subfacet_to_inoutcut,
    oid_to_ls,
    geom
  )
end

function cut_facets(model::DiscreteModel,geom::DiscreteGeometryFromFEFunction)
  cut_facets(PolytopalLevelSetCutter(),model,geom)
end

function cut_facets(::PolytopalLevelSetCutter,model::DiscreteModel,geom::DiscreteGeometryFromFEFunction{D,T}) where {D,T}
  tree = get_tree(geom)
  φh_s, oid_to_ls = _find_unique_leaves(tree);
  num_ls = length(φh_s)

  # Facet triangulation
  face_to_mask = ones(Bool,num_faces(model,D-1))
  facet_trian = BoundaryTriangulation(model,face_to_mask)
  facet_vertices = get_cell_ref_coordinates(facet_trian)
  ref_num_verts = num_vertices(first(get_polytopes(facet_trian)))
  ## Cell maps and caches
  cell_r2p = get_cell_map(facet_trian)
  cell_r2p_cache = array_cache(cell_r2p)
  ## Get level-set maps on facet_trian
  φh_s_moved = map(v->change_domain(v,facet_trian, ReferenceDomain()),φh_s)
  cell_map_ls = get_data.(φh_s_moved)
  cell_to_φs_maps_cache = array_cache.(cell_map_ls)
  p2φs_cache = return_cache(first(first(cell_map_ls)),[zero(Point{D-1,T}) for _ in 1:ref_num_verts]);
  ## Some cell info
  topo = get_grid_topology(model)
  p_ref = first(get_polytopes(topo))
  cell_nodes = Geometry.get_faces(topo,D,0)
  Tn = eltype(eltype(cell_nodes))
  polys = get_polytopes(facet_trian)
  p_ref = first(polys)
  @check isone(length(polys)) "Only models with a single reference polytope are currently supported."
  @check is_simplex(p_ref) "Only simplex background meshes are currently supported."
  expected_cell_simplex = ReferenceFEs.simplex_polytope(Val(D-1))
  simplex_num_verts = num_vertices(expected_cell_simplex)
  ## More caches
  p0 = zero(Point{D-1,T})
  cell_r2p_simplex_map_cache = return_cache(first(cell_r2p),[p0 for _ in 1:simplex_num_verts]);
  ## Compute in/out for bgcells
  cell_iscut = [zeros(Bool,num_cells(facet_trian)) for _ in Base.OneTo(num_ls)];
  ls_to_facet_to_inoutcut = [zeros(Int8,num_cells(facet_trian)) for _ in Base.OneTo(num_ls)];
  compute_ls_to_bgfacet_inoutcut_and_iscut!(ls_to_facet_to_inoutcut,cell_iscut,facet_vertices,cell_map_ls,cell_to_φs_maps_cache,p2φs_cache)

  ## Upper bounds
  cut_bgcells = findall(x->x>0,sum(cell_iscut))
  num_cut = length(cut_bgcells)
  max_subcells = num_cut * max(1, ref_num_verts - 4 + 2*2^num_ls)
  ## Preallocate SubCellData
  cell_to_bgcell = Vector{eltype(cut_bgcells)}(undef,max_subcells)
  cell_point_to_coords = Vector{Point{D,T}}(undef,max_subcells * simplex_num_verts)
  cell_point_to_rcoords = Vector{Point{D-1,T}}(undef,max_subcells * simplex_num_verts)
  ## Preallocate IN/OUT/CUT data
  ls_to_subcell_to_inoutcut = zeros(Int8,num_ls,max_subcells)

  # Setup cutter
  bg_vertices = get_cell_ref_coordinates(facet_trian)
  cut_bgcell_vertices = lazy_map(Reindex(bg_vertices),cut_bgcells);
  if D == 2
    cut_bgcell_graph = fill([[Int32(1)],[Int32(2)]],length(cut_bgcells));
  else
    bg_ctypes = get_cell_type(facet_trian)
    bgcell_to_polys = expand_cell_data(polys,bg_ctypes)
    bgcell_to_graph = lazy_map(pref_to_gen_polytope_graph,bgcell_to_polys);
    cut_bgcell_graph = lazy_map(Reindex(bgcell_to_graph),cut_bgcells);
  end
  cut_bgcell_to_ls_maps = map(a->lazy_map(Reindex(a),cut_bgcells),cell_map_ls);
  pcutter = lazy_map(PolytopalFacetCutterMap{D-1,num_ls,num_cut}(),cut_bgcell_vertices,cut_bgcell_graph,cut_bgcell_to_ls_maps...);
  pcutter_cache = array_cache(pcutter);

  subcell_count = 0
  subcell_node_count = 0
  for (i,cell) in enumerate(cut_bgcells)
    cache = getindex!(pcutter_cache,pcutter,i)
    cell_rvertices = get_fcell_rvertices(cache)
    subcell_inout = get_fsubcell_inout(cache)
    num_new_cells = length(cell_rvertices)
    num_new_cell_nodes = num_new_cells*simplex_num_verts
    # Store cut/in/out data
    my_setindex!(ls_to_subcell_to_inoutcut,subcell_inout,subcell_count+1,subcell_count+num_new_cells)
    # cell/facet to bgcell
    my_setindex!(cell_to_bgcell,cell,subcell_count+1,subcell_count+num_new_cells)
    # cell/facet points to coords/rcoords
    cell_map = getindex!(cell_r2p_cache,cell_r2p,cell)
    set_points_from_coords!(cell_point_to_coords,cell_point_to_rcoords,
      cell_rvertices,subcell_node_count,simplex_num_verts,cell_map,cell_r2p_simplex_map_cache)

    subcell_count += num_new_cells
    subcell_node_count += num_new_cell_nodes
  end

  # Resize
  _ls_to_subcell_to_inoutcut = [ls_to_subcell_to_inoutcut[i,1:subcell_count] for i in 1:size(ls_to_subcell_to_inoutcut,1)]
  resize!(cell_to_bgcell,subcell_count)
  resize!(cell_point_to_coords,subcell_node_count)
  resize!(cell_point_to_rcoords,subcell_node_count)

  cell_to_bgcell = convert(Vector{Tn},cell_to_bgcell)

  # Cell/facet to points
  cell_to_points_data = Base.OneTo(Tn(subcell_node_count+simplex_num_verts))
  cell_to_points_prts = convert(StepRange{Tn,Tn},1:simplex_num_verts:subcell_node_count+simplex_num_verts)
  cell_to_points = Table(collect(cell_to_points_data),collect(cell_to_points_prts))

  # Subcell/subfacet data
  subcells = LevelSetCutters.SubCellData(cell_to_points, cell_to_bgcell, cell_point_to_coords, cell_point_to_rcoords)

  return EmbeddedFacetDiscretization(
    model,
    ls_to_facet_to_inoutcut,
    subcells,
    _ls_to_subcell_to_inoutcut,
    oid_to_ls,
    geom
  )
end

function compute_bgcell_to_inoutcut(model::DiscreteModel,geom::DiscreteGeometryFromFEFunction)
  compute_bgcell_to_inoutcut(PolytopalLevelSetCutter(),model,geom)
end

function compute_bgcell_to_inoutcut(::PolytopalLevelSetCutter,model::DiscreteModel,geom::DiscreteGeometryFromFEFunction)
  _compute_bgcell_to_inoutcut(model,geom)
end

function compute_bgfacet_to_inoutcut(model::DiscreteModel,geom::DiscreteGeometryFromFEFunction)
  compute_bgfacet_to_inoutcut(PolytopalLevelSetCutter(),model,geom)
end

function compute_bgfacet_to_inoutcut(::PolytopalLevelSetCutter,model::DiscreteModel,geom::DiscreteGeometryFromFEFunction{D}) where D
  face_to_mask = ones(Bool,num_faces(model,D-1))
  facet_trian = BoundaryTriangulation(model,face_to_mask)
  _compute_bgcell_to_inoutcut(facet_trian,geom)
end

function _compute_ls_to_bgcell_to_inoutcut(model::DiscreteModel,geom::DiscreteGeometryFromFEFunction)
  tree = get_tree(geom)
  φh_s, oid_to_ls = _find_unique_leaves(tree)
  num_ls = length(φh_s)
  cell_values = map(Base.Fix1(extract_dualized_cell_values,model),φh_s)
  cell_val_cache = array_cache(first(cell_values))
  ls_to_cell_to_inoutcut = [zeros(Int8,num_cells(model)) for _ in Base.OneTo(num_ls)];
  compute_ls_to_bgcell_inoutcut!(ls_to_cell_to_inoutcut,cell_values,cell_val_cache)
  return ls_to_cell_to_inoutcut, tree, oid_to_ls
end

@inline function compute_ls_to_bgcell_inoutcut!(inoutcut,cell_values,cache)
  for (i,cell_vals) in enumerate(cell_values)
    for c in eachindex(cell_vals)
      φ_c = getindex!(cache,cell_vals,c)
      @views inoutcut[i][c] = ifelse(is_in(φ_c),IN,ifelse(is_out(φ_c),OUT,CUT))
    end
  end
end

# Compute facet inoutcut, moves FEFunctions to the facet triangulation and evaluates
# maps on the facets to determine in/out/cut states
function _compute_ls_to_bgcell_to_inoutcut(model::BoundaryTriangulation{D},geom::DiscreteGeometryFromFEFunction{P,T}) where {D,P,T}
  tree = get_tree(geom)
  φh_s, oid_to_ls = _find_unique_leaves(tree)
  num_ls = length(φh_s)
  polytopes = get_polytopes(model)
  @check length(polytopes) == 1
  ref_num_verts = num_vertices(first(polytopes))
  φsh_moved = map(v->change_domain(v,model, ReferenceDomain()),φh_s);

  ls_to_facet_to_inoutcut = [zeros(Int8,num_cells(model)) for _ in Base.OneTo(num_ls)];
  facet_vertices = get_cell_ref_coordinates(model)
  cell_map_ls = get_data.(φsh_moved)
  cell_to_φs_maps_cache = array_cache.(cell_map_ls)
  p2φs_cache = return_cache(first(first(cell_map_ls)),[zero(Point{D,T}) for _ in 1:ref_num_verts]);
  compute_ls_to_bgfacet_inoutcut!(ls_to_facet_to_inoutcut,facet_vertices,cell_map_ls,cell_to_φs_maps_cache,p2φs_cache)
  return ls_to_facet_to_inoutcut, tree, oid_to_ls
end

@inline function compute_ls_to_bgfacet_inoutcut!(inoutcut,ref_coords,φs_cell_maps,acache,rcache)
  for (i,cell_maps) in enumerate(φs_cell_maps)
    for c in eachindex(cell_maps)
      φ_c = getindex!(acache[i],cell_maps,c)
      φ_cv = evaluate!(rcache,φ_c,ref_coords[i])
      @views inoutcut[i][c] = ifelse(is_in(φ_cv),IN,ifelse(is_out(φ_cv),OUT,CUT))
    end
  end
end

## Helpers
function pref_to_gen_polytope_graph(p::Polytope)
  if p == TRI
    e_v_graph = [Int32[2,3],Int32[3,1],Int32[1,2]]
  elseif p == QUAD
    e_v_graph = [Int32[2,3],Int32[4,1],Int32[1,4],Int32[3,2]]
  elseif p == TET
    e_v_graph = [Int32[3,4,2],Int32[1,4,3],Int32[2,4,1],Int32[3,2,1]]
  elseif p == HEX
    e_v_graph = [
      Int32[5, 2, 3],
      Int32[6, 4, 1],
      Int32[7, 1, 4],
      Int32[8, 3, 2],
      Int32[1, 7, 6],
      Int32[2, 5, 8],
      Int32[3, 8, 5],
      Int32[4, 6, 7]
    ]
  else
    @notimplemented
  end
end

@inline function my_setindex!(v::Vector{V},val::V,start::T,stop::T) where {V,T}
  for i in UnitRange{T}(start,stop)
    @views v[i] = val
  end
end

@inline function my_setindex!(v::Vector{V},val::AbstractVector,start::T,stop::T) where {V,T}
  for (i,j) in enumerate(UnitRange{T}(start,stop))
    @views v[j] = val[i]
  end
end

@inline function my_setindex!(v::Matrix{V},val::AbstractVector{Vector{V}},start::T,stop::T) where {V,T}
  for (i,j) in enumerate(UnitRange{T}(start,stop))
    @views v[:,j] = val[i]
  end
end

@inline function set_points_from_coords!(p2c,p2rc,rverts,nodec,ex_nverts,cmap,cache)
  for k in eachindex(rverts)
    idx1 = nodec+1+(k-1)*ex_nverts
    idx2 = nodec+k*ex_nverts
    cell_vertices = evaluate!(cache,cmap,rverts[k])
    my_setindex!(p2c,cell_vertices,idx1,idx2)
    my_setindex!(p2rc,rverts[k],idx1,idx2)
  end
end

@inline function set_points_from_coords!(p2rc,rverts,nodec,ex_nverts)
  for k in eachindex(rverts)
    idx1 = nodec+1+(k-1)*ex_nverts
    idx2 = nodec+k*ex_nverts
    my_setindex!(p2rc,rverts[k],idx1,idx2)
  end
end

@inline is_in(vals) = all(v->v<0,vals)
@inline is_out(vals) = all(v->v>0,vals)
@inline is_cut(vals) = any(v->v<0,vals) && any(v->v>0,vals)

@inline function compute_ls_to_bgcell_inoutcut_and_iscut!(inoutcut,iscut,cell_values,cache)
  for (i,cell_vals) in enumerate(cell_values)
    for c in eachindex(cell_vals)
      φ_c = getindex!(cache,cell_vals,c)
      @views inoutcut[i][c] = ifelse(is_in(φ_c),IN,ifelse(is_out(φ_c),OUT,CUT))
      @views iscut[i][c] = is_cut(φ_c)
    end
  end
end

@inline function compute_ls_to_bgcell_iscut!(iscut,cell_values,cache)
  for (i,cell_vals) in enumerate(cell_values)
    for c in eachindex(cell_vals)
      φ_c = getindex!(cache,cell_vals,c)
      @views iscut[i][c] = is_cut(φ_c)
    end
  end
end

@inline function compute_ls_to_bgfacet_inoutcut_and_iscut!(inoutcut,iscut,ref_coords,φs_cell_maps,acache,rcache)
  for (i,cell_maps) in enumerate(φs_cell_maps)
    for c in eachindex(cell_maps)
      φ_c = getindex!(acache[i],cell_maps,c)
      φ_cv = evaluate!(rcache,φ_c,ref_coords[i])
      @views inoutcut[i][c] = ifelse(is_in(φ_cv),IN,ifelse(is_out(φ_cv),OUT,CUT))
      @views iscut[i][c] = is_cut(φ_cv)
    end
  end
end

include("DifferentiableCutPolyTriangulations.jl")

include("DistributedPolytopalCutters.jl")