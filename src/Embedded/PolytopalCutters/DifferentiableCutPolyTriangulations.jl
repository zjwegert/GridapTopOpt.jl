"""
    mutable struct DifferentiableCutPolyTriangulation{Dc,Dp,A,B} <: Triangulation{Dc,Dp}

A DifferentiableCutPolyTriangulation is a wrapper around an embedded triangulation
(i.e SubCellTriangulation or SubFacetTriangulation) implementing all the necessary
methods to compute derivatives w.r.t. deformations of one or more level-set functions.
This is a more general version of the `DifferentiableTriangulation` type in GridapEmbedded for
differentiation through a PolytopalLevelSetCutter.

# Constructors:

    DifferentiableTriangulation(cutgeo::EmbeddedDiscretization, args...)
    DifferentiableEmbeddedBoundary(cutgeo::EmbeddedDiscretization, args...)

where `cutgeo` is an embedded discretization resulting from `PolytopalLevelSetCutter`. The `args...`
are the same as for the constructors of `Triangulation` and `EmbeddedBoundary` in GridapEmbedded.

Geometries in `cutgeo` must be of type `DiscreteGeometryFromFEFunction`.
"""
mutable struct DifferentiableCutPolyTriangulation{Dc,Dp,A,B} <: Triangulation{Dc,Dp}
  const trian :: A
  const fe_space :: B
  const cutgeo :: EmbeddedDiscretization
  cell_maps
  cached_rcoords
  cached_normals
  const caches
  function DifferentiableCutPolyTriangulation(
    trian :: Triangulation{Dc,Dp},
    cutgeo :: EmbeddedDiscretization,
    cell_maps,
    cached_rcoords,
    cached_normals,
    caches
  ) where {Dc,Dp}
    tree = get_tree(cutgeo.geo)
    φh_s, _ = _find_unique_leaves(tree);
    Us = get_fe_space.(φh_s)
    length(Us) > 1 && @check any(!Base.Fix2(===,Us[1]), Us) """
    The autodiff algorithm requires that the FE spaces associated
    with each level-set function are programmatically different.

    E.g., please replace

      V_φ = TestFESpace(bgmodel, ...)
      φh1 = interpolate(..., V_φ)
      φh2 = interpolate(..., V_φ)

    with

      V_φ1 = TestFESpace(bgmodel, ...)
      V_φ2 = TestFESpace(bgmodel, ...)
      φh1 = interpolate(..., V_φ1)
      φh2 = interpolate(..., V_φ2)
    """
    A = typeof(trian)
    B = typeof(Us)
    new{Dc,Dp,A,B}(trian,Us,cutgeo,cell_maps,cached_rcoords,cached_normals,caches)
  end
end

# Getters

function get_differentiable_trian(trian::DifferentiableCutPolyTriangulation)
  trian
end

function get_differentiable_trian(::Triangulation)
  nothing
end

# Constructors

DifferentiableTriangulation(trian::Triangulation,cutgeo::EmbeddedDiscretization,args...) = trian

DifferentiableEmbeddedBoundary(trian::Triangulation,cutgeo::EmbeddedDiscretization,args...) = trian

function DifferentiableTriangulation(
  trian :: SubCellTriangulation,
  cutgeo :: EmbeddedDiscretization,
  in_or_out :: CutInOrOut,
  geo :: DiscreteGeometryFromFEFunction
)
  cut_bgcells,ls_to_bgcell_iscut = precompute_caches(cutgeo)
  mask = precompute_subcell_mask(trian,cutgeo,in_or_out,geo)
  caches = (;cut_bgcells,ls_to_bgcell_iscut,mask)
  return DifferentiableCutPolyTriangulation(trian,cutgeo,nothing,nothing,nothing,caches)
end

function DifferentiableTriangulation(
  trian :: SubCellTriangulation,
  cutgeo :: EmbeddedDiscretization,
  in_or_out :: CutInOrOut,
  name :: String
)
  geo = get_geometry(cutgeo.geo,name)
  return DifferentiableTriangulation(trian,cutgeo,in_or_out,geo)
end

function DifferentiableTriangulation(
  trian :: SubCellTriangulation,
  cutgeo :: EmbeddedDiscretization
)
  return DifferentiableTriangulation(trian,cutgeo,PHYSICAL_IN[1],cutgeo.geo)
end

function DifferentiableTriangulation(
  trian :: SubCellTriangulation,
  cutgeo :: EmbeddedDiscretization,
  in_or_out :: CutInOrOut
)
  return DifferentiableTriangulation(trian,cutgeo,in_or_out,cutgeo.geo)
end

function DifferentiableTriangulation(
  trian :: SubCellTriangulation,
  cutgeo :: EmbeddedDiscretization,
  in_or_out :: Tuple,
  args...
)
  return DifferentiableTriangulation(trian,cutgeo,in_or_out[1],args...)
end

function DifferentiableTriangulation(
  trian :: SubCellTriangulation,
  cutgeo :: EmbeddedDiscretization,
  name :: String
)
  geo = get_geometry(cutgeo.geo,name)
  return DifferentiableTriangulation(trian,cutgeo,PHYSICAL_IN,geo)
end

function DifferentiableTriangulation(
  trian :: SubCellTriangulation,
  cutgeo :: EmbeddedDiscretization,
  geo :: DiscreteGeometryFromFEFunction
)
  return DifferentiableTriangulation(trian,cutgeo,PHYSICAL_IN,geo)
end

function DifferentiableTriangulation(
  cutgeo :: EmbeddedDiscretization,
  args...
)
  trian = Triangulation(cutgeo,args...)
  return DifferentiableTriangulation(trian,cutgeo,args...)
end

function DifferentiableEmbeddedBoundary(
  trian :: SubFacetTriangulation,
  cutgeo :: EmbeddedDiscretization,
  geo :: DiscreteGeometryFromFEFunction
)
  cut_bgcells,ls_to_bgcell_iscut = precompute_caches(cutgeo)
  mask, orientation = precompute_subfacet_mask_and_orientation(trian,cutgeo,geo)
  caches = (;cut_bgcells,ls_to_bgcell_iscut,mask,orientation)
  return DifferentiableCutPolyTriangulation(trian,cutgeo,nothing,nothing,nothing,caches)
end

function DifferentiableEmbeddedBoundary(
  trian :: SubFacetTriangulation,
  cutgeo :: EmbeddedDiscretization,
  name :: String
)
  geo = get_geometry(cutgeo.geo,name)
  return DifferentiableEmbeddedBoundary(trian,cutgeo,geo)
end

function DifferentiableEmbeddedBoundary(
  trian :: SubFacetTriangulation,
  cutgeo :: EmbeddedDiscretization
)
  return DifferentiableEmbeddedBoundary(trian,cutgeo,cutgeo.geo)
end

function DifferentiableEmbeddedBoundary(
  trian :: SubFacetTriangulation,
  cutgeo :: EmbeddedDiscretization,
  geo1 :: DiscreteGeometryFromFEFunction,
  geo2 :: DiscreteGeometryFromFEFunction
)
  cut_bgcells,ls_to_bgcell_iscut = precompute_caches(cutgeo)
  mask, orientation = precompute_subfacet_mask_and_orientation(trian,cutgeo,geo1,geo2)
  caches = (;cut_bgcells,ls_to_bgcell_iscut,mask,orientation)
  return DifferentiableCutPolyTriangulation(trian,cutgeo,nothing,nothing,nothing,caches)
end

function DifferentiableEmbeddedBoundary(
  trian :: SubFacetTriangulation,
  cutgeo :: EmbeddedDiscretization,
  name1 :: String,
  name2 :: String
)
  geo1 = get_geometry(cutgeo.geo,name1)
  geo2 = get_geometry(cutgeo.geo,name2)
  return DifferentiableEmbeddedBoundary(trian,cutgeo,geo1,geo2)
end

function DifferentiableEmbeddedBoundary(
  cutgeo :: EmbeddedDiscretization,
  args...
)
  trian = EmbeddedBoundary(cutgeo,args...)
  return DifferentiableEmbeddedBoundary(trian,cutgeo,args...)
end

# Update cell values

(t::DifferentiableCutPolyTriangulation)(φh) = update_trian!(t,get_fe_space(φh),φh)

update_trian!(trian::Triangulation,U,φh) = trian

function update_trian!(trian::DifferentiableCutPolyTriangulation,space::FESpace,uh)
  i = findfirst(Base.Fix2(===,space),trian.fe_space)
  if isnothing(i)
    return trian
  end
  uh_cell_maps = extract_dualized_cell_maps(uh)
  tree = get_tree(trian.cutgeo.geo)
  φh_s, _ = _find_unique_leaves(tree);
  ls_cell_maps = get_data.(φh_s)
  cell_map = [ls_cell_maps[1:i-1]..., uh_cell_maps, ls_cell_maps[i+1:end]...];

  cut_bgcells = trian.caches.cut_bgcells
  cut_bgcell_to_ls_maps = map(a->lazy_map(Reindex(a),cut_bgcells),cell_map);
  trian.cell_maps = cut_bgcell_to_ls_maps
  return trian
end

function update_trian!(trian::DifferentiableCutPolyTriangulation,::FESpace,::Nothing)
  trian.cell_maps = nothing
  trian.cached_rcoords = nothing
  trian.cached_normals = nothing
  return trian
end

function update_trian!(trian::DifferentiableCutPolyTriangulation,space::MultiFieldFESpace,uh)
  Us = space.spaces
  i_s = indexin(trian.fe_space,Us)
  _in = findall(!isnothing,i_s)
  _missing = findall(isnothing,i_s)
  if isempty(_in)
    return trian
  end
  uhs = map(Base.Fix1(getindex,uh),i_s[_in])
  in_cell_maps = map(extract_dualized_cell_maps,uhs);
  tree = get_tree(trian.cutgeo.geo)
  φh_s, _ = _find_unique_leaves(tree);
  ls_cell_maps = get_data.(φh_s)
  missing_ls_cell_maps = ls_cell_maps[_missing]
  perm = sortperm([i_s[_in];_missing])
  cell_map = [in_cell_maps;missing_ls_cell_maps][perm]

  cut_bgcells = trian.caches.cut_bgcells
  cut_bgcell_to_ls_maps = map(a->lazy_map(Reindex(a),cut_bgcells),cell_map);
  trian.cell_maps = cut_bgcell_to_ls_maps
  return trian
end

function update_trian!(trian::DifferentiableCutPolyTriangulation,::MultiFieldFESpace,::Nothing)
  trian.cell_maps = nothing
  trian.cached_rcoords = nothing
  trian.cached_normals = nothing
  return trian
end

function extract_dualized_cell_maps(
  φh::CellField
)
  @assert isa(DomainStyle(φh),ReferenceDomain)
  bgcell_to_fields = get_data(φh)
  return bgcell_to_fields
end

# Autodiff

function FESpaces._change_argument(
  op,f,trian::DifferentiableCutPolyTriangulation,uh
)
  U = get_fe_space(uh)
  function g(cell_u)
    cf = CellField(U,cell_u)
    update_trian!(trian,U,cf)
    cell_grad = f(cf)
    update_trian!(trian,U,nothing)
    get_contribution(cell_grad,trian)
  end
  g
end

function FESpaces._compute_cell_ids(uh,ttrian::DifferentiableCutPolyTriangulation)
  FESpaces._compute_cell_ids(uh,ttrian.trian)
end

function Geometry.get_background_model(t::DifferentiableCutPolyTriangulation)
  get_background_model(t.trian)
end

function Geometry.get_grid(t::DifferentiableCutPolyTriangulation)
  get_grid(t.trian)
end

function Geometry.get_cell_reffe(t::DifferentiableCutPolyTriangulation)
  get_cell_reffe(t.trian)
end

function Geometry.get_cell_map(ttrian::DifferentiableCutPolyTriangulation)
  if isnothing(ttrian.cell_maps) || isempty(ttrian.cell_maps) || iszero(num_cells(ttrian))
    return get_cell_map(ttrian.trian)
  end
  cutgeo = ttrian.cutgeo
  c = ttrian.caches
  isnothing(ttrian.cached_rcoords) && cut_and_cache!(ttrian,c)
  subcell_to_rcoords = ttrian.cached_rcoords
  cell_map_r2p = get_cell_map(get_background_model(ttrian))
  cell_to_rcoords = lazy_map(Reindex(subcell_to_rcoords),c.mask)
  subcell_ref_to_phys_map = lazy_map(Reindex(cell_map_r2p),cutgeo.subcells.cell_to_bgcell);
  cell_to_ref_to_phys_map = lazy_map(Reindex(subcell_ref_to_phys_map),c.mask);
  cell_to_coords = lazy_map(evaluate,cell_to_ref_to_phys_map,cell_to_rcoords);

  cell_reffe = get_cell_reffe(ttrian)
  cell_map = compute_cell_maps(cell_to_coords,cell_reffe)
  return cell_map
end

function Geometry.get_facet_normal(
  ttrian::DifferentiableCutPolyTriangulation{Dc,Dp,<:SubFacetTriangulation}
) where {Dc,Dp}
  if isnothing(ttrian.cell_maps) || isempty(ttrian.cell_maps) || iszero(num_cells(ttrian))
    return get_facet_normal(ttrian.trian)
  end
  c = ttrian.caches
  isnothing(ttrian.cached_normals) && cut_and_cache!(ttrian,c)
  facet_to_normal = ttrian.cached_normals
  facet_normals = facet_to_normal[c.mask].*c.orientation
  return lazy_map(constant_field,facet_normals)
end

function Geometry.get_glue(ttrian::DifferentiableCutPolyTriangulation,val::Val{D}) where {D}
  glue = get_glue(ttrian.trian,val)
  if isnothing(glue) || isnothing(ttrian.cell_maps) || isempty(ttrian.cell_maps) || iszero(num_cells(ttrian))
    return glue
  end
  c = ttrian.caches
  isnothing(ttrian.cached_rcoords) && cut_and_cache!(ttrian,c)
  # New reference maps
  subcell_to_rcoords = ttrian.cached_rcoords
  cell_to_rcoords = lazy_map(Reindex(subcell_to_rcoords),c.mask)
  cell_reffe = get_cell_reffe(ttrian)
  ref_cell_map = compute_cell_maps(cell_to_rcoords,cell_reffe)
  return FaceToFaceGlue(
    glue.tface_to_mface,
    ref_cell_map,
    glue.mface_to_tface,
  )
end

function cut_and_cache!(ttrian::DifferentiableCutPolyTriangulation,cache)
    cell_maps = ttrian.cell_maps
    subcell_to_rcoords, _ = cut_and_get_rcoords(ttrian.trian,cell_maps,cache.ls_to_bgcell_iscut,cache.cut_bgcells);
    ttrian.cached_rcoords = subcell_to_rcoords
    nothing
end

function cut_and_cache!(ttrian::DifferentiableCutPolyTriangulation{Dc,Dp,<:SubFacetTriangulation},cache) where {Dc,Dp}
    cell_maps = ttrian.cell_maps
    subcell_to_rcoords, facet_to_normal = cut_and_get_rcoords(ttrian.trian,cell_maps,cache.ls_to_bgcell_iscut,cache.cut_bgcells);
    ttrian.cached_rcoords = subcell_to_rcoords
    ttrian.cached_normals = facet_to_normal
    nothing
end

function Geometry.is_change_possible(
  strian::A,ttrian::DifferentiableCutPolyTriangulation{Dc,Dp,A}
) where {Dc,Dp,A <: Union{SubCellTriangulation,SubFacetTriangulation}}
  return strian === ttrian.trian
end

function Geometry.best_target(
  strian::A,ttrian::DifferentiableCutPolyTriangulation{Dc,Dp,A}
) where {Dc,Dp,A <: Union{SubCellTriangulation,SubFacetTriangulation}}
  return ttrian
end

for tdomain in (:ReferenceDomain,:PhysicalDomain)
  for sdomain in (:ReferenceDomain,:PhysicalDomain)
    @eval begin
      function CellData.change_domain(
        a::CellField,strian::A,::$sdomain,ttrian::DifferentiableCutPolyTriangulation{Dc,Dp,A},::$tdomain
      ) where {Dc,Dp,A <: Union{SubCellTriangulation,SubFacetTriangulation}}
        @assert is_change_possible(strian,ttrian)
        b = change_domain(a,$(tdomain)())
        return CellData.similar_cell_field(a,CellData.get_data(b),ttrian,$(tdomain)())
      end
    end
  end
end

function FESpaces.get_cell_fe_data(fun,f,ttrian::DifferentiableCutPolyTriangulation)
  FESpaces.get_cell_fe_data(fun,f,ttrian.trian)
end

function cut_and_get_rcoords(ttrian,cut_bgcell_to_ls_maps,ls_to_bgcell_iscut,cut_bgcells)
  bgmodel = get_background_model(ttrian)
  D = num_dims(bgmodel)
  bg_vertices = get_cell_ref_coordinates(bgmodel)
  bg_ctypes = get_cell_type(bgmodel)
  bgcell_to_polys = expand_cell_data(get_polytopes(bgmodel),bg_ctypes)
  bgcell_to_graph = lazy_map(pref_to_gen_polytope_graph,bgcell_to_polys);

  cut_bgcell_graph = lazy_map(Reindex(bgcell_to_graph),cut_bgcells);
  cut_bgcell_vertices = lazy_map(Reindex(bg_vertices),cut_bgcells);
  cell_map_r2p = get_cell_map(bgmodel)
  cell_ref_to_phys_map = lazy_map(Reindex(cell_map_r2p),cut_bgcells)
  Nc = length(cut_bgcells)
  N = length(cut_bgcell_to_ls_maps)

  pcutter = lazy_map(
    PolytopalCellCutterMap{D,N,Nc}(),
    cut_bgcell_vertices,
    cut_bgcell_graph,
    cell_ref_to_phys_map,
    cut_bgcell_to_ls_maps...
  );

  v = testitem(pcutter)
  T = eltype(eltype(eltype(first(first(v)))))
  get_cell_to_rcoords(T,ttrian,bgmodel,ls_to_bgcell_iscut,pcutter);
end

function get_cell_to_rcoords(
  ::Type{T},
  ::SubCellTriangulation{D},
  bgmodel,
  ls_to_bgcell_iscut,
  pcutter
) where {T,D}
  pcache = array_cache(pcutter);
  cell_point_to_rcoords = allocate_cell_point_to_rcoords(T,bgmodel,ls_to_bgcell_iscut);
  expected_cell_simplex = ReferenceFEs.simplex_polytope(Val(D))
  simplex_num_verts = num_vertices(expected_cell_simplex)
  subcell_node_count = 0
  for i in eachindex(pcutter)
    cache = getindex!(pcache,pcutter,i)
    cell_rvertices = get_cell_rvertices(cache)
    num_new_cells = length(cell_rvertices)
    num_new_cell_nodes = num_new_cells*simplex_num_verts
    set_points_from_coords!(cell_point_to_rcoords,cell_rvertices,subcell_node_count,simplex_num_verts)
    subcell_node_count += num_new_cell_nodes
  end
  resize!(cell_point_to_rcoords,subcell_node_count)

  cell_to_points_data = Base.OneTo(subcell_node_count+simplex_num_verts);
  cell_to_points_prts = 1:simplex_num_verts:subcell_node_count+simplex_num_verts;
  cell_to_points = Table(collect(cell_to_points_data),collect(cell_to_points_prts));

  return lazy_map(Broadcasting(Reindex(cell_point_to_rcoords)),cell_to_points), nothing
end

function get_cell_to_rcoords(
  ::Type{T},
  ::SubFacetTriangulation{D},
  bgmodel,
  ls_to_bgcell_iscut,
  pcutter
) where {T,D}
  pcache = array_cache(pcutter);
  facet_point_to_rcoords, facet_to_normal = allocate_face_point_to_rcoords_and_normals(T,bgmodel,ls_to_bgcell_iscut);
  expected_facet_simplex = ReferenceFEs.simplex_polytope(Val(D))
  facet_simplex_num_verts = num_vertices(expected_facet_simplex)
  subfacet_count = 0
  subfacet_node_count = 0
  for i in eachindex(pcutter)
    cache  = getindex!(pcache,pcutter,i)
    subfacet_normals = get_subfacet_normals(cache)
    facet_rvertices = get_facet_rvertices(cache)
    num_new_facets = length(facet_rvertices)
    num_new_facets_nodes = num_new_facets*facet_simplex_num_verts
    set_points_from_coords!(facet_point_to_rcoords,facet_rvertices,subfacet_node_count,facet_simplex_num_verts)
    my_setindex!(facet_to_normal,subfacet_normals,subfacet_count+1,subfacet_count+num_new_facets)
    subfacet_count += num_new_facets
    subfacet_node_count += num_new_facets_nodes
  end
  resize!(facet_point_to_rcoords,subfacet_node_count)
  resize!(facet_to_normal,subfacet_count)

  facet_to_points_data = Base.OneTo(subfacet_node_count+facet_simplex_num_verts)
  facet_to_points_prts = 1:facet_simplex_num_verts:subfacet_node_count+facet_simplex_num_verts
  facet_to_points = Table(collect(facet_to_points_data),collect(facet_to_points_prts))

  return lazy_map(Broadcasting(Reindex(facet_point_to_rcoords)),facet_to_points), facet_to_normal
end

function allocate_cell_point_to_rcoords(::Type{T},bgmodel::DiscreteModel{D},ls_to_bgcell_iscut) where {T,D}
  polys = get_polytopes(bgmodel);
  p_ref = first(polys);
  @check isone(length(polys)) "Only models with a single reference polytope are currently supported."
  @check is_simplex(p_ref) "Only simplex background meshes are currently supported."
  nv = num_vertices(p_ref)
  num_ls = length(ls_to_bgcell_iscut)
  num_cut = count(>(0), sum(ls_to_bgcell_iscut))
  max_subcells = num_cut * ifelse(D==3, 2*nv, nv) * 2^num_ls
  return Vector{Point{D,T}}(undef,nv*max_subcells)
end

function allocate_face_point_to_rcoords_and_normals(::Type{T},bgmodel::DiscreteModel{D},ls_to_bgcell_iscut) where {T,D}
  polys = get_polytopes(bgmodel);
  p_ref = first(polys);
  @check isone(length(polys)) "Only models with a single reference polytope are currently supported."
  @check is_simplex(p_ref) "Only simplex background meshes are currently supported."
  num_ls = length(ls_to_bgcell_iscut)
  num_cut = count(>(0), sum(ls_to_bgcell_iscut))
  nv = num_vertices(p_ref)
  max_cutnodes = num_cut * num_ls * max(1, nv - 4 + 2^num_ls) * D
  topo = get_grid_topology(bgmodel)
  restricted_topo,_ = Geometry.restrict(topo,findall(sum(ls_to_bgcell_iscut) .> 0))
  max_nodes = num_faces(restricted_topo,0) + max_cutnodes
  facet_point_to_rcoords = Vector{Point{D,T}}(undef,max_nodes)
  facet_to_normal = Vector{VectorValue{D,T}}(undef,max_nodes)
  return facet_point_to_rcoords, facet_to_normal
end

# Caches

function precompute_caches(cutgeo)
  bgmodel = get_background_model(cutgeo)
  cut_bgcells = unique(cutgeo.subcells.cell_to_bgcell)
  tree = get_tree(cutgeo.geo)
  φh_s, _ = _find_unique_leaves(tree);
  cell_values = map(Base.Fix1(extract_dualized_cell_values,bgmodel),φh_s)
  cell_val_cache = array_cache(first(cell_values))
  ls_to_bgcell_iscut = [zeros(Bool,num_cells(bgmodel)) for _ in 1:length(φh_s)];
  compute_ls_to_bgcell_iscut!(ls_to_bgcell_iscut,cell_values,cell_val_cache)
  return cut_bgcells,ls_to_bgcell_iscut
end

function precompute_subcell_mask(::SubCellTriangulation,cutgeo,in_or_out,geo)
  bgcell_to_inoutcut = compute_bgcell_to_inoutcut(cutgeo,geo)
  subcell_to_inoutcut = lazy_map(Reindex(bgcell_to_inoutcut),cutgeo.subcells.cell_to_bgcell)
  subcell_to_inout = compute_subcell_to_inout(cutgeo,geo)
  mask = lazy_map( (a,b) -> a==CUT && b==in_or_out.in_or_out, subcell_to_inoutcut, subcell_to_inout)
  return findall(mask)
end

function precompute_subfacet_mask_and_orientation(::SubFacetTriangulation,cutgeo,geo)
  function conversion(data)
    f,name,meta = data
    oid = objectid(f)
    ls = cutgeo.oid_to_ls[oid]
    cell_to_inoutcut = cutgeo.ls_to_subfacet_to_inout[ls]
    cell_to_inoutcut, name, meta
  end

  tree = get_tree(geo)
  newtree = replace_data(identity,conversion,tree)
  subfacet_to_inoutcut, orientation = compute_inoutboundary(newtree)
  newsubfacets = findall(subfacet_to_inoutcut .== INTERFACE)
  neworientation = orientation[newsubfacets]
  return newsubfacets, neworientation
end

function precompute_subfacet_mask_and_orientation(::SubFacetTriangulation,cutgeo,geo1,geo2)
  function conversion(data)
    f,name,meta = data
    oid = objectid(f)
    ls = cutgeo.oid_to_ls[oid]
    cell_to_inoutcut = cutgeo.ls_to_subfacet_to_inout[ls]
    cell_to_inoutcut, name, meta
  end

  tree1 = get_tree(geo1)
  tree2 = get_tree(geo2)
  newtree1 = replace_data(identity,conversion,tree1)
  newtree2 = replace_data(identity,conversion,tree2)
  subfacet_to_inoutcut1, orientation = compute_inoutboundary(newtree1)
  subfacet_to_inoutcut2 = compute_inoutcut(newtree2)
  mask = lazy_map( (i,j)->(i==INTERFACE) && (j==INTERFACE), subfacet_to_inoutcut1, subfacet_to_inoutcut2 )
  newsubfacets = findall( mask )
  neworientation = orientation[newsubfacets]
  return newsubfacets, neworientation
end

# TriangulationView

const DifferentiableCutPolyTriangulationView{Dc,Dp} = Geometry.TriangulationView{Dc,Dp,<:DifferentiableCutPolyTriangulation}

function DifferentiableTriangulation(
  trian :: Geometry.TriangulationView,
  cutgeo :: EmbeddedDiscretization,
  args...
)
  parent = DifferentiableTriangulation(trian.parent,cutgeo,args...)
  return Geometry.TriangulationView(parent,trian.cell_to_parent_cell)
end

function DifferentiableEmbeddedBoundary(
  trian :: Geometry.TriangulationView,
  cutgeo :: EmbeddedDiscretization,
  args...
)
  parent = DifferentiableEmbeddedBoundary(trian.parent,cutgeo,args...)
  return Geometry.TriangulationView(parent,trian.cell_to_parent_cell)
end

function get_differentiable_trian(trian::DifferentiableCutPolyTriangulationView)
  get_differentiable_trian(trian.parent)
end

function update_trian!(trian::Geometry.TriangulationView,U,φh)
  update_trian!(trian.parent,U,φh)
  return trian
end

function FESpaces._change_argument(
  op,f,trian::DifferentiableCutPolyTriangulationView,uh
)
  U = get_fe_space(uh)
  function g(cell_u)
    cf = CellField(U,cell_u)
    update_trian!(trian,U,cf)
    cell_grad = f(cf)
    update_trian!(trian,U,nothing)
    get_contribution(cell_grad,trian)
  end
  g
end

# AppendedTriangulation

const DifferentiableAppendedCutPolyTriangulation{Dc,Dp,A} =
  AppendedTriangulation{Dc,Dp,<:Union{<:DifferentiableCutPolyTriangulation,<:DifferentiableCutPolyTriangulationView{Dc,Dp}}}

function DifferentiableTriangulation(
  trian::AppendedTriangulation, cutgeo::EmbeddedDiscretization, in_or_out::Tuple, args...
)
  a = DifferentiableTriangulation(trian.a,cutgeo,in_or_out[1], args...)
  b = DifferentiableTriangulation(trian.b,cutgeo,in_or_out[2], args...)
  return AppendedTriangulation(a,b)
end

function DifferentiableTriangulation(
  trian::AppendedTriangulation, cutgeo::EmbeddedDiscretization, args...
)
  a = DifferentiableTriangulation(trian.a,cutgeo,args...)
  b = DifferentiableTriangulation(trian.b,cutgeo,args...)
  return AppendedTriangulation(a,b)
end

function DifferentiableEmbeddedBoundary(
  trian::AppendedTriangulation, cutgeo::EmbeddedDiscretization, in_or_out::Tuple, args...
)
  a = DifferentiableEmbeddedBoundary(trian.a,cutgeo,in_or_out[1], args...)
  b = DifferentiableEmbeddedBoundary(trian.b,cutgeo,in_or_out[2], args...)
  return AppendedTriangulation(a,b)
end

function DifferentiableEmbeddedBoundary(
  trian::AppendedTriangulation, cutgeo::EmbeddedDiscretization, args...
)
  a = DifferentiableEmbeddedBoundary(trian.a,cutgeo,args...)
  b = DifferentiableEmbeddedBoundary(trian.b,cutgeo,args...)
  return AppendedTriangulation(a,b)
end

function get_differentiable_trian(trian::DifferentiableAppendedCutPolyTriangulation)
  get_differentiable_trian(trian.a) # cuttrian is always stored in a, do we need a check here?
end

function update_trian!(trian::DifferentiableAppendedCutPolyTriangulation,U,φh)
  update_trian!(trian.a,U,φh)
  update_trian!(trian.b,U,φh)
  return trian
end

function FESpaces._change_argument(
  op,f,trian::DifferentiableAppendedCutPolyTriangulation,uh
)
  U = get_fe_space(uh)
  function g(cell_u)
    cf = CellField(U,cell_u)
    update_trian!(trian,U,cf)
    cell_grad = f(cf)
    update_trian!(trian,U,nothing)
    get_contribution(cell_grad,trian)
  end
  g
end

# get_cell_measure

# Allow us to handle differentiation for following example:
# function J(v)
#    meas_K1 = get_cell_measure(diffable_Ω2, Ω_bg, diffable_Γ23)
#    κ1 = CellField( meas_K1, Ω_bg)
#    ∫( κ1 )dΓ23
# end
# In this case, using the usual get_cell_measure attempts to use the cell_maps
# from diffable_Ω2, which have not been set because update_trian! was applied
# to diffable_Γ23 in _change_argument. Thus, we have to manually set the cell
# maps in diffable_Ω2 to those from diffable_Γ23.

function CellData.get_cell_measure(strian::Triangulation, ttrian::Triangulation, itrian::Triangulation)
  dstrian = get_differentiable_trian(strian)
  ditrian = get_differentiable_trian(itrian)
  if isnothing(dstrian) || isnothing(ditrian)
    return get_cell_measure(strian,ttrian)
  end
  dstrian.cell_maps = ditrian.cell_maps # Use cell_maps from active integration triangulation
  scell_measure = get_cell_measure(strian)
  dstrian.cell_maps = nothing # Reset cell_maps
  move_contributions(scell_measure,strian,ttrian) |> collect
end