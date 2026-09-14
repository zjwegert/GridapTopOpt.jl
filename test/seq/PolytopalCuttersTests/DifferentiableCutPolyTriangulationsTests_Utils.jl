using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.ReferenceFEs, Gridap.CellData, Gridap.Fields, Gridap.Arrays, Gridap.Helpers, Gridap.FESpaces
using GridapEmbedded, GridapEmbedded.LevelSetCutters, GridapEmbedded.Interfaces, GridapEmbedded.CSG
using GridapEmbedded.Interfaces: get_ghost_normal_vector, get_subfacet_normal_vector, get_conormal_vector
using Gridap.Arrays: getindex!
using Gridap.Geometry: FaceToFaceGlue, UnstructuredGrid, UnstructuredDiscreteModel, CompositeTriangulation
using Gridap.Geometry: UnstructuredGridTopology, get_grid_topology, num_faces, get_node_coordinates
using Gridap.Geometry: FaceLabeling, num_cell_dims, get_reffes, get_cell_type, expand_cell_data
using Gridap.ReferenceFEs: LagrangianRefFE, simplex_polytope, get_shapefuns
using Gridap.Fields: constant_field, inverse_map, GenericField
using Gridap.CellData: GenericCellField, ReferenceDomain, get_normal_vector
using GridapTopOpt
using FiniteDiff
using Test
using Random
using AbstractTrees: PreOrderDFS

function generate_model(D,n)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  base_model = UnstructuredDiscreteModel((CartesianDiscreteModel(domain,cell_partition)))
  ref_model = refine(base_model, refinement_method = "barycentric")
  return ref_model.model
end

################################################################
#               Methods for analytic computation               #
################################################################
# Please note, these methods are not intended to be general-purpose
# and serve only to validate the analytic computations & AD

# Fix degenerate facet normals
function GridapEmbedded.Interfaces.merge_sub_face_data(ls_to_subfacets, ls_to_ls_to_facet_to_inout)
  fst = empty(first(ls_to_subfacets))
  ls_to_facet_to_inout = [empty(first(ls_to_f_to_i)) for ls_to_f_to_i in ls_to_ls_to_facet_to_inout]
  for (i, fst_i) in enumerate(ls_to_subfacets)
    non_degen = findall(j -> !iszero(norm(fst_i.facet_to_normal[j])),
                        eachindex(fst_i.facet_to_normal))
    if length(non_degen) < length(fst_i.facet_to_normal)
      fst_i        = GridapEmbedded.Interfaces.SubFacetData(fst_i, non_degen, fill(Int8(1), length(non_degen)))
      ls_to_f_to_i = [v[non_degen] for v in ls_to_ls_to_facet_to_inout[i]]
    else
      ls_to_f_to_i = ls_to_ls_to_facet_to_inout[i]
    end
    append!(fst, fst_i)
    for (j, facet_to_inout) in enumerate(ls_to_f_to_i)
      append!(ls_to_facet_to_inout[j], facet_to_inout)
    end
  end
  fst, ls_to_facet_to_inout
end

# consistent_facet_to_points for multiple lsfs
function GridapEmbedded.Interfaces.consistent_facet_to_points(
  facet_to_points::Table, point_to_coords::Vector
)
  f(pt::VectorValue) = VectorValue(round.(pt.data;sigdigits=12))
  f(id::Integer) = f(point_to_coords[id])

  # Create a list of the unique points composing the facets
  npts = length(point_to_coords)
  nfaces = length(facet_to_points)
  touched = zeros(Bool,npts)
  for face in 1:nfaces
    pts = view(facet_to_points,face)
    touched[pts] .= true
  end
  touched_ids = findall(touched)
  unique_ids = unique(f,touched_ids)

  # Create a mapping from the old point ids to the new ones
  touched_to_uid = collect(Int32,indexin(f.(touched_ids),f.(unique_ids)))
  point_to_uid = extend(touched_to_uid,PosNegPartition(touched_ids,npts))

  facet_to_uids = Table(
    collect(Int32,lazy_map(Reindex(point_to_uid),facet_to_points.data)),
    facet_to_points.ptrs
  )

  # Filter out degenerate facets where ANY two vertices collapse to the same point
  # (not just when ALL vertices are the same)
  ids = map(allunique, facet_to_uids)
  return facet_to_uids[findall(ids)], unique_ids, findall(ids)
end

function GridapEmbedded.Interfaces.compute_active_model(trian::GridapEmbedded.Interfaces.SubFacetTriangulation)
  subgrid = trian.subgrid
  subfacets = trian.subfacets
  facet_to_uids, uid_to_point, ids = GridapEmbedded.Interfaces.consistent_facet_to_points(
    subfacets.facet_to_points, subfacets.point_to_coords
  )
  consistent_grid = UnstructuredGrid(
    subgrid.node_coordinates,
    subgrid.cell_node_ids[ids],
    subgrid.reffes,
    subgrid.cell_types[ids],
    subgrid.orientation_style,
    subgrid.facet_normal
  )
  topo = UnstructuredGridTopology(
    consistent_grid, facet_to_uids, uid_to_point
  )
  return UnstructuredDiscreteModel(consistent_grid,topo,FaceLabeling(topo))
end

# Override SkeletonTriangulation for SubFacetTriangulation so that CompositeTriangulation
# glue composition is correct when degenerate facets have been filtered.
#
# Root issue: compute_active_model produces a face_model with M filtered cells (1..M),
# but the original face_trian has N cells (1..N).  CompositeTriangulation composes
# glues via  rface_to_mface[dface_to_rface]  where rface_to_mface =
# face_trian.subfacets.facet_to_bgcell (N entries) and dface_to_rface contains
# face_model cell indices (1..M).  The fix: build a filtered SubFacetTriangulation
# (effective_trian) with M entries so its facet_to_bgcell[k] ==
# original_facet_to_bgcell[ids[k]], aligning 1-to-1 with face_model's M cells.
function Gridap.Geometry.SkeletonTriangulation(
  face_trian::GridapEmbedded.Interfaces.SubFacetTriangulation
)
  bgmodel    = get_background_model(face_trian)
  face_model = get_active_model(face_trian)

  # Re-derive the non-degenerate facet indices directly from subfacets —
  # the same computation used by compute_active_model, so effective_trian
  # has exactly M cells aligned 1-to-1 with face_model.
  _, _, ids = GridapEmbedded.Interfaces.consistent_facet_to_points(
    face_trian.subfacets.facet_to_points, face_trian.subfacets.point_to_coords
  )

  effective_trian = if length(ids) == length(face_trian.subfacets.facet_to_bgcell)
    face_trian   # no degenerate facets filtered, original indexing is fine
  else
    filtered_sf = GridapEmbedded.Interfaces.SubFacetData(
      face_trian.subfacets, ids, fill(Int8(1), length(ids))
    )
    GridapEmbedded.Interfaces.SubFacetTriangulation(filtered_sf, bgmodel)
  end

  ghost_mask    = GridapEmbedded.Interfaces.get_ghost_mask(effective_trian, face_model)
  face_skeleton = SkeletonTriangulation(face_model, ghost_mask)
  cell_skeleton = CompositeTriangulation(effective_trian, face_skeleton)
  ghost_skeleton = GridapEmbedded.Interfaces.generate_ghost_trian(cell_skeleton, bgmodel)

  ctrian_plus  = CompositeTriangulation(effective_trian, face_skeleton.plus)
  ctrian_minus = CompositeTriangulation(effective_trian, face_skeleton.minus)
  isign_plus   = GridapEmbedded.Interfaces.get_interface_sign(
    ctrian_plus, effective_trian, ghost_skeleton.plus)
  isign_minus  = GridapEmbedded.Interfaces.get_interface_sign(
    ctrian_plus, effective_trian, ghost_skeleton.minus)

  plus  = GridapEmbedded.Interfaces.CutFaceBoundaryTriangulation(
    face_model, effective_trian, ctrian_plus,
    face_skeleton.plus,  ghost_skeleton.plus,  isign_plus)
  minus = GridapEmbedded.Interfaces.CutFaceBoundaryTriangulation(
    face_model, effective_trian, ctrian_minus,
    face_skeleton.minus, ghost_skeleton.minus, isign_minus)
  return SkeletonTriangulation(plus, minus)
end

# generate_ghost_trian for the FaceToFaceGlue (boundary) case.
# The original asserts length(bfaces)==1, which fails for corner bgcells (2 boundary faces).
# We take first(bfaces) to handle corners.
# Note: BoundaryTriangulation pre-filters to ∂D-touching vertices, so bfaces is always non-empty.
function GridapEmbedded.Interfaces.generate_ghost_trian(
  trian::CompositeTriangulation, bgmodel, cell_glue::FaceToFaceGlue
)
  Dc = num_cell_dims(bgmodel)
  topo = get_grid_topology(bgmodel)
  face_to_cell = Gridap.Geometry.get_faces(topo, Dc-1, Dc)
  cell_to_face = Gridap.Geometry.get_faces(topo, Dc, Dc-1)
  is_boundary(f) = isone(length(view(face_to_cell, f)))

  n_faces = num_cells(trian)
  ghost_faces = zeros(Int32, n_faces)
  for (i, cell) in enumerate(cell_glue.tface_to_mface)
    bfaces = filter(is_boundary, view(cell_to_face, cell))
    ghost_faces[i] = Int32(first(bfaces))
  end
  return BoundaryTriangulation(bgmodel, ghost_faces)
end

# Fix for BoundaryTriangulation:
# 1. The original builds CompositeTriangulation(face_trian, face_boundary) where face_trian has N cells
#    but face_model (and thus face_boundary) has M filtered cells — fixed via effective_trian.
# 2. BoundaryTriangulation(face_model) returns all topological boundary facets, including
#    level-set intersection facets interior to D.  We restrict to facets on ∂D.
#    In 2D: face_model is 1D; facets = vertices (0-faces); a vertex is on ∂D iff its
#            one incident edge's bgcell has ≥1 boundary bg-face.
#    In 3D: face_model is 2D; facets = edges (1-faces); an edge is on ∂D iff its
#            one incident triangle's bgcell has ≥1 boundary bg-face.
#    General: a (Df-1)-facet of face_model is on ∂D iff its one incident Df-cell's bgcell
#             has ≥1 boundary bg-face.
function Gridap.Geometry.BoundaryTriangulation(
  face_trian::GridapEmbedded.Interfaces.SubFacetTriangulation
)
  bgmodel    = get_background_model(face_trian)
  face_model = get_active_model(face_trian)

  _, _, ids = GridapEmbedded.Interfaces.consistent_facet_to_points(
    face_trian.subfacets.facet_to_points, face_trian.subfacets.point_to_coords
  )

  effective_trian = if length(ids) == length(face_trian.subfacets.facet_to_bgcell)
    face_trian
  else
    filtered_sf = GridapEmbedded.Interfaces.SubFacetData(
      face_trian.subfacets, ids, fill(Int8(1), length(ids))
    )
    GridapEmbedded.Interfaces.SubFacetTriangulation(filtered_sf, bgmodel)
  end

  # Identify topological boundary facets of face_model that lie on ∂D.
  # A boundary facet (codim-1 face of face_model) is on ∂D iff its one incident
  # face_model cell's bgcell has at least one boundary face of bgmodel.
  Dc = num_cell_dims(bgmodel)
  bg_topo         = get_grid_topology(bgmodel)
  face_to_cell_bg = Gridap.Geometry.get_faces(bg_topo, Dc-1, Dc)
  cell_to_face_bg = Gridap.Geometry.get_faces(bg_topo, Dc, Dc-1)
  is_bgbdy(f)      = isone(length(view(face_to_cell_bg, f)))
  bgcell_on_bdy(c) = any(f -> is_bgbdy(f), view(cell_to_face_bg, c))

  Df = num_cell_dims(face_model)        # 1 in 2D, 2 in 3D
  face_topo          = get_grid_topology(face_model)
  fm_facet_to_cells  = Gridap.Geometry.get_faces(face_topo, Df-1, Df)  # (Df-1)-facet → Df-cells
  is_bdy_fm_facet    = get_isboundary_face(face_topo, Df-1)
  facet_to_bgcell    = effective_trian.subfacets.facet_to_bgcell

  n_fm_facets    = num_faces(face_model, Df-1)   # == num_facets(face_model)
  is_on_phys_bdy = fill(false, n_fm_facets)
  for f in 1:n_fm_facets
    is_bdy_fm_facet[f] || continue
    # Topological boundary facet → exactly one incident cell in face_model
    c = first(fm_facet_to_cells[f])
    is_on_phys_bdy[f] = bgcell_on_bdy(facet_to_bgcell[c])
  end

  face_boundary  = BoundaryTriangulation(face_model, is_on_phys_bdy, fill(Int8(1), num_facets(face_model)))
  cell_boundary  = CompositeTriangulation(effective_trian, face_boundary)
  ghost_boundary = GridapEmbedded.Interfaces.generate_ghost_trian(cell_boundary, bgmodel)
  interface_sign = GridapEmbedded.Interfaces.get_interface_sign(
    cell_boundary, effective_trian, ghost_boundary)

  return GridapEmbedded.Interfaces.CutFaceBoundaryTriangulation(
    face_model, effective_trian, cell_boundary, face_boundary, ghost_boundary, interface_sign
  )
end

#######################
# IntersectionTriangulation
#######################

struct IntersectionData{Dp,T}
  point_to_coords::Vector{Point{Dp,T}}
  point_to_rcoords::Vector{Point{Dp,T}}
  cell_to_points::Table{Int32,Vector{Int32},Vector{Int32}}
  cell_to_bgcell::Vector{Int32}
  cell_to_normal_1::Vector{Point{Dp,T}}
  cell_to_normal_2::Vector{Point{Dp,T}}
  cell_to_tangent_1::Vector{Point{Dp,T}}
  cell_to_tangent_2::Vector{Point{Dp,T}}
end

"""
    IntersectionTriangulation{Dc,Dp,T,A}

A triangulation representing the intersection of two EmbeddedBoundaries.
- In 2D (Dp=2): The intersection is a set of vertices (Dc=0)
- In 3D (Dp=3): The intersection is a set of line segments (Dc=1)

Stores both SubFacetTriangulations and provides normals for each boundary.
"""

struct IntersectionTriangulation{Dc,Dp,T,A} <: Triangulation{Dc,Dp}
  Γ1::GridapEmbedded.Interfaces.SubFacetTriangulation
  Γ2::GridapEmbedded.Interfaces.SubFacetTriangulation
  IntData::IntersectionData{Dp,T}
  bgmodel::A
  grid::UnstructuredGrid{Dc,Dp,T,NonOriented}
end

function IntersectionTriangulation(
  Γ1::GridapEmbedded.Interfaces.SubFacetTriangulation{Dc,Dp},
  Γ2::GridapEmbedded.Interfaces.SubFacetTriangulation{Dc,Dp}
) where {Dc,Dp}
  @assert Dp == Dc + 1 "Boundaries must be codimension-1"
  T = Float64
  bgmodel = get_background_model(Γ1)

  # Find intersection points from active models (approximate matching)
  _rnd(pt) = VectorValue(round.(pt.data; sigdigits=12))
  model_Γ1 = get_active_model(Γ1)
  model_Γ2 = get_active_model(Γ2)
  pts1 = model_Γ1.grid_topology.vertex_coordinates
  pts2 = model_Γ2.grid_topology.vertex_coordinates
  rnd2 = Set(_rnd.(pts2))
  point_to_coords = [pt for pt in pts1 if _rnd(pt) ∈ rnd2]
  unique!(_rnd, point_to_coords)

  if isempty(point_to_coords)
    return _empty_intersection(Γ1, Γ2, bgmodel, Val(Dp))
  end

  # Build cell connectivity (0D cells in 2D, 1D cells in 3D)
  cell_to_points = _build_intersection_cells(point_to_coords, model_Γ1, model_Γ2, Val(Dp))

  # Find bgcells, normals, and tangents from both boundaries per cell
  cell_to_bgcell, cell_to_normal_1, cell_to_normal_2, cell_to_tangent_1, cell_to_tangent_2 =
    _find_cell_bgcells_and_normals(Γ1.subfacets, Γ2.subfacets, point_to_coords, cell_to_points)

  # Reference coordinates from Γ1
  point_to_rcoords = _find_intersection_rcoords(Γ1.subfacets, point_to_coords)

  grid = _create_intersection_grid(point_to_coords, cell_to_points, Val(Dp))

  idata = IntersectionData{Dp,T}(
    point_to_coords, point_to_rcoords, cell_to_points,
    cell_to_bgcell, cell_to_normal_1, cell_to_normal_2,
    cell_to_tangent_1, cell_to_tangent_2
  )

  Dc_int = Dp - 2
  IntersectionTriangulation{Dc_int,Dp,T,typeof(bgmodel)}(
    Γ1, Γ2, idata, bgmodel, grid
  )
end

# --- Empty intersection ---

function _empty_intersection(Γ1, Γ2, bgmodel, ::Val{Dp}) where Dp
  T = Float64; Dc = Dp - 2
  pts = Point{Dp,T}[]
  cell_to_points = Table(Int32[], Int32[1])
  reffe = LagrangianRefFE(T, simplex_polytope(Val(Dc)), 1)
  grid = UnstructuredGrid(pts, cell_to_points, [reffe], Int8[])
  idata = IntersectionData{Dp,T}(pts, pts, cell_to_points, Int32[], pts, pts, pts, pts)
  IntersectionTriangulation{Dc,Dp,T,typeof(bgmodel)}(
    Γ1, Γ2, idata, bgmodel, grid
  )
end

# --- Cell connectivity ---

# 2D: each intersection point is a 0D cell
function _build_intersection_cells(pts::Vector{<:Point{2}}, model1, model2, ::Val{2})
  n = length(pts)
  Table(collect(Int32, 1:n), collect(Int32, 1:(n+1)))
end

# 3D: edges from both active models whose both endpoints are intersection points
function _build_intersection_cells(pts::Vector{<:Point{3}}, model_Γ1, model_Γ2, ::Val{3})
  _rnd(pt) = VectorValue(round.(pt.data; sigdigits=12))
  rnd_to_idx = Dict(_rnd(pt) => Int32(i) for (i, pt) in enumerate(pts))
  seen = Set{Tuple{Int32,Int32}}()

  data = Int32[]
  ptrs = Int32[1]
  for model in (model_Γ1, model_Γ2)
    edge_to_verts = Gridap.Geometry.get_faces(model.grid_topology, 1, 0)
    vtx_coords = model.grid_topology.vertex_coordinates
    for eid in 1:length(edge_to_verts)
      v = view(edge_to_verts, eid)
      r1, r2 = _rnd(vtx_coords[v[1]]), _rnd(vtx_coords[v[2]])
      if haskey(rnd_to_idx, r1) && haskey(rnd_to_idx, r2)
        i1, i2 = rnd_to_idx[r1], rnd_to_idx[r2]
        edge_key = minmax(i1, i2)
        if edge_key ∉ seen
          push!(seen, edge_key)
          push!(data, i1, i2)
          push!(ptrs, length(data) + 1)
        end
      end
    end
  end
  Table(data, ptrs)
end

# --- Per-cell bgcells, normals, and tangents ---

# Get bgcells from all subfacets touching a point
function _subfacet_bgcells_at_point(sf, pt)
  pt_ids = findall(x -> x ≈ pt, sf.point_to_coords)
  facet_ids = findall(x -> any(x .∈ (pt_ids,)), sf.facet_to_points)
  sf.facet_to_bgcell[facet_ids]
end

# Get the normal from a subfacet at a point, preferring subfacets in the given bgcell.
# Within a bgcell the level set is linear, so the surface normal is constant —
# using a subfacet from the wrong bgcell gives an inconsistent normal.
function _subfacet_normal_in_bgcell(sf, pt, bgcell)
  pt_ids = findall(x -> x ≈ pt, sf.point_to_coords)
  facet_ids = findall(x -> any(x .∈ (pt_ids,)), sf.facet_to_points)
  for fid in facet_ids
    sf.facet_to_bgcell[fid] == bgcell && return sf.facet_to_normal[fid]
  end
  sf.facet_to_normal[first(facet_ids)]  # fallback
end

# Conormal: projection of n2 onto the tangent plane of surface 1, normalized.
# This gives the outward conormal of ∂Γ1 at the intersection, pointing away from
# the intersection region (in the direction of increasing φ2 along Γ1).
# Works for any dimension: in 3D this lies in TΓ1 ⊥ to the intersection curve,
# in 2D it gives the correctly oriented tangent to Γ1.
function _compute_conormal(n1::VectorValue{D,T}, n2::VectorValue{D,T}) where {D,T}
  m = n2 - (n1 ⋅ n2) * n1
  nm = sqrt(m ⋅ m)
  nm > 0 ? m / nm : zero(m)
end

function _find_cell_bgcells_and_normals(sf1, sf2, point_to_coords, cell_to_points)
  Dp = length(eltype(point_to_coords))
  T = eltype(eltype(point_to_coords))
  n_cells = length(cell_to_points)

  cell_to_bgcell = Vector{Int32}(undef, n_cells)
  cell_to_n1 = Vector{Point{Dp,T}}(undef, n_cells)
  cell_to_n2 = Vector{Point{Dp,T}}(undef, n_cells)
  cell_to_t1 = Vector{Point{Dp,T}}(undef, n_cells)
  cell_to_t2 = Vector{Point{Dp,T}}(undef, n_cells)

  for i in 1:n_cells
    cell_pts = view(cell_to_points, i)

    # First pass: determine bgcell by intersecting across all cell points
    common_bg = nothing
    for pid in cell_pts
      pt = point_to_coords[pid]
      bgcells_1 = _subfacet_bgcells_at_point(sf1, pt)
      bgcells_2 = _subfacet_bgcells_at_point(sf2, pt)
      pt_bgcells = union(Set(bgcells_1), Set(bgcells_2))
      common_bg = common_bg === nothing ? pt_bgcells : intersect(common_bg, pt_bgcells)
    end
    bgcell = first(common_bg)
    cell_to_bgcell[i] = bgcell

    # Second pass: get normals from subfacets in the assigned bgcell
    pt = point_to_coords[first(cell_pts)]
    n1 = _subfacet_normal_in_bgcell(sf1, pt, bgcell)
    n2 = _subfacet_normal_in_bgcell(sf2, pt, bgcell)
    cell_to_n1[i] = n1
    cell_to_n2[i] = n2
    cell_to_t1[i] = _compute_conormal(n1, n2)
    cell_to_t2[i] = _compute_conormal(n2, n1)
  end

  return cell_to_bgcell, cell_to_n1, cell_to_n2, cell_to_t1, cell_to_t2
end

# --- Reference coordinates ---

function _find_intersection_rcoords(sf, point_to_coords)
  map(point_to_coords) do pt
    pt_ids = findall(x -> x ≈ pt, sf.point_to_coords)
    isempty(pt_ids) ? zero(pt) : sf.point_to_rcoords[first(pt_ids)]
  end
end

# --- Grid creation ---

function _create_intersection_grid(point_to_coords::Vector{Point{Dp,T}}, cell_to_points, ::Val{Dp}) where {Dp,T}
  Dc = Dp - 2
  reffe = LagrangianRefFE(T, simplex_polytope(Val(Dc)), 1)
  if isempty(point_to_coords) || isempty(cell_to_points)
    return UnstructuredGrid(point_to_coords, cell_to_points, [reffe], Int8[])
  end
  cell_types = fill(Int8(1), length(cell_to_points))
  UnstructuredGrid(point_to_coords, cell_to_points, [reffe], cell_types)
end

# --- Triangulation interface ---

Gridap.Geometry.get_background_model(t::IntersectionTriangulation) = t.bgmodel
Gridap.Geometry.get_grid(t::IntersectionTriangulation) = t.grid

function Gridap.Geometry.get_glue(t::IntersectionTriangulation{Dc,Dp}, ::Val{D}) where {Dc,Dp,D}
  D != Dp && return nothing
  d = t.IntData
  tface_to_mface = d.cell_to_bgcell

  # Compute reference coordinates via inverse map of assigned bgcells
  bg_trian = Triangulation(t.bgmodel)
  bg_cell_map = get_cell_map(bg_trian)
  bg_inv_maps = lazy_map(inverse_map, bg_cell_map)
  cell_to_inv_map = lazy_map(Reindex(bg_inv_maps), tface_to_mface)
  cell_to_coords = lazy_map(Broadcasting(Reindex(d.point_to_coords)), d.cell_to_points)
  cell_to_rcoords = lazy_map(evaluate, cell_to_inv_map, cell_to_coords)

  ctype_to_reffe = get_reffes(t.grid)
  cell_to_ctype = get_cell_type(t.grid)
  if isempty(ctype_to_reffe)
    tface_to_mface_map = Fill(GenericField(identity), 0)
  else
    reffe = first(ctype_to_reffe)
    cell_to_shapefuns = expand_cell_data([get_shapefuns(reffe)], cell_to_ctype)
    tface_to_mface_map = lazy_map(Gridap.Fields.linear_combination, cell_to_rcoords, cell_to_shapefuns)
  end
  FaceToFaceGlue(tface_to_mface, tface_to_mface_map, nothing)
end

# --- Normal and tangent vectors ---

function get_tangent_vector(Σ::IntersectionTriangulation, Γ_id::Int)
  if Γ_id == 1
    cell_tangent = lazy_map(Gridap.Fields.constant_field, Σ.IntData.cell_to_tangent_1)
  elseif Γ_id == 2
    cell_tangent = lazy_map(Gridap.Fields.constant_field, Σ.IntData.cell_to_tangent_2)
  else
    error("Γ_id should be 1 or 2, got $Γ_id")
  end
  GenericCellField(cell_tangent, Σ, ReferenceDomain())
end

function Gridap.CellData.get_normal_vector(Σ::IntersectionTriangulation, Γ_id::Int)
  if Γ_id == 1
    cell_normal = lazy_map(Gridap.Fields.constant_field, Σ.IntData.cell_to_normal_1)
  elseif Γ_id == 2
    cell_normal = lazy_map(Gridap.Fields.constant_field, Σ.IntData.cell_to_normal_2)
  else
    error("Γ_id should be 1 or 2, got $Γ_id")
  end
  GenericCellField(cell_normal, Σ, ReferenceDomain())
end

# RestrictedIntersectionTriangulation

# RestrictedIntersectionTriangulation.jl
#
# Defines RestrictedIntersectionTriangulation: an IntersectionTriangulation filtered
# to retain only intersection cells (points in 2D, line segments in 3D) that lie
# inside a given geometry Dφ (or outside when using !Dφ).
#
# Filtering uses the precomputed ls_to_subfacet_to_inout data in cutgeo so that
# no level-set re-evaluation is needed — avoiding reference-frame mismatches that
# arise when an intersection point sits on a bgcell boundary.
#
# Methods:
#   Σ = RestrictedIntersectionTriangulation(cutgeo, "φi", "φj", Dφ)
#   Σ = RestrictedIntersectionTriangulation(cutgeo, "φi", "φj", !Dφ)  # outside Dφ
struct RestrictedIntersectionTriangulation{Dc,Dp,T,A} <: Triangulation{Dc,Dp}
  Γ1::GridapEmbedded.Interfaces.SubFacetTriangulation
  Γ2::GridapEmbedded.Interfaces.SubFacetTriangulation
  IntData::IntersectionData{Dp,T}
  bgmodel::A
  grid::UnstructuredGrid{Dc,Dp,T,NonOriented}
end

function RestrictedIntersectionTriangulation(
  cutgeo  :: EmbeddedDiscretization,
  φi_name :: String,
  φj_name :: String,
  Dφ      :: DiscreteGeometryFromFEFunction
)
  Γ1 = EmbeddedBoundary(cutgeo, φi_name)
  Γ2 = EmbeddedBoundary(cutgeo, φj_name)
  Σ  = IntersectionTriangulation(Γ1, Γ2)
  _restrict(Σ, Dφ, cutgeo)
end

function _restrict(
  Σ      :: IntersectionTriangulation{Dc,Dp,T,A},
  Dφ     :: DiscreteGeometryFromFEFunction,
  cutgeo :: EmbeddedDiscretization
) where {Dc,Dp,T,A}
  idata   = Σ.IntData
  n_cells = length(idata.cell_to_bgcell)

  # Empty triangulation — nothing to filter
  if n_cells == 0
    return RestrictedIntersectionTriangulation{Dc,Dp,T,A}(
      Σ.Γ1, Σ.Γ2, idata, Σ.bgmodel, Σ.grid
    )
  end

  tree       = get_tree(Dφ)
  leaf_cache = _build_leaf_cache(tree)

  # Inverse maps from physical → reference coords, one per bgcell
  bg_trian    = Triangulation(Σ.bgmodel)
  bg_cell_map = get_cell_map(bg_trian)
  bg_inv_maps = lazy_map(inverse_map, bg_cell_map)

  keep = Vector{Bool}(undef, n_cells)
  for i in 1:n_cells
    bgcell   = Int(idata.cell_to_bgcell[i])
    cell_pts = view(idata.cell_to_points, i)
    n_pts    = length(cell_pts)

    # Physical midpoint of the cell.
    midpt = sum(idata.point_to_coords[pid] for pid in cell_pts) / n_pts

    # Compute reference coordinates of the midpoint inside the assigned bgcell.
    rcoord = evaluate(bg_inv_maps[bgcell], midpt)

    keep[i] = _eval_tree(tree, bgcell, rcoord, leaf_cache)
  end

  idata_f = _filter_idata(idata, keep)
  grid_f  = _create_intersection_grid(idata_f.point_to_coords, idata_f.cell_to_points, Val(Dp))

  RestrictedIntersectionTriangulation{Dc,Dp,T,A}(
    Σ.Γ1, Σ.Γ2, idata_f, Σ.bgmodel, grid_f
  )
end

# Build a dict: objectid(φh) → get_data(φh) for every unique leaf in the tree.
function _build_leaf_cache(tree)
  cache = IdDict{UInt, Any}()
  for node in PreOrderDFS(tree)
    if isnothing(node.leftchild) && isnothing(node.rightchild)  # Leaf
      φh, _, _ = node.data
      id = objectid(φh)
      haskey(cache, id) || (cache[id] = get_data(φh))
    end
  end
  cache
end

# Leaf: evaluate φh at (bgcell, rcoord); inside ≡ value < 0.
function _eval_tree(node::Node{Td,Nothing,Nothing}, bgcell, rcoord, cache) where Td
  φh, _, _ = node.data
  val = evaluate(cache[objectid(φh)][bgcell], rcoord)
  return val < 0
end

# Internal node: recurse and apply the CSG operation.
function _eval_tree(node::Node, bgcell, rcoord, cache)
  op, _, _ = node.data
  if op === :!
    return !_eval_tree(node.leftchild, bgcell, rcoord, cache)
  elseif op === :∪
    return  _eval_tree(node.leftchild,  bgcell, rcoord, cache) ||
            _eval_tree(node.rightchild, bgcell, rcoord, cache)
  elseif op === :∩
    return  _eval_tree(node.leftchild,  bgcell, rcoord, cache) &&
            _eval_tree(node.rightchild, bgcell, rcoord, cache)
  elseif op === :-  # setdiff(A, B) = A ∩ ¬B
    return  _eval_tree(node.leftchild,  bgcell, rcoord, cache) &&
           !_eval_tree(node.rightchild, bgcell, rcoord, cache)
  else
    error("_eval_tree: unknown CSG operation $op")
  end
end

function _filter_idata(idata::IntersectionData{Dp,T}, keep::AbstractVector{Bool}) where {Dp,T}
  keep_ids = findall(keep)

  if isempty(keep_ids)
    empty_pts   = Point{Dp,T}[]
    empty_cells = Table(Int32[], Int32[1])
    return IntersectionData{Dp,T}(
      empty_pts, empty_pts, empty_cells, Int32[],
      empty_pts, empty_pts, empty_pts,  empty_pts
    )
  end

  # Collect all point ids referenced by the kept cells (in order, then unique)
  all_pids = Int32[]
  for i in keep_ids
    append!(all_pids, view(idata.cell_to_points, i))
  end
  used_ids  = unique(all_pids)
  old_to_new = Dict{Int32,Int32}(old => Int32(k) for (k, old) in enumerate(used_ids))

  # Compact point arrays
  new_coords  = idata.point_to_coords[used_ids]
  new_rcoords = idata.point_to_rcoords[used_ids]

  # Rebuild cell-to-points Table with renumbered point ids
  new_data = Int32[]
  new_ptrs = Int32[1]
  for i in keep_ids
    for p in view(idata.cell_to_points, i)
      push!(new_data, old_to_new[p])
    end
    push!(new_ptrs, Int32(length(new_data) + 1))
  end
  new_cell_to_points = Table(new_data, new_ptrs)

  # Filtered per-cell arrays
  new_bgcells = idata.cell_to_bgcell[keep_ids]
  new_n1      = idata.cell_to_normal_1[keep_ids]
  new_n2      = idata.cell_to_normal_2[keep_ids]
  new_t1      = idata.cell_to_tangent_1[keep_ids]
  new_t2      = idata.cell_to_tangent_2[keep_ids]

  IntersectionData{Dp,T}(
    new_coords, new_rcoords, new_cell_to_points,
    new_bgcells, new_n1, new_n2, new_t1, new_t2
  )
end

Gridap.Geometry.get_background_model(t::RestrictedIntersectionTriangulation) = t.bgmodel
Gridap.Geometry.get_grid(t::RestrictedIntersectionTriangulation)             = t.grid

function Gridap.Geometry.get_glue(
  t::RestrictedIntersectionTriangulation{Dc,Dp}, ::Val{D}
) where {Dc,Dp,D}
  D != Dp && return nothing
  d = t.IntData
  tface_to_mface = d.cell_to_bgcell

  bg_trian    = Triangulation(t.bgmodel)
  bg_cell_map = get_cell_map(bg_trian)
  bg_inv_maps = lazy_map(inverse_map, bg_cell_map)
  cell_to_inv_map  = lazy_map(Reindex(bg_inv_maps), tface_to_mface)
  cell_to_coords   = lazy_map(Broadcasting(Reindex(d.point_to_coords)), d.cell_to_points)
  cell_to_rcoords  = lazy_map(evaluate, cell_to_inv_map, cell_to_coords)

  ctype_to_reffe  = get_reffes(t.grid)
  cell_to_ctype   = get_cell_type(t.grid)
  if isempty(ctype_to_reffe)
    tface_to_mface_map = Fill(GenericField(identity), 0)
  else
    reffe = first(ctype_to_reffe)
    cell_to_shapefuns  = expand_cell_data([get_shapefuns(reffe)], cell_to_ctype)
    tface_to_mface_map = lazy_map(Gridap.Fields.linear_combination, cell_to_rcoords, cell_to_shapefuns)
  end
  FaceToFaceGlue(tface_to_mface, tface_to_mface_map, nothing)
end

function get_tangent_vector(Σ::RestrictedIntersectionTriangulation, Γ_id::Int)
  if Γ_id == 1
    cell_tangent = lazy_map(Gridap.Fields.constant_field, Σ.IntData.cell_to_tangent_1)
  elseif Γ_id == 2
    cell_tangent = lazy_map(Gridap.Fields.constant_field, Σ.IntData.cell_to_tangent_2)
  else
    error("Γ_id must be 1 or 2, got $Γ_id")
  end
  GenericCellField(cell_tangent, Σ, ReferenceDomain())
end

function Gridap.CellData.get_normal_vector(Σ::RestrictedIntersectionTriangulation, Γ_id::Int)
  if Γ_id == 1
    cell_normal = lazy_map(Gridap.Fields.constant_field, Σ.IntData.cell_to_normal_1)
  elseif Γ_id == 2
    cell_normal = lazy_map(Gridap.Fields.constant_field, Σ.IntData.cell_to_normal_2)
  else
    error("Γ_id must be 1 or 2, got $Γ_id")
  end
  GenericCellField(cell_normal, Σ, ReferenceDomain())
end