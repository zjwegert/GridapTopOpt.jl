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
  bgcell_on_bdy(c) = findall(f -> is_bgbdy(f), view(cell_to_face_bg, c))

  Df = num_cell_dims(face_model)        # 1 in 2D, 2 in 3D
  face_topo          = get_grid_topology(face_model)
  fm_facet_to_cells  = Gridap.Geometry.get_faces(face_topo, Df-1, Df)  # (Df-1)-facet → Df-cells
  is_bdy_fm_facet    = get_isboundary_face(face_topo, Df-1)
  facet_to_bgcell    = effective_trian.subfacets.facet_to_bgcell

  get_face_coordinates(topo,d) = lazy_map(Broadcasting(Reindex(get_vertex_coordinates(topo))), get_faces(topo, d, 0))
  face_coords_bg = get_face_coordinates(bg_topo, Dc-1)
  face_coords_fm = get_face_coordinates(face_topo, Df-1)
  function on_face(cf,bgf)
    cfx = face_coords_fm[cf]
    bgx = face_coords_bg[bgf]
    is_on_face(bgx, cfx)
  end

  function on_phys_body(f)
    c = first(fm_facet_to_cells[f])
    bgbndfacet_loc = bgcell_on_bdy(facet_to_bgcell[c])
    if isempty(bgbndfacet_loc) # My attached bgcell is not on the boundary cell layer
      is_on_phys_bdy[f] = false
    else # I need to check if I am actually on that face
      is_on_phys_bdy[f] = any(bgbndfacet_loc) do bgbndfacet_loc
        bgbndfacet = view(cell_to_face_bg, facet_to_bgcell[c])[bgbndfacet_loc]
        on_face(f,bgbndfacet)
      end
    end
  end

  n_fm_facets    = num_faces(face_model, Df-1)   # == num_facets(face_model)
  is_on_phys_bdy = fill(false, n_fm_facets)
  for f in 1:n_fm_facets
    is_bdy_fm_facet[f] || continue
    # Topological boundary facet → exactly one incident cell in face_model
    is_on_phys_bdy[f] = on_phys_body(f)
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

function is_on_face(bgx::Vector{VectorValue{3,T}}, cfx::Vector{VectorValue{3,T}}) where T<:Number
  n = cross(bgx[1] - bgx[2], bgx[1] - bgx[3])  # normal to bg-face
  d = maximum(norm.(dot.(n, cfx .- bgx[1])))  # max distance of face_model facet vertices to bg-face plane
  return d < 100eps()
end

function is_on_face(bgx::Vector{VectorValue{2,T}}, cfx::Vector{VectorValue{2,T}}) where T<:Number
  d = norm(cross(bgx[2] - bgx[1], cfx[1] - bgx[1]))
  return d < 100eps()
end