
using GridapEmbedded

using GridapEmbedded.Interfaces
using GridapEmbedded.Interfaces: SubFacetData, SubFacetTriangulation

using Gridap.Geometry
using Gridap.Geometry: CompositeTriangulation

Base.round(a::VectorValue{D,T};kwargs...) where {D,T} = VectorValue{D,T}(round.(a.data;kwargs...))

function consistent_facet_to_points(
  facet_to_points::Table, point_to_coords::Vector
)
  f(pt::Number) = round(pt;sigdigits=12)
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
  return facet_to_uids, unique_ids
end

function Geometry.get_active_model(
  trian::SubFacetTriangulation{Dc,Dp}
) where {Dc,Dp}
  subgrid = trian.subgrid
  subfacets = trian.subfacets
  facet_to_uids, uid_to_point = consistent_facet_to_points(
    subfacets.facet_to_points,subfacets.point_to_coords
  )
  topo = UnstructuredGridTopology(
    subgrid,facet_to_uids,uid_to_point
  )
  return UnstructuredDiscreteModel(subgrid,topo,FaceLabeling(topo))
end

function generate_ghost_trian(
  trian::CompositeTriangulation,
  bgmodel::UnstructuredDiscreteModel{Dc}
) where {Dc}
  cell_glue = get_glue(trian,Val(Dc))
  return generate_ghost_trian(trian,bgmodel,cell_glue)
end

function generate_ghost_trian(
  trian::CompositeTriangulation,
  bgmodel::UnstructuredDiscreteModel{Dc},
  cell_glue::SkeletonPair{<:FaceToFaceGlue}
) where {Dc}
  topo = get_grid_topology(bgmodel)
  face_to_cell = get_faces(topo,Dc-1,Dc)
  cell_to_face = get_faces(topo,Dc,Dc-1)

  n_bgfaces = num_faces(bgmodel,Dc-1)
  n_faces = num_cells(trian)
  ghost_faces = zeros(Int32,n_faces)
  p_lcell = ones(Int8,n_bgfaces)
  m_lcell = ones(Int8,n_bgfaces)
  for (i,(p_cell, m_cell)) in enumerate(zip(cell_glue.plus.tface_to_mface,cell_glue.minus.tface_to_mface))
    inter = intersect(cell_to_face[p_cell],cell_to_face[m_cell])
    @assert length(inter) == 1
    face = first(inter)
    ghost_faces[i] = face

    nbors = face_to_cell[ghost_faces[i]]
    p_lcell[face] = findfirst(==(p_cell),nbors)
    m_lcell[face] = findfirst(==(m_cell),nbors)
  end

  plus = BoundaryTriangulation(bgmodel,ghost_faces,p_lcell)
  minus = BoundaryTriangulation(bgmodel,ghost_faces,m_lcell)
  return SkeletonTriangulation(plus,minus)
end

function generate_ghost_trian(
  trian::CompositeTriangulation,
  bgmodel::UnstructuredDiscreteModel{Dc},
  cell_glue::FaceToFaceGlue
) where {Dc}
  topo = get_grid_topology(bgmodel)
  face_to_cell = get_faces(topo,Dc-1,Dc)
  cell_to_face = get_faces(topo,Dc,Dc-1)
  is_boundary(f) = isone(length(view(face_to_cell,f)))

  n_faces = num_cells(trian)
  ghost_faces = zeros(Int32,n_faces)
  for (i,cell) in enumerate(cell_glue.tface_to_mface)
    faces = filter(is_boundary,view(cell_to_face,cell))
    @assert length(faces) == 1 # TODO: This will break if we are in a corner
    face = first(faces)
    ghost_faces[i] = face
  end

  # NOTE: lcell is always 1 for boundary facets
  return BoundaryTriangulation(bgmodel,ghost_faces)
end

"""
    get_ghost_mask(
      face_trian::SubFacetTriangulation{Df,Dc},
      face_model::UnstructuredDiscreteModel{Df,Dc}
    ) where {Df,Dc}

    get_ghost_mask(face_trian::SubFacetTriangulation) = get_ghost_mask(face_trian,get_active_model(face_trian))

  Returns a mask for ghost faces. We define ghost faces as the interfaces between two 
  different cut facets that are located in different background cells.

  The second condition is important: In 3D, some cuts subcells may not be simplices. 
  In this case, we simplexify the subcell. This creates extra cut interfaces that are 
  interior to a background cell. These are not considered ghost faces.

  - In 2D: Dc = 2, Df = 1 -> Ghost faces have dimension 0 (i.e interface points)
  - In 3D: Dc = 3, Df = 2 -> Ghost faces have dimension 1 (i.e interface edges)
"""
function get_ghost_mask(
  face_trian::SubFacetTriangulation{Df,Dc},
  face_model::UnstructuredDiscreteModel{Df,Dc}
) where {Df,Dc}
  topo = get_grid_topology(face_model)
  face_to_facets = get_faces(topo,Df-1,Df)

  subfacets = face_trian.subfacets
  facet_to_bgcell = subfacets.facet_to_bgcell
  
  n_faces = num_faces(topo,Df-1)
  face_is_ghost = zeros(Bool,n_faces)
  for face in 1:n_faces
    facets = view(face_to_facets,face)
    is_boundary = isone(length(facets))
    if !is_boundary
      @assert length(facets) == 2
      bgcells = view(facet_to_bgcell,facets)
      is_ghost = (bgcells[1] != bgcells[2])
      face_is_ghost[face] = is_ghost
    end
  end

  return face_is_ghost
end

function get_ghost_mask(
  face_trian::SubFacetTriangulation
)
  face_model = get_active_model(face_trian)
  return get_ghost_mask(face_trian,face_model)
end

struct SubFacetSkeletonTriangulation{Di,Df,Dp} <: Triangulation{Di,Dp}
  cell_skeleton::CompositeTriangulation{Di,Dp} # Interface -> BG Cell pair
  face_skeleton::SkeletonTriangulation{Di,Dp}  # Interface -> Cut Facet pair
  ghost_skeleton::SkeletonTriangulation{Df,Dp} # Ghost Facet -> BG Cell pair
  face_trian::SubFacetTriangulation{Df,Dp}     # Cut Facet -> BG Cell
  face_model::UnstructuredDiscreteModel{Df,Dp}
end

function Geometry.SkeletonTriangulation(face_trian::SubFacetTriangulation)
  bgmodel = get_background_model(face_trian)
  face_model = get_active_model(face_trian)

  ghost_mask = get_ghost_mask(face_trian,face_model)
  face_skeleton = SkeletonTriangulation(face_model,ghost_mask)
  cell_skeleton = CompositeTriangulation(face_trian,face_skeleton)
  ghost_skeleton = generate_ghost_trian(cell_skeleton,bgmodel)
  return SubFacetSkeletonTriangulation(cell_skeleton,face_skeleton,ghost_skeleton,face_trian,face_model)
end

function Geometry.get_background_model(t::SubFacetSkeletonTriangulation)
  get_background_model(t.cell_skeleton)
end

function Geometry.get_active_model(t::SubFacetSkeletonTriangulation)
  get_active_model(t.cell_skeleton)
end

function Geometry.get_grid(t::SubFacetSkeletonTriangulation)
  get_grid(t.cell_skeleton)
end

# Domain changes

function Geometry.get_glue(ttrian::SubFacetSkeletonTriangulation{Di,Df,Dp},::Val{D}) where {D,Di,Df,Dp}
  get_glue(ttrian.cell_skeleton,Val(D))
end

function Geometry.is_change_possible(
  strian::SubFacetTriangulation,ttrian::SubFacetSkeletonTriangulation
)
  return strian === ttrian.face_trian
end

function CellData.change_domain(
  a::CellField,ttrian::SubFacetSkeletonTriangulation,tdomain::DomainStyle
)
  strian = get_triangulation(a)
  if strian === ttrian
    # 1) CellField defined on the skeleton
    return change_domain(a,DomainStyle(a),tdomain)
  end

  if is_change_possible(strian,ttrian.cell_skeleton)
    # 2) CellField defined on the bgmodel
    b = change_domain(a,ttrian.cell_skeleton,tdomain)
  elseif strian === ttrian.face_trian
    # 3) CellField defined on the cut facets
    itrian = Triangulation(ttrian.face_model)
    _a = CellData.similar_cell_field(a,CellData.get_data(a),itrian,DomainStyle(a))
    b = change_domain(_a,ttrian.face_skeleton,tdomain)
  else
    @notimplemented
  end
  return CellData.similar_cell_field(b,CellData.get_data(b),ttrian,DomainStyle(b))
end

function CellData.change_domain(
  f::CellData.OperationCellField,ttrian::SubFacetSkeletonTriangulation,tdomain::DomainStyle
)
  args = map(i->change_domain(i,ttrian,tdomain),f.args)
  CellData.OperationCellField(f.op,args...)
end

# Normal vector to the cut facets , n_∂Ω
function get_subfacet_normal_vector(trian::SubFacetSkeletonTriangulation{0})
  n_∂Ω = get_normal_vector(trian.face_trian)
  plus = change_domain(n_∂Ω.plus,trian,ReferenceDomain())
  minus = change_domain(-n_∂Ω.minus,trian,ReferenceDomain())
  return SkeletonPair(plus,minus)
end
function get_subfacet_normal_vector(trian::SubFacetSkeletonTriangulation{1})
  n_∂Ω = get_normal_vector(trian.face_trian)
  plus = change_domain(n_∂Ω.plus,trian,ReferenceDomain())
  minus = change_domain(n_∂Ω.minus,trian,ReferenceDomain()) # This needs to be postive for mk to be correctly oriented in 3d
  return SkeletonPair(plus,minus)
end

# Normal vector to the ghost facets, n_k
# function get_ghost_normal_vector(trian::SubFacetSkeletonTriangulation)
#   n = get_normal_vector(trian.ghost_skeleton)
#   plus  = GenericCellField(CellData.get_data(n.plus),trian,ReferenceDomain())
#   minus = GenericCellField(CellData.get_data(n.minus),trian,ReferenceDomain())
#   return SkeletonPair(plus,minus)
# end
# The above fails as fields defined on SubFacetSkeletonTriangulation have 1 input in 3d points
#   but fields defined on ghost skeleton have 2 inputs. Should change_domain fix this?

function get_ghost_normal_vector(trian::SubFacetSkeletonTriangulation{0})
  n = get_normal_vector(trian.ghost_skeleton)
  plus  = CellField(evaluate(get_data(n.plus),Point(0,)),trian,ReferenceDomain())
  minus = CellField(evaluate(get_data(n.minus),Point(0,)),trian,ReferenceDomain())
  return SkeletonPair(plus,minus)
end
function get_ghost_normal_vector(trian::SubFacetSkeletonTriangulation{1})
  n = get_normal_vector(trian.ghost_skeleton)
  plus  = CellField(evaluate(get_data(n.plus),Point(0,0)),trian,ReferenceDomain())
  minus = CellField(evaluate(get_data(n.minus),Point(0,0)),trian,ReferenceDomain())
  return SkeletonPair(plus,minus)
end

"""
# Returns a consistent tangent vector in the ReferenceDomain. Consistency
# is achieved by choosing it's direction as going from the node with the
# smallest id to the one with the largest id.
function get_tangent_vector(
  trian::BoundaryTriangulation{1};
  ttrian = trian
)
  flip(e,t::T) where T = (e[1] < e[2]) ? t : -t :: T
  bgmodel = get_background_model(trian)
  topo = get_grid_topology(bgmodel)
  glue = trian.glue
  cell_to_poly = lazy_map(get_polytope,get_cell_reffe(bgmodel))
  cell_to_ltangents = lazy_map(get_edge_tangent,cell_to_poly)
  edge_to_ltangents = lazy_map(Reindex(cell_to_ltangents),glue.face_to_cell)
  edge_to_tangent = lazy_map(getindex,edge_to_ltangents,glue.face_to_lface)
  edge_to_nodes = lazy_map(Reindex(get_faces(topo,1,0)),glue.face_to_bgface)
  data = lazy_map(ConstantField,lazy_map(flip,edge_to_nodes,edge_to_tangent))
  return GenericCellField(data,ttrian,ReferenceDomain())
end
"""

# Returns the tangent vector in the PhysicalDomain
function get_tangent_vector(
  trian::BoundaryTriangulation{1};
  ttrian = trian
)
  function t(c)
    @assert length(c) == 2
    t = c[2] - c[1]
    return t/norm(t)
  end
  data = lazy_map(constant_field,lazy_map(t,get_cell_coordinates(trian)))
  return GenericCellField(data,ttrian,ReferenceDomain())
end

function get_tangent_vector(
  trian::SkeletonTriangulation{1};
  ttrian = trian
)
  plus = get_tangent_vector(trian.plus;ttrian=ttrian)
  minus = get_tangent_vector(trian.minus;ttrian=ttrian)
  return SkeletonPair(plus,minus)
end

_2d_cross(n) = VectorValue(-n[2],n[1]);
function normalise(v) # <- Double check RE if this neccessary?
  m = sqrt(inner(v,v))
  if m < eps()
    return zero(v)
  else
    return v/m
  end
end

# Normal vector to the cut interface, n_S
function CellData.get_normal_vector(trian::SubFacetSkeletonTriangulation{Di}) where {Di}
  if Di == 0
    n_k = get_ghost_normal_vector(trian)
    n_S = Operation(_2d_cross)(n_k) # nS = nk x tS and tS = ±e₃ in 2D
    # n_S = get_tangent_vector(trian.ghost_skeleton;ttrian=trian) # Not oriented correctly in 2d. # TODO: understand why!?
  elseif Di == 1
    n_k = get_ghost_normal_vector(trian)
    t_S = get_tangent_vector(trian.face_skeleton;ttrian=trian)
    n_S = Operation(normalise ∘ cross)(n_k,t_S) # nk = tS x nS -> nS = nk x tS (eq 6.25)
  else
    @notimplemented
  end
  # We pick the plus side. Consistency is given by later
  # computing t_S as t_S = n_S x n_k
  return n_S.plus
end
# function alt_get_normal_vector(trian::SubFacetSkeletonTriangulation{Di}) where {Di}
#   if Di == 0
#     n_k = alt_get_ghost_normal_vector(trian)
#     n_S = Operation(_2d_cross)(n_k) # nS = nk x tS and tS = ±e₃ in 2D
#   elseif Di == 1
#     n_k = alt_get_ghost_normal_vector(trian)
#     t_S = alt_get_tangent_vector(trian)
#     n_S = Operation(cross)(n_k,t_S) # nk = tS x nS -> nS = nk x tS (eq 6.25)
#   else
#     @notimplemented
#   end
#   # We pick the plus side. Consistency is given by later
#   # computing t_S as t_S = n_S x n_k
#   return n_S.plus
# end

# Tangent vector to the cut interface, t_S = n_S x n_k
function get_tangent_vector(trian::SubFacetSkeletonTriangulation{Di}) where {Di}
  @notimplementedif Di != 1
  n_S = get_normal_vector(trian)
  n_k = get_ghost_normal_vector(trian)
  return Operation(normalise ∘ cross)(n_S,n_k)
end
# function alt_get_tangent_vector(trian::SubFacetSkeletonTriangulation{1})
#   n_∂Ω = get_subfacet_normal_vector(trian)
#   n_k = get_ghost_normal_vector(trian)
#   return Operation(normalise ∘ cross)(n_k,n_∂Ω) # t_S = n_k × n_∂Ω # <- need to show this if we wanted to actually use it!
# end

# Conormal vectors, m_k = t_S x n_∂Ω
function get_conormal_vector(trian::SubFacetSkeletonTriangulation{Di}) where {Di}
  op = Operation(GridapEmbedded.LevelSetCutters._normal_vector)
  n_∂Ω = get_subfacet_normal_vector(trian)
  if Di == 0 # 2D
    m_k = op(n_∂Ω)
  elseif Di == 1 # 3D
    t_S = get_tangent_vector(trian)
    m_k = op(t_S,n_∂Ω) # m_k = t_S x n_∂Ω (eq 6.26)
  else
    @notimplemented
  end
  return m_k
end
# function alt_get_conormal_vector(trian::SubFacetSkeletonTriangulation{Di}) where {Di}
#   op = Operation(GridapEmbedded.LevelSetCutters._normal_vector)
#   n_∂Ω = get_subfacet_normal_vector(trian)
#   if Di == 0 # 2D
#     m_k = op(n_∂Ω)
#   elseif Di == 1 # 3D
#     t_S = alt_get_tangent_vector(trian)
#     m_k = op(t_S,n_∂Ω) # m_k = t_S x n_∂Ω (eq 6.26)
#   else
#     @notimplemented
#   end
#   return m_k
# end

# This will go to Gridap

function Arrays.evaluate!(cache,k::Operation,a::SkeletonPair{<:CellField})
  plus = k(a.plus)
  minus = k(a.minus)
  SkeletonPair(plus,minus)
end

function Arrays.evaluate!(cache,k::Operation,a::SkeletonPair{<:CellField},b::SkeletonPair{<:CellField})
  plus = k(a.plus,b.plus)
  minus = k(a.minus,b.minus)
  SkeletonPair(plus,minus)
end

import Gridap.TensorValues: inner, outer
import LinearAlgebra: dot
import Base: abs, *, +, -, /

for op in (:/,)
  @eval begin
    ($op)(a::CellField,b::SkeletonPair{<:CellField}) = Operation($op)(a,b)
    ($op)(a::SkeletonPair{<:CellField},b::CellField) = Operation($op)(a,b)
  end
end

for op in (:outer,:*,:dot,:/)
  @eval begin
    ($op)(a::SkeletonPair{<:CellField},b::SkeletonPair{<:CellField}) = Operation($op)(a,b)
  end
end

function CellData.change_domain(a::SkeletonPair, ::ReferenceDomain, ::PhysicalDomain)
  plus = change_domain(a.plus,ReferenceDomain(),PhysicalDomain())
  minus = change_domain(a.minus,ReferenceDomain(),PhysicalDomain())
  return SkeletonPair(plus,minus)
end
