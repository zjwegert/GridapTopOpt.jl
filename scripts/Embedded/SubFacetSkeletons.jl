
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

  topo = get_grid_topology(bgmodel)
  face_to_cell = get_faces(topo,Dc-1,Dc)
  cell_to_face = get_faces(topo,Dc,Dc-1)

  n_bgfaces = num_faces(bgmodel,1)
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

struct SubFacetSkeletonTriangulation{Di,Df,Dp} <: Triangulation{Di,Dp}
  cell_skeleton::CompositeTriangulation{Di,Dp} # Interface -> BG Cell pair
  face_skeleton::SkeletonTriangulation{Di,Dp}  # Interface -> Cut Facet pair
  ghost_skeleton::SkeletonTriangulation{Df,Dp} # Ghost Facet -> BG Cell pair
  face_trian::SubFacetTriangulation{Df,Dp}     # Cut Facet -> BG Cell
  face_model::UnstructuredDiscreteModel{Df,Dp}
end

function SkeletonTriangulation(face_trian::SubFacetTriangulation)
  bgmodel = get_background_model(face_trian)
  face_model = get_active_model(face_trian)
  face_skeleton = SkeletonTriangulation(face_model)
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
function get_subfacet_normal_vector(trian::SubFacetSkeletonTriangulation) 
  return get_normal_vector(trian.face_trian)
end

# Normal vector to the ghost facets, n_k
function get_ghost_normal_vector(trian::SubFacetSkeletonTriangulation)
  n_out = get_normal_vector(trian.ghost_skeleton)
  n_in  = SkeletonPair((-1) * n_out.plus, (-1) * n_out.minus)
  plus  = GenericCellField(CellData.get_data(n_in.plus),trian,ReferenceDomain())
  minus = GenericCellField(CellData.get_data(n_in.minus),trian,ReferenceDomain())
  return SkeletonPair(plus,minus)
end

# Tangent vector to a skeleton triangulation
function get_tangent_vector(
  trian::SkeletonTriangulation{1};
  ttrian = trian
)
  # TODO: Can we create this in the ReferenceDomain? 
  # If so, we need to be careful to orient both sides so that their physical 
  # representations are consistent.
  function t(c) 
    @assert length(c) == 2
    t = c[2] - c[1]
    return t/norm(t)
  end
  data = lazy_map(ConstantField,lazy_map(t,get_cell_coordinates(trian)))
  plus = GenericCellField(data,ttrian,PhysicalDomain())
  minus = GenericCellField(data,ttrian,PhysicalDomain())
  return SkeletonPair(plus,minus)
end

# Normal vector to the cut interface, n_S
function CellData.get_normal_vector(trian::SubFacetSkeletonTriangulation{Di}) where {Di}
  if Di == 0
    n_S = get_tangent_vector(trian.ghost_skeleton;ttrian=trian)
  elseif Di == 1
    n_k = get_ghost_normal_vector(trian)
    t_S = get_tangent_vector(trian.cell_skeleton;ttrian=trian)
    n_S = Operation(cross)(n_k,t_S) # nk = tS x nS -> nS = nk x tS (eq 6.25)
  else
    @notimplemented
  end
  return n_S
end

# Tangent vector to the cut interface, t_S
function get_tangent_vector(trian::SubFacetSkeletonTriangulation{Di}) where {Di}
  @notimplementedif Di != 1
  n_S = get_normal_vector(trian)
  n_k = get_ghost_normal_vector(trian)
  return Operation(cross)(n_S,n_k)
end

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
