## Triangulations of the intersections Σᵢⱼ = Γᵢ ∩ Γⱼ computed by `cut_intersections`

using GridapEmbedded.CSG: get_geometry_node

"""
    struct IntersectionData{Dp,T,Tn}

Pieces of the pairwise level-set intersections Σᵢⱼ = Γᵢ ∩ Γⱼ, i < j, attached to the
background cells. Each piece is a point (2D) or a segment (3D) with its bgcell, the unit
normals (nᵢ,nⱼ) of Γᵢ and Γⱼ in the bgcell (oriented towards φ > 0), and its CUT/IN/OUT
state with respect to every level set (CUT for i and j).
"""
struct IntersectionData{Dp,T,Tn}
  cell_to_points::Table{Tn,Vector{Tn},Vector{Tn}}
  cell_to_bgcell::Vector{Tn}
  cell_to_normals::NTuple{2,Vector{VectorValue{Dp,T}}}
  ls_to_cell_to_inoutcut::Vector{Vector{Int8}}
  point_to_coords::Vector{Point{Dp,T}}
  point_to_rcoords::Vector{Point{Dp,T}}
end

function IntersectionData(st::IntersectionData,newcells::AbstractVector{<:Integer})
  cell_to_points = Table(lazy_map(Reindex(st.cell_to_points),newcells))
  cell_to_bgcell = st.cell_to_bgcell[newcells]
  cell_to_normals = map(n -> n[newcells],st.cell_to_normals)
  ls_to_cell_to_inoutcut = map(io -> io[newcells],st.ls_to_cell_to_inoutcut)
  IntersectionData(
    cell_to_points,
    cell_to_bgcell,
    cell_to_normals,
    ls_to_cell_to_inoutcut,
    st.point_to_coords,
    st.point_to_rcoords)
end

function IntersectionData(st::IntersectionData,cell_to_mask::AbstractVector{Bool})
  IntersectionData(st,findall(cell_to_mask))
end

function Geometry.UnstructuredGrid(st::IntersectionData{Dp}) where Dp
  Dc = Dp - 2
  reffe = LagrangianRefFE(Float64,ReferenceFEs.simplex_polytope(Val(Dc)),1)
  cell_types = fill(Int8(1),length(st.cell_to_points))
  UnstructuredGrid(
    st.point_to_coords,
    st.cell_to_points,
    [reffe,],
    cell_types)
end

"""
    struct IntersectionTriangulation{Dc,Dp} <: Triangulation{Dc,Dp}

Triangulation of the level-set intersections (vertices in 2D, segments in 3D), glued to the
background cells. Built with [`cut_intersections`](@ref), it contains the intersections of
all pairs of level sets; use [`restrict_intersections`](@ref) to select a pair.

Once restricted to the level sets `name1` and `name2`, `get_normal_vector(Σ,k)` and
`get_tangent_vector(Σ,k)` refer to `name1` for k = 1 and to `name2` for k = 2.
"""
struct IntersectionTriangulation{Dc,Dp,T,A,B} <: Triangulation{Dc,Dp}
  intersections::IntersectionData{Dp,T}
  bgmodel::A
  subgrid::B
  geo::DiscreteGeometryFromFEFunction
  oid_to_ls::Dict{UInt,Int}
  ls_pair::NTuple{2,Int} # (0,0) when not restricted
  function IntersectionTriangulation(
    intersections::IntersectionData{Dp,T},
    bgmodel::DiscreteModel,
    geo::DiscreteGeometryFromFEFunction,
    oid_to_ls::Dict{UInt,Int},
    ls_pair::NTuple{2,Int}=(0,0)
  ) where {Dp,T}
    Dc = Dp - 2
    subgrid = UnstructuredGrid(intersections)
    A = typeof(bgmodel)
    B = typeof(subgrid)
    new{Dc,Dp,T,A,B}(intersections,bgmodel,subgrid,geo,oid_to_ls,ls_pair)
  end
end

Geometry.get_background_model(t::IntersectionTriangulation) = t.bgmodel
Geometry.get_grid(t::IntersectionTriangulation) = t.subgrid

function Geometry.get_glue(t::IntersectionTriangulation{Dc,Dp},::Val{D}) where {Dc,Dp,D}
  D != Dp && return nothing
  st = t.intersections
  tface_to_mface = st.cell_to_bgcell
  reffe = first(get_reffes(t.subgrid))
  cell_to_shapefuns = Fill(get_shapefuns(reffe),length(tface_to_mface))
  cell_to_rcoords = lazy_map(Broadcasting(Reindex(st.point_to_rcoords)),st.cell_to_points)
  tface_to_mface_map = lazy_map(Fields.linear_combination,cell_to_rcoords,cell_to_shapefuns)
  FaceToFaceGlue(tface_to_mface,tface_to_mface_map,nothing)
end

"""
    restrict_intersections(trian::IntersectionTriangulation,name1::String,name2::String)
    restrict_intersections(trian::IntersectionTriangulation,name1::String,name2::String,geo)

Keep the intersections Γ₁ ∩ Γ₂ of the level sets named `name1` and `name2`. When `geo` is
given, only the intersections inside `geo` are kept.
"""
function restrict_intersections(trian::IntersectionTriangulation,name1::String,name2::String)
  _restrict_intersections(trian,name1,name2,c -> true)
end

function restrict_intersections(
  trian::IntersectionTriangulation,name1::String,name2::String,geo::DiscreteGeometryFromFEFunction
)
  tree = get_tree(geo)
  _restrict_intersections(trian,name1,name2,c -> _is_inside(tree,trian,c))
end

function _restrict_intersections(trian::IntersectionTriangulation,name1,name2,keep)
  a, b = _ls_index(trian,name1), _ls_index(trian,name2)
  @check a != b "The two level sets must be different"
  st = trian.intersections
  a_to_inoutcut, b_to_inoutcut = st.ls_to_cell_to_inoutcut[a], st.ls_to_cell_to_inoutcut[b]
  cell_to_mask = map(eachindex(st.cell_to_bgcell)) do c
    a_to_inoutcut[c] == CUT && b_to_inoutcut[c] == CUT && keep(c)
  end
  IntersectionTriangulation(IntersectionData(st,cell_to_mask),trian.bgmodel,trian.geo,trian.oid_to_ls,(a,b))
end

function _ls_index(trian::IntersectionTriangulation,name::String)
  node = get_geometry_node(get_tree(trian.geo),name)
  @check isnothing(node.leftchild) && isnothing(node.rightchild) "$name is not a level set"
  φh, = node.data
  trian.oid_to_ls[objectid(φh)]
end

# Evaluate the CSG tree on the CUT/IN/OUT state of an intersection
function _is_inside(node,trian::IntersectionTriangulation,cell)
  op, = node.data
  if isnothing(node.leftchild) && isnothing(node.rightchild)
    ls = trian.oid_to_ls[objectid(op)]
    inout = trian.intersections.ls_to_cell_to_inoutcut[ls][cell]
    @check inout != CUT "The restriction geometry cannot depend on the intersecting level sets"
    return inout == IN
  elseif op === :!
    return !_is_inside(node.leftchild,trian,cell)
  elseif op === :∪
    return _is_inside(node.leftchild,trian,cell) || _is_inside(node.rightchild,trian,cell)
  elseif op === :∩
    return _is_inside(node.leftchild,trian,cell) && _is_inside(node.rightchild,trian,cell)
  elseif op === :-
    return _is_inside(node.leftchild,trian,cell) && !_is_inside(node.rightchild,trian,cell)
  end
  @unreachable "Unknown CSG operation $op"
end

# Normals of the (name1,name2) level sets, the data stores them ordered as (i,j) with i < j
function _get_normals(Σ::IntersectionTriangulation)
  a, b = Σ.ls_pair
  @check a != b "Restrict the intersections to a pair of level sets first, see `restrict_intersections`"
  n_i, n_j = Σ.intersections.cell_to_normals
  a < b ? (n_i,n_j) : (n_j,n_i)
end

"""
    get_normal_vector(Σ::IntersectionTriangulation,k::Integer)

Unit normal of Γₖ (k = 1,2) on Σ, oriented towards φₖ > 0.
"""
function CellData.get_normal_vector(Σ::IntersectionTriangulation,k::Integer)
  @check k ∈ (1,2) "k should be 1 or 2, got $k"
  cell_normal = lazy_map(Fields.constant_field,_get_normals(Σ)[k])
  CellData.GenericCellField(cell_normal,Σ,ReferenceDomain())
end

"""
    get_tangent_vector(Σ::IntersectionTriangulation,k::Integer)

Unit conormal of Σ in Γₖ (k = 1,2): the projection of the other normal onto the tangent
space of Γₖ, i.e. pointing towards the positive side of the other level set.
"""
function CellData.get_tangent_vector(Σ::IntersectionTriangulation,k::Integer)
  @check k ∈ (1,2) "k should be 1 or 2, got $k"
  normals = _get_normals(Σ)
  tangents = map(normals[k],normals[3-k]) do n, m
    t = m - (n⋅m)*n
    t/norm(t)
  end
  cell_tangent = lazy_map(Fields.constant_field,tangents)
  CellData.GenericCellField(cell_tangent,Σ,ReferenceDomain())
end
