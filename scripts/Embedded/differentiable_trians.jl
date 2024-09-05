
using Gridap
using Gridap.CellData, Gridap.Geometry, Gridap.Helpers, Gridap.Arrays
using Gridap.Fields, Gridap.ReferenceFEs

using Gridap.Geometry: num_nodes

using FillArrays

using GridapEmbedded
using GridapEmbedded.LevelSetCutters, GridapEmbedded.Interfaces

# GridapEmbedded

function compute_cell_maps(cell_coords,cell_reffes)
  cell_shapefuns = lazy_map(get_shapefuns,cell_reffes)
  default_cell_map = lazy_map(linear_combination,cell_coords,cell_shapefuns)
  default_cell_grad = lazy_map(∇,default_cell_map)
  cell_poly = lazy_map(get_polytope,cell_reffes)
  cell_q0 = lazy_map(p->zero(first(get_vertex_coordinates(p))),cell_poly)
  origins = lazy_map(evaluate,default_cell_map,cell_q0)
  gradients = lazy_map(evaluate,default_cell_grad,cell_q0)
  cell_map = lazy_map(Gridap.Fields.affine_map,gradients,origins)
  return cell_map
end

function get_edge_list(poly::Polytope)
  ltcell_to_lpoints, simplex = simplexify(poly)
  simplex_edges = get_faces(simplex,1,0)
  ltcell_to_edges = map(pts -> map(e -> pts[e], simplex_edges), ltcell_to_lpoints)
  return unique(sort,vcat(ltcell_to_edges...))
end

function belongs_to_edge(p,edge,bgpts)
  p1,p2 = bgpts[edge]
  return iszero(cross(p-p1,p2-p1))
end

function compute_coords(
  coords ::Vector{<:Point{Dp,Tp}},
  rcoords::Vector{<:Point{Dp,Tp}},
  bg_coords::Vector{<:Point{Dp,Tp}},
  bg_rcoords::Vector{<:Point{Dp,Tp}},
  values::Vector{Tv},
  edges
) where {Dp,Tp,Tv}
  T = Point{Dp,Tv}
  new_coords = Vector{T}(undef,length(coords))
  new_rcoords = Vector{T}(undef,length(rcoords))
  for (i,(q,p)) in enumerate(zip(coords,rcoords))
    if p ∈ bg_rcoords
      new_coords[i] = q
      new_rcoords[i] = p
      continue
    end
    e = findfirst(edge -> belongs_to_edge(p,edge,bg_rcoords), edges)
    n1, n2 = edges[e]
    q1, q2 = bg_coords[n1], bg_coords[n2]
    p1, p2 = bg_rcoords[n1], bg_rcoords[n2]
    v1, v2 = values[n1], values[n2]
    w1, w2 = abs(v1), abs(v2)
    λ = w1/(w1+w2)
    new_coords[i] = q1 + λ*(q2-q1)
    new_rcoords[i] = p1 + λ*(p2-p1)
  end
  return new_coords, new_rcoords
end

function compute_coords(
  cell_to_coords::AbstractVector{<:AbstractVector{<:Point{Dp,Tp}}},
  cell_to_rcoords::AbstractVector{<:AbstractVector{<:Point{Dp,Tp}}},
  cell_to_bgcoords::AbstractVector{<:AbstractVector{<:Point{Dp,Tp}}},
  cell_to_bg_rcoords::AbstractVector{<:AbstractVector{<:Point{Dp,Tp}}},
  cell_to_values::AbstractVector{<:AbstractVector{Tv}},
  cell_to_edges
) where {Dp,Tp,Tv}
  T = Point{Dp,Tv}
  results = lazy_map(compute_coords,cell_to_coords,cell_to_rcoords,cell_to_bgcoords,cell_to_bg_rcoords,cell_to_values,cell_to_edges)
  cache = array_cache(results)
  ncells = length(cell_to_coords)
  new_cell_to_coords = Vector{Vector{T}}(undef,ncells)
  new_cell_to_rcoords = Vector{Vector{T}}(undef,ncells)
  for cell in 1:ncells
    new_coords, new_rcoords = getindex!(cache,results,cell)
    new_cell_to_coords[cell] = new_coords
    new_cell_to_rcoords[cell] = new_rcoords
  end
  return new_cell_to_coords, new_cell_to_rcoords
end

function compute_dual_values(
  trian::GridapEmbedded.Interfaces.SubCellTriangulation,
  φh::CellField
)
  bgmodel = get_background_model(trian)
  subcells = trian.subcells

  cell_to_bgcell   = subcells.cell_to_bgcell
  cell_to_points   = subcells.cell_to_points
  point_to_rcoords = subcells.point_to_rcoords
  point_to_coords  = subcells.point_to_coords

  ctypes = get_cell_type(bgmodel)
  bgcell_to_polys = expand_cell_data(get_polytopes(bgmodel),ctypes)
  bgcell_to_coords = get_cell_coordinates(bgmodel)
  bgcell_to_rcoords = lazy_map(get_vertex_coordinates,bgcell_to_polys)
  bgcell_to_edges = lazy_map(get_edge_list,bgcell_to_polys)
  bgcell_to_values = lazy_map(evaluate,CellData.get_data(φh),bgcell_to_rcoords)
  
  cell_to_reffes = Fill(LagrangianRefFE(Float64,TRI,1),length(cell_to_points))
  cell_to_bgcoords = lazy_map(Reindex(bgcell_to_coords),cell_to_bgcell)
  cell_to_bgrcoords = lazy_map(Reindex(bgcell_to_rcoords),cell_to_bgcell)
  cell_to_values = lazy_map(Reindex(bgcell_to_values),cell_to_bgcell)
  cell_to_edges = lazy_map(Reindex(bgcell_to_edges),cell_to_bgcell)
  
  old_cell_to_rcoords = lazy_map(Broadcasting(Reindex(point_to_rcoords)),cell_to_points)
  old_cell_to_coords = lazy_map(Broadcasting(Reindex(point_to_coords)),cell_to_points)
  
  new_cell_to_coords, new_cell_to_rcoords = compute_coords(
    old_cell_to_coords,old_cell_to_rcoords,cell_to_bgcoords,cell_to_bgrcoords,cell_to_values,cell_to_edges
  )
  #@assert all(new_cell_to_rcoords .== old_cell_to_rcoords)
  #@assert all(new_cell_to_coords .== old_cell_to_coords)
  
  new_cmaps = compute_cell_maps(new_cell_to_coords,cell_to_reffes)
  new_ref_cmaps = compute_cell_maps(new_cell_to_rcoords,cell_to_reffes)
  return new_cell_to_coords, new_cell_to_rcoords, new_cmaps, new_ref_cmaps
end

function GridapEmbedded.LevelSetCutters._get_value_at_coords(
  φh::FEFunction,model::DiscreteModel{Dc,Dp}
) where {Dc,Dp}
  values = get_free_dof_values(φh)
  return values
end

# Autodiff

function FESpaces._compute_cell_ids(uh,ttrian::AppendedTriangulation)
  ids_a = FESpaces._compute_cell_ids(uh,ttrian.a)
  ids_b = FESpaces._compute_cell_ids(uh,ttrian.b)
  lazy_append(ids_a,ids_b)
end

# DifferentiableTriangulation

mutable struct DifferentiableTriangulation{Dc,Dp} <: Triangulation{Dc,Dp}
  trian :: Triangulation{Dc,Dp}
  state
end

function DifferentiableTriangulation(trian::Triangulation)
  DifferentiableTriangulation(trian,nothing)
end

(t::DifferentiableTriangulation)(φh) = update_trian!(t,φh)

function update_trian!(trian::DifferentiableTriangulation,φh)
  trian.state = compute_dual_values(trian.trian,φh)
  return trian
end

function FESpaces._change_argument(
  op,f,trian::DifferentiableTriangulation,uh::SingleFieldFEFunction
)
  U = get_fe_space(uh)
  function g(cell_u)
    cf = CellField(U,cell_u)
    update_trian!(trian,cf)
    cell_grad = f(cf)
    get_contribution(cell_grad,trian)
  end
  g
end

function FESpaces._compute_cell_ids(uh,ttrian::DifferentiableTriangulation)
  FESpaces._compute_cell_ids(uh,ttrian.trian)
end

function Geometry.get_background_model(t::DifferentiableTriangulation)
  get_background_model(t.trian)
end

function Geometry.get_grid(t::DifferentiableTriangulation)
  get_grid(t.trian)
end

function CellData.get_cell_points(ttrian::DifferentiableTriangulation)
  if isnothing(ttrian.state)
    return get_cell_points(ttrian.trian)
  end
  cell_to_coords, cell_to_rcoords, _, _ = ttrian.state
  return CellPoint(cell_to_rcoords, cell_to_coords, ttrian, ReferenceDomain())
end

function Geometry.get_cell_map(ttrian::DifferentiableTriangulation)
  if isnothing(ttrian.state)
    return get_cell_map(ttrian.trian)
  end
  _, _, cmaps, _ = ttrian.state
  return cmaps
end

function Geometry.get_glue(ttrian::DifferentiableTriangulation{Dc},val::Val{d}) where {Dc,d}
  if isnothing(ttrian.state)
    get_glue(ttrian.trian,val)
  end
  if d != Dc
    return nothing
  end
  _, _, _, ref_cmaps = ttrian.state
  tface_to_mface = ttrian.trian.subcells.cell_to_bgcell
  tface_to_mface_map = ref_cmaps
  FaceToFaceGlue(tface_to_mface,tface_to_mface_map,nothing)
end
