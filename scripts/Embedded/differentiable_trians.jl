
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
  return collect(Vector{Int8},unique(sort,vcat(ltcell_to_edges...)))
end

function belongs_to_edge(
  p::Point{D,T},edge::Vector{<:Integer},bgpts::Vector{Point{D,T}}
) where {D,T}
  tol = 10*eps(T)
  p1, p2 = bgpts[edge]
  return abs(cross(p-p1,p2-p1)) < tol
end

struct DualizeCoordsMap <: Map end

function Arrays.return_cache(
  k::DualizeCoordsMap,
  coords::Vector{<:Point{Dp,Tp}},
  bg_coords::Vector{<:Point{Dp,Tp}},
  values::Vector{Tv},
  edges::Vector{Int8},
  edge_list::Vector{Vector{Int8}}
) where {Dp,Tp,Tv}
  T = Point{Dp,Tv}
  return CachedArray(zeros(T, length(coords)))
end

function Arrays.evaluate!(
  cache,
  k::DualizeCoordsMap,
  coords::Vector{<:Point{Dp,Tp}},
  bg_coords::Vector{<:Point{Dp,Tp}},
  values::Vector{Tv},
  edges::Vector{Int8},
  edge_list::Vector{Vector{Int8}}
) where {Dp,Tp,Tv}
  setsize!(cache,(length(coords),))
  new_coords = cache.array
  for (i,e) in enumerate(edges)
    if e == -1
      new_coords[i] = coords[i]
    else
      n1, n2 = edge_list[e]
      q1, q2 = bg_coords[n1], bg_coords[n2]
      v1, v2 = abs(values[n1]), abs(values[n2])
      λ = v1/(v1+v2)
      new_coords[i] = q1 + λ*(q2-q1)
    end
  end
  return new_coords
end

function precompute_cut_edge_ids(
  rcoords::Vector{<:Point{Dp,Tp}},
  bg_rcoords::Vector{<:Point{Dp,Tp}},
  edge_list::Vector{<:Vector{<:Integer}}
) where {Dp,Tp}
  tol = 10*eps(Tp)
  edges = Vector{Int8}(undef,length(rcoords))
  for (i,p) in enumerate(rcoords)
    if any(q -> norm(q-p) < tol, bg_rcoords)
      edges[i] = Int8(-1)
    else
      e = findfirst(edge -> belongs_to_edge(p,edge,bg_rcoords), edge_list)
      edges[i] = Int8(e)
    end
  end
  return edges
end

function precompute_cut_edge_ids(
  cell_to_rcoords::AbstractArray,
  cell_to_bg_rcoords::AbstractArray,
  cell_to_edge_list::AbstractArray
)
  collect(lazy_map(precompute_cut_edge_ids,cell_to_rcoords,cell_to_bg_rcoords,cell_to_edge_list))
end

function dualize_cell_coordinates(
  trian::GridapEmbedded.Interfaces.SubCellTriangulation,
  φh::CellField
)
  bgmodel = get_background_model(trian)
  subcells = trian.subcells

  cell_to_bgcell   = subcells.cell_to_bgcell
  cell_to_points   = subcells.cell_to_points
  point_to_rcoords = subcells.point_to_rcoords
  point_to_coords  = subcells.point_to_coords

  bg_ctypes = get_cell_type(bgmodel)

  bgcell_to_polys = expand_cell_data(get_polytopes(bgmodel),bg_ctypes)
  bgcell_to_coords = get_cell_coordinates(bgmodel)
  bgcell_to_rcoords = lazy_map(get_vertex_coordinates,bgcell_to_polys)
  bgcell_to_edge_lists = lazy_map(get_edge_list,bgcell_to_polys)
  bgcell_to_values = lazy_map(evaluate,CellData.get_data(φh),bgcell_to_rcoords)
  
  cell_to_reffes = Fill(LagrangianRefFE(Float64,TRI,1),length(cell_to_points))
  cell_to_bgcoords = lazy_map(Reindex(bgcell_to_coords),cell_to_bgcell)
  cell_to_bgrcoords = lazy_map(Reindex(bgcell_to_rcoords),cell_to_bgcell)
  cell_to_values = lazy_map(Reindex(bgcell_to_values),cell_to_bgcell)
  cell_to_edge_lists = lazy_map(Reindex(bgcell_to_edge_lists),cell_to_bgcell)
  
  old_cell_to_rcoords = lazy_map(Broadcasting(Reindex(point_to_rcoords)),cell_to_points)
  old_cell_to_coords = lazy_map(Broadcasting(Reindex(point_to_coords)),cell_to_points)

  cell_to_edges = precompute_cut_edge_ids(old_cell_to_rcoords,cell_to_bgrcoords,cell_to_edge_lists)
  
  new_cell_to_coords = lazy_map(DualizeCoordsMap(),old_cell_to_coords,cell_to_bgcoords,cell_to_values,cell_to_edges,cell_to_edge_lists)
  new_cell_to_rcoords = lazy_map(DualizeCoordsMap(),old_cell_to_rcoords,cell_to_bgrcoords,cell_to_values,cell_to_edges,cell_to_edge_lists)
  
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
  trian.state = dualize_cell_coordinates(trian.trian,φh)
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
