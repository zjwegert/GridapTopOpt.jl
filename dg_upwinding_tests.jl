
using Gridap
using Gridap.Geometry, Gridap.FESpaces, Gridap.ReferenceFEs
using GridapDistributed, PartitionedArrays

nprocs = (2,1)
ranks  = with_debug() do distribute
  distribute(LinearIndices((prod(nprocs),)))
end

D  = 2
nc = (10,1)
domain = (0,1,0,1)
model  = CartesianDiscreteModel(ranks,nprocs,domain,nc;isperiodic=(true,false))

order = 2
poly  = QUAD
reffe = LagrangianRefFE(Float64,poly,order)
V = FESpace(model,reffe)

cell_masks = map(local_views(model)) do model
  coords      = get_node_coordinates(model)
  cell_ids    = get_cell_node_ids(model)
  JaggedArray(map(ids -> lazy_map(Reindex(coords),ids),cell_ids))
end
gids = get_cell_gids(model)
p_coords = PVector(cell_coords,partition(gids))
consistent!(p_coords) |> fetch


function mark_nodes(f,model)
  local_masks = map(local_views(model)) do model
    topo   = get_grid_topology(model)
    coords = get_vertex_coordinates(topo)
    mask = map(f,coords)
    return mask
  end
  gids = get_face_gids(model,0)
  mask = PVector(local_masks,partition(gids))
  assemble!(|,mask) |> fetch
  consistent!(mask) |> fetch
  return mask
end

fΓ(x::T) where T = (x ≈ zero(T))
mask = mark_nodes(fΓ,model)


gids = get_face_gids(model,0)
local_masks = map(local_views(model),partition(gids)) do model, gids
  topo   = get_grid_topology(model)
  coords = get_vertex_coordinates(topo)
  mask = map(fΓ,coords)
  println(mask)
  println(local_to_owner(gids))
  return nothing
end



_model = CartesianDiscreteModel(domain,nc)
coords1 = get_node_coordinates(_model)
coords2 = get_vertex_coordinates(get_grid_topology(_model))
