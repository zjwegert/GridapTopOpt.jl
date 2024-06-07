using Gridap, GridapDistributed, PartitionedArrays

mesh_partition = (2,3);
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(mesh_partition),)))
end

model = CartesianDiscreteModel(ranks,mesh_partition,(0,1,0,1),(100,100));
V_φ = TestFESpace(model,ReferenceFE(lagrangian,Float64,1))
φh = interpolate(x->-cos(4π*x[1])*cos(4*pi*x[2])/4-0.2/4,V_φ)

gids = get_cell_gids(model)
geo = map(local_views(model),local_views(gids),local_views(own_values(get_free_dof_values(φh))),local_views(V_φ)) do bgmodel,gids,φ,V_φ
  ownmodel = GridapEmbedded.Distributed.remove_ghost_cells(bgmodel,gids)
  point_to_coords = collect1d(get_node_coordinates(ownmodel))
  geo = DiscreteGeometry(φ,point_to_coords)
end

# length(geo.items[1].tree.data[1])
# length(geo.items[1].point_to_coords)

cuts = map(local_views(bgmodel),local_views(gids),geo) do bgmodel,gids,geo_loc
  ownmodel = GridapEmbedded.Distributed.remove_ghost_cells(bgmodel,gids)
  cutgeo = cut(cutter,ownmodel,geo_loc)
  GridapEmbedded.Distributed.change_bgmodel(cutgeo,bgmodel,own_to_local(gids))
end
GridapEmbedded.Distributed.consistent_bgcell_to_inoutcut!(cuts,gids)
embedded_dis = GridapEmbedded.Distributed.DistributedEmbeddedDiscretization(cuts,bgmodel)
Ω1 = Triangulation(embedded_dis,PHYSICAL)
writevtk(Ω,"./results/discrete_geo_serial")

# length(geo.items[1].point_to_coords) != length(geo.items[1].tree.data[1])

# own_to_local(gids).items[1]

