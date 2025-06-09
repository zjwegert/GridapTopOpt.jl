using Gridap
using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using GridapEmbedded.Distributed
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity, Gridap.Arrays

using GridapDistributed, PartitionedArrays

# function main(distribute,mesh_partition)

mesh_partition = (6,2)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(mesh_partition),)))
end

# ranks = distribute(LinearIndices((prod(mesh_partition),)))
n = 40
model = simplexify(CartesianDiscreteModel(ranks,mesh_partition,(0,1,0,1),(n,n)))
update_labels!(1,model,x->x[1]≈0,"Gamma_D")
reffe_scalar = ReferenceFE(lagrangian,Float64,1)
V_φ = TestFESpace(model,reffe_scalar)

f(x,y0) = abs(x[2]-y0) - 0.05
g(x,x0,y0,r) = sqrt((x[1]-x0)^2 + (x[2]-y0)^2) - r
f(x) = min(f(x,0.75),f(x,0.25),
  g(x,0.15,0.5,0.1),
  g(x,0.5,0.6,0.2),
  g(x,0.85,0.5,0.1),
  g(x,0.5,0.15,0.05))
φh = interpolate(f,V_φ)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

bgcell_to_inoutcut = compute_bgcell_to_inoutcut(cutgeo,geo)
cell_to_color, color_to_group, color_gids = GridapTopOpt.tag_isolated_volumes(model,bgcell_to_inoutcut;groups=((GridapTopOpt.CUT,IN),OUT));

consit_cell_to_color = map(cell_to_color,local_to_global(color_gids)) do cell_to_color,color_gids
  map(c-> c == 0 ? 0 : color_gids[c],cell_to_color)
end

μ = GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_D"])

_data = map(local_views(bgcell_to_inoutcut),local_views(model)) do bgcell_to_inoutcut,model
  CellField(bgcell_to_inoutcut,Triangulation(model))
end
bgcell_to_inoutcut_field = GridapDistributed.DistributedCellField(_data,Triangulation(model))

_data_cell_to_color = map(local_views(cell_to_color),local_views(model),local_views(color_gids)) do cell_to_color,model,color_gids
  CellField(cell_to_color,Triangulation(model))
end
cell_to_color_field = GridapDistributed.DistributedCellField(_data_cell_to_color,Triangulation(model))

_data_cell_to_color = map(local_views(consit_cell_to_color),local_views(model),local_views(color_gids)) do cell_to_color,model,color_gids
  CellField(cell_to_color,Triangulation(model))
end
consit_cell_to_color_field = GridapDistributed.DistributedCellField(_data_cell_to_color,Triangulation(model))

writevtk(
  Triangulation(model),"results/background",
  cellfields=[
    "φh"=>φh,
    "μ"=>μ,
    "inoutcut"=>bgcell_to_inoutcut_field,
    "volumes"=>cell_to_color_field,
    "gid_vols"=>consit_cell_to_color_field
  ],
  append=false
);
# end

# mesh_partition = (2,2)
# with_mpi() do distribute
#   main(distribute, mesh_partition)
# end