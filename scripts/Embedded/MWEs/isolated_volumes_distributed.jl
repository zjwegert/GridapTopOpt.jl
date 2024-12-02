
using GridapTopOpt
using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity, Gridap.Arrays

using GridapDistributed, PartitionedArrays

parts = (1,2)
ranks = DebugArray(LinearIndices((prod(parts),)))

order = 1
n = 8
model = UnstructuredDiscreteModel(CartesianDiscreteModel(ranks,parts,(0,1,0,1),(n,n)))
Ω = Triangulation(model)

reffe = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe)

φh = interpolate(x->cos(2π*x[1])*cos(2π*x[2])-0.11,V_φ)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

cell_to_state = map(compute_bgcell_to_inoutcut,local_views(model),local_views(geo))
cell_to_lcolor, lcolor_to_group = map(local_views(model),cell_to_state) do model, cell_to_state
  GridapTopOpt.tag_isolated_volumes(model,cell_to_state;groups=((GridapTopOpt.CUT,IN),OUT))
end |> tuple_of_arrays;

n_lcolor = map(length,lcolor_to_group)
cell_ids = partition(get_cell_gids(model))

color_gids = GridapTopOpt.generate_volume_gids(
  cell_ids,n_lcolor,cell_to_lcolor
)


