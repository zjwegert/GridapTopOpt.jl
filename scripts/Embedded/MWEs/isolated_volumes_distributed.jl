
using GridapTopOpt
using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity, Gridap.Arrays

using GridapDistributed, PartitionedArrays

parts = (12,10)
ranks = DebugArray(LinearIndices((prod(parts),)))

order = 1
n = 100
model = UnstructuredDiscreteModel(CartesianDiscreteModel(ranks,parts,(0,1,0,1),(n,n)))
Ω = Triangulation(model)

reffe = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe)

# φh = interpolate(x->cos(2π*x[1])*cos(2π*x[2])-0.11,V_φ)

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

cell_to_state = map(compute_bgcell_to_inoutcut,local_views(model),local_views(geo))
cell_to_lcolor, lcolor_to_group = map(local_views(model),cell_to_state) do model, cell_to_state
  GridapTopOpt.tag_isolated_volumes(model,cell_to_state;groups=((GridapTopOpt.CUT,IN),OUT))
end |> tuple_of_arrays;

n_lcolor = map(length,lcolor_to_group)
cell_ids = partition(get_cell_gids(model))

color_gids = GridapTopOpt.generate_volume_gids(
  cell_ids,n_lcolor,cell_to_lcolor
)

cell_to_lcolor, lcolor_to_group, color_gids = GridapTopOpt.tag_isolated_volumes(model,cell_to_state;groups=((GridapTopOpt.CUT,IN),OUT));

μ = GridapTopOpt.get_isolated_volumes_mask(cutgeo,[1,2,5,7])

cell_to_color = map(cell_to_lcolor,partition(color_gids)) do cell_to_lcolor, colors
  local_to_global(colors)[cell_to_lcolor]
end

Ω_φ = get_triangulation(V_φ)
writevtk(
  Ω,"results/background",
  cellfields=[
    "φh"=>φh,
    "μ"=>μ,
    "inoutcut"=>CellField(cell_to_state,Ω_φ),
    "loc_vols"=>CellField(cell_to_lcolor,Ω_φ),
    "vols"=>CellField(cell_to_color,Ω_φ),
  ],
  append=false
);

p = 4
getitem(x) = getitem(local_views(x))
getitem(x::AbstractArray) = x.items[p]
writevtk(
  Ω_φ.trians.items[p],"results/background_$p",
  cellfields=[
    "μ"=>getitem(μ),
    "inoutcut"=>CellField(getitem(cell_to_state),getitem(Ω_φ)),
    "loc_vols"=>CellField(getitem(cell_to_lcolor),getitem(Ω_φ)),
    "vols"=>CellField(getitem(cell_to_color),getitem(Ω_φ)),
  ],
  append=false
);
