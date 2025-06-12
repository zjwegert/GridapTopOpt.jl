using GridapTopOpt
using Gridap
using FillArrays

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity, Gridap.Arrays

order = 1
n = 41
model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1),(n,n)))
update_labels!(1,model,x->x[1]≈0,"Gamma_D")
Ω = Triangulation(model)

reffe_scalar = ReferenceFE(lagrangian,Float64,order)
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

bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo)
cell_to_color, color_to_group = GridapTopOpt.tag_disconnected_volumes(model,bgcell_to_inoutcut;groups=((GridapTopOpt.CUT,IN),OUT))
color_to_isolated = GridapTopOpt.find_isolated_volumes(model,["Gamma_D"],cell_to_color,color_to_group)
cell_to_isolated = map(c -> color_to_isolated[c] && isone(color_to_group[c]), cell_to_color)
μ = GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_D"])

writevtk(
  Ω,"results/background",
  cellfields=[
    "φh"=>φh,
    "μ"=>μ,
  ],
  celldata=[
    "inoutcut"=>bgcell_to_inoutcut,
    "volumes"=>cell_to_color,
    "tagged"=>cell_to_isolated,
  ];
  append=false
)

############################################################################################

scmodel, subcell_to_inout, _ = GridapTopOpt.generate_subcell_model(cutgeo)
subcell_to_color, _ = GridapTopOpt.tag_disconnected_volumes(scmodel,subcell_to_inout;groups=(IN,OUT))
μ_IN, μ_OUT = GridapTopOpt.get_isolated_volumes_mask_v2(cutgeo,["Gamma_D"])

writevtk(
  Triangulation(scmodel),"results/subcell",
  celldata=[
    "inout"=>subcell_to_inout,
    "volumes"=>subcell_to_color,
  ];
  append=false
)

writevtk(
  Ω,"results/subcell_isolated",
  cellfields=[
    "φh"=>φh,
    "μ_IN"=>μ_IN,
    "μ_OUT"=>μ_OUT,
  ],
  celldata=[
    "inoutcut"=>bgcell_to_inoutcut,
  ];
  append=false
)
