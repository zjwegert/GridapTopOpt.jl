using GridapTopOpt
using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity, Gridap.Arrays

order = 1
n = 100
model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1),(n,n)))
Ω = Triangulation(model)

reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe_scalar)

φh = interpolate(x->cos(2π*x[1])*cos(2π*x[2])-0.11,V_φ)

# R = 0.195
# R = 0.2 # This fails
# f(x0,r) = x -> sqrt((x[1]-x0[1])^2 + (x[2]-x0[2])^2) - r
# φh = interpolate(x->-f([0.5,0.5],R)(x),V_φ)
# φh = interpolate(x->min(f([0.25,0.5],R)(x),f([0.75,0.5],R)(x)),V_φ)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo)
cell_to_color, color_to_group = GridapTopOpt.tag_isolated_volumes(model,bgcell_to_inoutcut;groups=((GridapTopOpt.CUT,IN),OUT))

color_to_tagged = GridapTopOpt.find_tagged_volumes(model,["tag_5","tag_7"],cell_to_color,color_to_group)
cell_to_tagged = map(c -> color_to_tagged[c], cell_to_color)

μ = GridapTopOpt.get_isolated_volumes_mask(cutgeo,["tag_5","tag_7"])

writevtk(
  Ω,"results/background",
  cellfields=[
    "φh"=>φh,
    "μ"=>μ,
  ],
  celldata=[
    "inoutcut"=>bgcell_to_inoutcut,
    "volumes"=>cell_to_color,
    "tagged"=>cell_to_tagged,
  ];
  append=false
)
