module IsolatedVolumeTests
using Test
using GridapTopOpt
using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity, Gridap.Arrays

function main_2d(n;vtk)
  order = 1
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

  bgcell_to_inoutcut = compute_bgcell_to_inoutcut(cutgeo,geo)
  cell_to_color, color_to_group = GridapTopOpt.tag_isolated_volumes(model,bgcell_to_inoutcut;groups=((GridapTopOpt.CUT,IN),OUT))

  color_to_tagged = GridapTopOpt.find_tagged_volumes(model,["Gamma_D"],cell_to_color,color_to_group)
  cell_to_tagged = map(c -> color_to_tagged[c], cell_to_color)

  μ = GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_D"])

  # Expected
  f(x) = min(g(x,0.15,0.5,0.1),g(x,0.85,0.5,0.1))
  fh = interpolate(f,V_φ)
  _geo = DiscreteGeometry(fh,model)
  _bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,_geo)
  _data = CellField(collect(Float64,_bgcell_to_inoutcut .<= 0),Triangulation(model))

  @test get_data(μ) == get_data(_data)

  if vtk
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
  end
end

main_2d(41;vtk=false)
main_2d(100;vtk=false)

end