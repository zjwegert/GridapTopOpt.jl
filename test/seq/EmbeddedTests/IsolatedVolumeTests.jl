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

function main_gmsh(;vtk=false)
  path = "./results/IsolatedGmsh_BiteTest/"
  files_path = path*"data/"
  mkpath(files_path)

  # Load gmsh mesh (Currently need to update mesh.geo and these values concurrently)
  H = 0.5;
  x0 = 0.5;
  l = 0.4;
  w = 0.025;
  a = 0.3;
  b = 0.01;

  model = GmshDiscreteModel("test/meshes/mesh_finer.msh")
  vtk && writevtk(model,path*"model")

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)

  _e = 1e-3
  f0((x,y),W,H) = max(2/W*abs(x-x0),1/(H/2+1)*abs(y-H/2+1))-1
  f1((x,y),q,r) = - cos(q*π*x)*cos(q*π*y)/q - r/q
  fin(x) = f0(x,l*(1+5_e),a*(1+5_e))
  fsolid(x) = min(f0(x,l*(1+_e),b*(1+_e)),f0(x,w*(1+_e),a*(1+_e)))
  fholes((x,y),q,r) = max(f1((x,y),q,r),f1((x-1/q,y),q,r))
  φf(x) = min(max(fin(x),fholes(x,25,0.2)),fsolid(x))

  _φf2(x) = max(φf(x),-(max(2/0.2*abs(x[1]-0.319),2/0.2*abs(x[2]-0.3))-1))
  φf2(x) = min(_φf2(x),sqrt((x[1]-0.35)^2+(x[2]-0.26)^2)-0.025)
  φh = interpolate(φf2,V_φ)

  # Setup integration meshes and measures
  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)

  Ωs = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
  Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)

  ψ_s =  GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_s_D"];groups=((GridapTopOpt.CUT,IN),OUT))
  ψ_f =  GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_f_D"];groups=((GridapTopOpt.CUT,OUT),IN))

  if vtk
    writevtk(get_triangulation(φh),path*"initial_islands",cellfields=["φh"=>φh,"ψ_f"=>ψ_f,"ψ_s"=>ψ_s];append=false)
    writevtk(Ωs,path*"Omega_s_initial";append=false)
    writevtk(Ωf,path*"Omega_f_initial";append=false)
  end
end

main_2d(41;vtk=false)
main_2d(100;vtk=false)

main_gmsh(;vtk=true)

end