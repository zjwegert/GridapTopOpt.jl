module IsolatedVolumeTests
using Test
using GridapTopOpt
using Gridap
using GridapGmsh

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using GridapEmbedded.Distributed
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity, Gridap.Arrays

using GridapDistributed, PartitionedArrays

function main_2d(n;vtk,two_refinement=false)
  order = 1
  base_model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1),(n,n)))
  ref_model = refine(base_model, refinement_method = "barycentric")
  if two_refinement
    ref_model = refine(ref_model)
  end
  model = Adaptivity.get_model(ref_model)
  Ω = Triangulation(model)

  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V_φ = TestFESpace(model,reffe_scalar)

  f(x,y0) = abs(x[2]-y0) - 0.05
  g(x,x0,y0,r) = sqrt((x[1]-x0)^2 + (x[2]-y0)^2) - r
  q(x) = min(f(x,0.75),f(x,0.25),
    g(x,0.15,0.5,0.09),
    g(x,0.5,0.6,0.2),
    g(x,0.85,0.5,0.09),
    g(x,0.5,0.15,0.05))
  φh = interpolate(q,V_φ)
  GridapTopOpt.correct_ls!(φh)

  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)

  bgcell_to_inoutcut = compute_bgcell_to_inoutcut(cutgeo,geo)
  cell_to_color, color_to_group = GridapTopOpt.tag_disconnected_volumes(model,bgcell_to_inoutcut;groups=((GridapTopOpt.CUT,IN),OUT))

  color_to_tagged = GridapTopOpt.find_isolated_volumes(model,[1,2,5,7],cell_to_color,color_to_group)
  cell_to_tagged = map(c -> color_to_tagged[c], cell_to_color)

  φ_cell_values = get_cell_dof_values(φh)
  μ,_ = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,[1,2,5,7])

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

  # Expected
  f2(x) = min(g(x,0.15,0.5,0.09),g(x,0.85,0.5,0.09))
  fh = interpolate(f2,V_φ)
  _geo = DiscreteGeometry(fh,model)
  _bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,_geo)
  _data = CellField(collect(Float64,_bgcell_to_inoutcut .<= 0),Triangulation(model))

  @test get_data(μ) == get_data(_data)
end

function main_3d(n;vtk)
  order = 1
  base_model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1,0,1),(n,n,n)))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)
  Ω = Triangulation(model)

  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V_φ = TestFESpace(model,reffe_scalar)

  f(x,y0) = abs(x[2]-y0) + abs(x[3]-0.5) - 0.05
  g(x,x0,y0,r) = sqrt((x[1]-x0)^2 + (x[2]-y0)^2 + (x[3]-0.5)^2) - r
  q(x) = min(f(x,0.75),f(x,0.25),
    g(x,0.15,0.5,0.09),
    g(x,0.5,0.6,0.2),
    g(x,0.85,0.5,0.09),
    g(x,0.5,0.15,0.05))
  φh = interpolate(q,V_φ)
  GridapTopOpt.correct_ls!(φh)

  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)

  bgcell_to_inoutcut = compute_bgcell_to_inoutcut(cutgeo,geo)
  cell_to_color, color_to_group = GridapTopOpt.tag_disconnected_volumes(model,bgcell_to_inoutcut;groups=((GridapTopOpt.CUT,IN),OUT))

  color_to_tagged = GridapTopOpt.find_isolated_volumes(model,[25,],cell_to_color,color_to_group)
  cell_to_tagged = map(c -> color_to_tagged[c], cell_to_color)

  φ_cell_values = get_cell_dof_values(φh)
  μ,_ = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,[25,])

  if vtk
    writevtk(
      Ω,"results/background_3d",
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

  # Expected
  f_exp(x) = min(g(x,0.15,0.5,0.09),g(x,0.85,0.5,0.09))
  fh = interpolate(f_exp,V_φ)
  _geo = DiscreteGeometry(fh,model)
  _bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,_geo)
  _data = CellField(collect(Float64,_bgcell_to_inoutcut .<= 0),Triangulation(model))

  @test get_data(μ) == get_data(_data)
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
  GridapTopOpt.correct_ls!(φh)

  # Setup integration meshes and measures
  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)

  Ωs = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
  Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)

  φ_cell_values = get_cell_dof_values(φh)
  ψ_s,_ = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Gamma_s_D"])
  _,ψ_f = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Gamma_f_D"])

  if vtk
    writevtk(get_triangulation(φh),path*"initial_islands",cellfields=["φh"=>φh,"ψ_f"=>ψ_f,"ψ_s"=>ψ_s];append=false)
    writevtk(Ωs,path*"Omega_s_initial";append=false)
    writevtk(Ωf,path*"Omega_f_initial";append=false)
  end
end

main_2d(41;vtk=false)
main_2d(41;vtk=false,two_refinement=true)
main_2d(101;vtk=false)
main_3d(31;vtk=false)
main_gmsh(;vtk=true)

end