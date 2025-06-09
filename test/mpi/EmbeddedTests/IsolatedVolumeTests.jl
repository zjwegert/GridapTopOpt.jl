module IsolatedVolumeTestsMPI
using Test
using GridapTopOpt
using Gridap
using GridapGmsh

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using GridapEmbedded.Distributed
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity, Gridap.Arrays

using GridapDistributed, PartitionedArrays

function generate_model(D,n,ranks,mesh_partition)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  base_model = UnstructuredDiscreteModel(CartesianDiscreteModel(ranks,mesh_partition,domain,cell_partition))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)
  return model
end

function main_2d_sub(model,name;vtk=false)
  order = 1
  Ω = Triangulation(model)

  reffe = ReferenceFE(lagrangian,Float64,order)
  V_φ = TestFESpace(model,reffe)

  f(x,y0) = abs(x[2]-y0) - 0.05
  g(x,x0,y0,r) = sqrt((x[1]-x0)^2 + (x[2]-y0)^2) - r
  f(x) = min(f(x,0.75),f(x,0.25),
    g(x,0.15,0.5,0.09),
    g(x,0.5,0.6,0.2),
    g(x,0.85,0.5,0.09),
    g(x,0.5,0.15,0.05))
  φh = interpolate(f,V_φ)
  GridapTopOpt.correct_ls!(φh)

  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)

  cell_to_state = map(compute_bgcell_to_inoutcut,local_views(model),local_views(geo))
  cell_to_lcolor, lcolor_to_group = map(local_views(model),cell_to_state) do model, cell_to_state
    GridapTopOpt.tag_disconnected_volumes(model,cell_to_state;groups=((GridapTopOpt.CUT,IN),OUT))
  end |> tuple_of_arrays;

  cell_ids = partition(get_cell_gids(model))

  cell_to_lcolor, lcolor_to_group, color_gids = GridapTopOpt.generate_volume_gids(
    cell_ids,cell_to_lcolor,lcolor_to_group
  )

  cell_to_lcolor, lcolor_to_group, color_gids = GridapTopOpt.tag_disconnected_volumes(model,cell_to_state;groups=((GridapTopOpt.CUT,IN),OUT));

  φ_cell_values = map(get_cell_dof_values,local_views(φh))
  μ,_ = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,[1,2,5,7])

  cell_to_color = map(cell_to_lcolor,partition(color_gids)) do cell_to_lcolor, colors
    local_to_global(colors)[cell_to_lcolor]
  end

  if vtk
    Ω_φ = get_triangulation(V_φ)
    writevtk(
      Ω,"results/IsolatedVol_$name",
      cellfields=[
        "φh"=>φh,
        "μ"=>μ,
        "inoutcut"=>CellField(cell_to_state,Ω_φ),
        "loc_vols"=>CellField(cell_to_lcolor,Ω_φ),
        "vols"=>CellField(cell_to_color,Ω_φ),
      ],
      append=false
    );
  end

  return μ,V_φ
end

function main_2d(distribute,mesh_partition;vtk=false)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  model = generate_model(2,40,ranks,mesh_partition)

  μ,V_φ = main_2d_sub(model,"distributed_$mesh_partition";vtk)

  g(x,x0,y0,r) = sqrt((x[1]-x0)^2 + (x[2]-y0)^2) - r
  f(x) = min(g(x,0.15,0.5,0.09),g(x,0.85,0.5,0.09))
  fh = interpolate(f,V_φ)
  geo = DiscreteGeometry(fh,model)
  cell_to_state = map(compute_bgcell_to_inoutcut,local_views(model),local_views(geo))

  i_am_main(ranks) && println("Testing $mesh_partition")
  map(local_views(μ),cell_to_state) do μ,cell_to_state
    @test getfield.(μ.cell_field,:value)==collect(cell_to_state.<=0)
  end
end

function main_3d_sub(model,name;vtk=false)
  order = 1
  Ω = Triangulation(model)

  reffe = ReferenceFE(lagrangian,Float64,order)
  V_φ = TestFESpace(model,reffe)

  f(x,y0) = abs(x[2]-y0) + abs(x[3]-0.5) - 0.05
  g(x,x0,y0,r) = sqrt((x[1]-x0)^2 + (x[2]-y0)^2 + (x[3]-0.5)^2) - r
  f(x) = min(f(x,0.75),f(x,0.25),
    g(x,0.15,0.5,0.09),
    g(x,0.5,0.6,0.2),
    g(x,0.85,0.5,0.09),
    g(x,0.5,0.15,0.05))
  φh = interpolate(f,V_φ)
  GridapTopOpt.correct_ls!(φh)

  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)

  cell_to_state = map(compute_bgcell_to_inoutcut,local_views(model),local_views(geo))
  cell_to_lcolor, lcolor_to_group = map(local_views(model),cell_to_state) do model, cell_to_state
    GridapTopOpt.tag_disconnected_volumes(model,cell_to_state;groups=((GridapTopOpt.CUT,IN),OUT))
  end |> tuple_of_arrays;

  cell_ids = partition(get_cell_gids(model))
  cell_to_lcolor, lcolor_to_group, color_gids = GridapTopOpt.generate_volume_gids(
    cell_ids,cell_to_lcolor,lcolor_to_group
  )

  cell_to_lcolor, lcolor_to_group, color_gids = GridapTopOpt.tag_disconnected_volumes(model,cell_to_state;groups=((GridapTopOpt.CUT,IN),OUT));

  φ_cell_values = map(get_cell_dof_values,local_views(φh))
  μ,_ = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,[25])

  cell_to_color = map(cell_to_lcolor,partition(color_gids)) do cell_to_lcolor, colors
    local_to_global(colors)[cell_to_lcolor]
  end

  if vtk
    Ω_φ = get_triangulation(V_φ)
    writevtk(
      Ω,"results/IsolatedVol_$name",
      cellfields=[
        "φh"=>φh,
        "μ"=>μ,
        "inoutcut"=>CellField(cell_to_state,Ω_φ),
        "loc_vols"=>CellField(cell_to_lcolor,Ω_φ),
        "vols"=>CellField(cell_to_color,Ω_φ),
      ],
      append=false
    );
  end

  return μ,V_φ
end

function main_3d(distribute,mesh_partition;vtk=false)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  model = generate_model(3,40,ranks,mesh_partition)

  μ,V_φ = main_3d_sub(model,"distributed_$mesh_partition";vtk)

  g(x,x0,y0,r) = sqrt((x[1]-x0)^2 + (x[2]-y0)^2 + (x[3]-0.5)^2) - r
  f(x) = min(g(x,0.15,0.5,0.09),g(x,0.85,0.5,0.09))
  fh = interpolate(f,V_φ)
  geo = DiscreteGeometry(fh,model)
  cell_to_state = map(compute_bgcell_to_inoutcut,local_views(model),local_views(geo))
  μ_expected = map(cell_to_state) do cell_to_state
    collect(Float64,cell_to_state.<=0)
  end

  if vtk
    writevtk(
      get_triangulation(model),"results/expected_IsolatedVol_$mesh_partition",
      cellfields=[
        "μ_expected"=>CellField(μ_expected,get_triangulation(V_φ)),
        "diff"=>CellField(μ_expected,get_triangulation(V_φ))-μ
      ],
      append=false
    );
  end

  i_am_main(ranks) && println("Testing $mesh_partition")
  map(local_views(μ),cell_to_state) do μ,cell_to_state
    @test getfield.(μ.cell_field,:value)==collect(Float64,cell_to_state.<=0)
  end
end

function main_gmsh(ranks;vtk=false)
  path = "./results/IsolatedGmsh_BiteTest_MPI/"
  files_path = path*"data/"
  i_am_main(ranks) && mkpath(files_path)

  # Load gmsh mesh (Currently need to update mesh.geo and these values concurrently)
  H = 0.5;
  x0 = 0.5;
  l = 0.4;
  w = 0.025;
  a = 0.3;
  b = 0.01;

  model = GmshDiscreteModel(ranks,"test/meshes/mesh_finer.msh")
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

  φ_cell_values = map(get_cell_dof_values,local_views(φh))
  ψ_s,_ = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Gamma_s_D"])
  _,ψ_f = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Gamma_f_D"])

  if vtk
    writevtk(get_triangulation(φh),path*"initial_islands",cellfields=["φh"=>φh,"ψ_f"=>ψ_f,"ψ_s"=>ψ_s];append=false)
    writevtk(Ωs,path*"Omega_s_initial";append=false)
    writevtk(Ωf,path*"Omega_f_initial";append=false)
  end
end

with_mpi() do distribute
  main_2d(distribute,(2,2);vtk=false)
  main_2d(distribute,(1,4);vtk=false)
  main_2d(distribute,(4,1);vtk=false)
end

with_debug() do distribute
  main_2d(distribute,(6,10);vtk=false)
  main_2d(distribute,(7,3);vtk=false)
end

with_mpi() do distribute
  main_3d(distribute,(2,2,1);vtk=false)
  main_3d(distribute,(2,1,2);vtk=false)
  main_3d(distribute,(1,4,1);vtk=false)
  main_3d(distribute,(4,1,1);vtk=false)
  main_3d(distribute,(1,1,4);vtk=false)
end

with_debug() do distribute
  main_3d(distribute,(3,4,5);vtk=false)
  main_3d(distribute,(5,5,5);vtk=false)
end

with_mpi() do distribute
  ranks = distribute(LinearIndices((4,)))
  main_gmsh(ranks;vtk=true)
end

end