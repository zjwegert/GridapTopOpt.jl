module IsolatedVolumeTestsMPI
using Test
using GridapTopOpt
using Gridap

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

function main_2d(model,name;vtk=false)
  order = 1
  Ω = Triangulation(model)

  reffe = ReferenceFE(lagrangian,Float64,order)
  V_φ = TestFESpace(model,reffe)

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

  if vtk
    Ω_φ = get_triangulation(V_φ)
    writevtk(
      Ω,"results/background_$name",
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

function main(distribute,mesh_partition;vtk=false)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  model = generate_model(2,40,ranks,mesh_partition)

  μ,V_φ = main_2d(model,"distributed";vtk)

  g(x,x0,y0,r) = sqrt((x[1]-x0)^2 + (x[2]-y0)^2) - r
  f(x) = min(g(x,0.15,0.5,0.1),g(x,0.85,0.5,0.1))
  fh = interpolate(f,V_φ)
  geo = DiscreteGeometry(fh,model)
  cell_to_state = map(compute_bgcell_to_inoutcut,local_views(model),local_views(geo))

  map(local_views(μ),cell_to_state) do μ,cell_to_state
    @test getfield.(μ.cell_field,:value)==collect(cell_to_state.<=0)
  end
end

with_mpi() do distribute
  main(distribute,(2,2);vtk=false)
  main(distribute,(1,4);vtk=false)
  main(distribute,(4,1);vtk=false)
end

with_debug() do distribute
  main(distribute,(6,10);vtk=false)
  main(distribute,(7,3);vtk=false)
end

end