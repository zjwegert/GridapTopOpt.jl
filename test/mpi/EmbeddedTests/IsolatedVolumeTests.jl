# module IsolatedVolumeTestsMPI
using Test
using GridapTopOpt
using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using GridapEmbedded.Distributed
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity, Gridap.Arrays

using GridapDistributed, PartitionedArrays

function generate_model(D,n)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  base_model = UnstructuredDiscreteModel(CartesianDiscreteModel(domain,cell_partition))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)
  update_labels!(1,model,x->x[1]≈0,"Gamma_D")
  return model
end

function generate_model(D,n,ranks,mesh_partition)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  base_model = UnstructuredDiscreteModel(CartesianDiscreteModel(ranks,mesh_partition,domain,cell_partition))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)
  update_labels!(1,model,x->x[1]≈0,"Gamma_D")
  return model
end

function main_2d(model,name;vtk=false)
  order = 1

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

  μ = GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_D"])

  if vtk
    writevtk(
      Triangulation(model),"results/background_$name",
      cellfields=[
        "φh"=>φh,
        "μ"=>μ,
      ],
      celldata=[
        "inoutcut"=>bgcell_to_inoutcut,
        "volumes"=>cell_to_color,
      ];
      append=false
    )
  end

  return μ
end

function main_2d_distributed(model,name;vtk=false)
  order = 1

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

  μ = GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_D"])

  _data = map(local_views(bgcell_to_inoutcut),local_views(model)) do bgcell_to_inoutcut,model
    CellField(bgcell_to_inoutcut,Triangulation(model))
  end
  bgcell_to_inoutcut_field = GridapDistributed.DistributedCellField(_data,Triangulation(model))

  _data_cell_to_color = map(local_views(cell_to_color),local_views(model)) do cell_to_color,model
    CellField(cell_to_color,Triangulation(model))
  end
  cell_to_color_field = GridapDistributed.DistributedCellField(_data_cell_to_color,Triangulation(model))

  if vtk
    writevtk(
      Triangulation(model),"results/background_$name",
      cellfields=[
        "φh"=>φh,
        "μ"=>μ,
        "inoutcut"=>bgcell_to_inoutcut_field,
        "volumes"=>cell_to_color_field,
      ],
      append=false
    )
  end

  return μ
end

function main(distribute,mesh_partition;vtk=false)
  # Serial
  serial_model = generate_model(2,40)
  μ_serial = main_2d(serial_model,"serial";vtk)
  μ_serial_val = getfield.(μ_serial.cell_field,:value)

  # Distributed
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  model = generate_model(2,40,ranks,mesh_partition)

  μ = main_2d_distributed(model,"distributed";vtk)
  μ_data = map(local_views(μ)) do μ
    getfield.(μ.cell_field,:value)
  end
  μ_pvec = PVector(μ_data,partition(get_face_gids(model,2)))

  @test μ_serial_val == collect(μ_pvec)

  if vtk
    writevtk(Triangulation(serial_model),"iso_serial",cellfields=["χ_s"=>μ_serial]);
    writevtk(Triangulation(model),"iso_dist",cellfields=["χ_s"=>μ]);
  end
end

with_debug() do distribute
  @test main(distribute,(2,2);vtk=true)
end

# end