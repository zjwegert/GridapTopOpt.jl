module DistributedDifferentiableCutPolyTriangulationsTests

using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.ReferenceFEs, Gridap.CellData, Gridap.Fields, Gridap.Arrays, Gridap.Helpers, Gridap.FESpaces
using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using GridapEmbedded.LevelSetCutters: DifferentiableTriangulation
using Test

using GridapDistributed
using PartitionedArrays

using GridapTopOpt

function generate_model(D,n,simplex_bgmodel)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  base_model = UnstructuredDiscreteModel(CartesianDiscreteModel(domain,cell_partition))
  if simplex_bgmodel
    ref_model = refine(base_model, refinement_method = "barycentric")
    model = ref_model.model
    return model
  else
    return base_model
  end
end

function run_single_ls_tests(ranks,φ;ls_name,write_vtk,nc=21)
  for D in (2,3)
    for simplex_bgmodel ∈ (true,) # Currently only simplex background models supported for multi-level-set cutting
      function driver(bgmodel,type)
        order = 1
        reffe = ReferenceFE(lagrangian,Float64,order)
        V_φ = TestFESpace(bgmodel,reffe)
        φh1 = interpolate(x->φ(x...),V_φ)
        geo1 = DiscreteGeometryFromFEFunction(φh1,bgmodel)

        cutgeo = cut(PolytopalLevelSetCutter(),bgmodel,geo1)

        Ω = Triangulation(cutgeo)
        diffable_Ω = DifferentiableTriangulation(Ω,cutgeo)

        fh = interpolate(x->cos(x[1]+x[2]),V_φ)

        dΩ = Measure(diffable_Ω,2*order)
        g(fh) = ∇(fh)⋅∇(fh)
        _J(dΩ) = (x) -> ∫(g(fh))dΩ
        dJ = gradient(_J(dΩ),φh1)
        dj = assemble_vector(dJ,V_φ)

        if write_vtk
          writevtk(Triangulation(bgmodel),"results/$(type)_Simplex$(simplex_bgmodel)_D$(D)_$(ls_name)_gradient_Ω",cellfields=["dj"=>FEFunction(V_φ,dj)])
          writevtk(Ω,"results/$(type)_Simplex$(simplex_bgmodel)_D$(D)_$(ls_name)_Ω")
        end

        ###
        Γ = EmbeddedBoundary(cutgeo)
        diffable_Γ = DifferentiableEmbeddedBoundary(Γ,cutgeo)

        dΓ = Measure(diffable_Γ,2*order)
        function J_2(Γ,dΓ)
          function _J_2(x)
            n = get_normal_vector(Γ)
            ∫(∇(fh)⋅n)dΓ
          end
        end
        dJ_2 = gradient(J_2(diffable_Γ,dΓ),φh1)
        dj_2 = assemble_vector(dJ_2,V_φ)

        if write_vtk
          writevtk(Triangulation(bgmodel),"results/$(type)_Simplex$(simplex_bgmodel)_D$(D)_$(ls_name)_gradient_Γ",cellfields=["dj"=>FEFunction(V_φ,dj_2)])
          writevtk(Γ,"results/$(type)_Simplex$(simplex_bgmodel)_D$(D)_$(ls_name)_Γ",cellfields=["n"=>get_normal_vector(Γ)])
        end

        return FEFunction(V_φ,dj),FEFunction(V_φ,dj_2),V_φ
      end

      bgmodel = generate_model(D,nc,simplex_bgmodel)
      serial_dj,serial_dj_2,serial_space = driver(bgmodel,"Serial")

      bgmodel_dist = ordered_distributed_model_from_serial_model(ranks,bgmodel)
      dist_dj,dist_dj_2,dist_space = driver(bgmodel_dist,"Distributed")

      result1 = test_serial_and_distributed_fields(dist_dj,dist_space,serial_dj,serial_space)
      result2 = test_serial_and_distributed_fields(dist_dj_2,dist_space,serial_dj_2,serial_space)
      map(result1,result2) do result1,result2
        @test result1
        @test result2
      end;
    end
  end
end

function run_multi_ls_tests(ranks,φ1,φ2;ls_name,write_vtk,nc=21)
  for D in (2,3)
    for simplex_bgmodel ∈ (true,) # Currently only simplex background models supported for multi-level-set cutting
      function driver(bgmodel,type)
        order = 1
        reffe = ReferenceFE(lagrangian,Float64,order)
        V_φ1 = TestFESpace(bgmodel,reffe)
        V_φ2 = TestFESpace(bgmodel,reffe)
        φh1 = interpolate(x->φ1(x...),V_φ1)
        φh2 = interpolate(x->φ2(x...),V_φ2)
        geo1 = DiscreteGeometryFromFEFunction(φh1,bgmodel,name="φ1")
        geo2 = DiscreteGeometryFromFEFunction(φh2,bgmodel,name="φ2")
        setdiff_geo1_geo2 = setdiff(geo1,geo2,name="Ω1")
        setdiff_geo2_geo1 = setdiff(geo2,geo1,name="Ω2")
        intersect_geo1_geo2 = intersect(geo1,geo2,name="Ω3")
        outside_geo1_geo2 = !(union(geo1,geo2),name="Ω4")
        _all = union(union(union(setdiff_geo1_geo2,setdiff_geo2_geo1),intersect_geo1_geo2),outside_geo1_geo2)

        cutgeo = cut(PolytopalLevelSetCutter(),bgmodel,_all)

        Ω2 = Triangulation(cutgeo,"Ω2")
        diffable_Ω2 = DifferentiableTriangulation(Ω2,cutgeo,PHYSICAL,"Ω2")

        fh = interpolate(x->cos(x[1]+x[2]),V_φ1)

        dΩ2 = Measure(diffable_Ω2,2*order)
        g(fh) = ∇(fh)⋅∇(fh)
        _J(dΩ) = (x) -> ∫(g(fh))dΩ
        dJ = gradient(_J(dΩ2),φh1)
        dj = assemble_vector(dJ,V_φ1)

        if write_vtk
          writevtk(Triangulation(bgmodel),"results/$(type)_Simplex$(simplex_bgmodel)_D$(D)_$(ls_name)_gradient_Ω2",cellfields=["dj"=>FEFunction(V_φ1,dj)])
          writevtk(Ω2,"results/$(type)_Simplex$(simplex_bgmodel)_D$(D)_$(ls_name)_Ω2")
        end

        #######
        cutgeo = cut(PolytopalLevelSetCutter(),bgmodel,_all)

        Γ23 = EmbeddedBoundary(cutgeo,"Ω2","Ω3")
        diffable_Γ23 = DifferentiableEmbeddedBoundary(Γ23,cutgeo,"Ω2","Ω3")

        dΓ23 = Measure(diffable_Γ23,2*order)
        function J_2(Γ,dΓ)
          function _J_2(x)
            n = get_normal_vector(Γ)
            ∫(∇(fh)⋅n)dΓ
          end
        end
        dJ_2 = gradient(J_2(diffable_Γ23,dΓ23),φh1)
        dj_2 = assemble_vector(dJ_2,V_φ1)

        if write_vtk
          writevtk(Triangulation(bgmodel),"results/$(type)_Simplex$(simplex_bgmodel)_D$(D)_$(ls_name)_gradient_Γ23",cellfields=["dj"=>FEFunction(V_φ1,dj_2)])
          writevtk(Γ23,"results/$(type)_Simplex$(simplex_bgmodel)_D$(D)_$(ls_name)_Γ23",cellfields=["n"=>get_normal_vector(Γ23)])
        end
        return FEFunction(V_φ1,dj),FEFunction(V_φ1,dj_2),V_φ1
      end

      bgmodel = generate_model(D,nc,simplex_bgmodel)
      serial_dj,serial_dj_2,serial_space = driver(bgmodel,"Serial")

      bgmodel_dist = ordered_distributed_model_from_serial_model(ranks,bgmodel)
      dist_dj,dist_dj_2,dist_space = driver(bgmodel_dist,"Distributed")

      result1 = test_serial_and_distributed_fields(dist_dj,dist_space,serial_dj,serial_space)
      result2 = test_serial_and_distributed_fields(dist_dj_2,dist_space,serial_dj_2,serial_space)
      map(result1,result2) do result1,result2
        @test result1
        @test result2
      end;
    end
  end
end

## Util
# From GridapTopOpt tests
function test_serial_and_distributed_fields(fhd::CellField,Vd,fhs::FEFunction,Vs)
  fhd_cell_values = map(local_views(Vd),local_views(fhd)) do Vd,fhd
    free = get_free_dof_values(fhd)
    diri = get_dirichlet_dof_values(Vd)
    scatter_free_and_dirichlet_values(Vd,free,diri)
  end

  free = get_free_dof_values(fhs)
  diri = get_dirichlet_dof_values(Vs)
  fhs_cell_values = scatter_free_and_dirichlet_values(Vs,free,diri)

  dmodel = get_background_model(get_triangulation(Vd))
  map(partition(get_cell_gids(dmodel)),fhd_cell_values) do gids,lfhd_cell_values
    lfhd_cell_values ≈ fhs_cell_values[local_to_global(gids)]
  end
end
function ordered_distributed_model_from_serial_model(ranks,model_serial)
  cell_to_part = reduce(vcat,[[i for j in 1:num_cells(model_serial)/length(ranks)] for i in 1:length(ranks)])
  append!(cell_to_part,[length(ranks) for i = 1: num_cells(model_serial) % length(ranks)]...)
  @assert length(cell_to_part) == num_cells(model_serial)
  DiscreteModel(ranks,model_serial,cell_to_part)
end
function test_serial_and_distributed_fields(fhd::GridapDistributed.DistributedMultiFieldCellField,Vd,fhs::Gridap.MultiField.MultiFieldFEFunction,Vs)
  @assert num_fields(fhd)==num_fields(Vd)==num_fields(fhs)==num_fields(Vs)
  result = map(i->test_serial_and_distributed_fields(fhd[i],Vd[i],fhs[i],Vs[i]),1:num_fields(fhd)) |> to_parray_of_arrays
  map(all,result)
end

function run_test(ranks)
  φ_0(x,y) = x^2+y^2-(0.1)^2
  φ_0(x,y,z) = x^2+y^2+z^2-(0.1)^2
  run_single_ls_tests(ranks,φ_0;ls_name="Edge case",write_vtk=false)

  φ_1(x,y) = cos(2π*x)*cos(2π*y)-0.11
  φ_1(x,y,z) = cos(2π*x)*cos(2π*y)*cos(2π*z)-0.11
  run_single_ls_tests(ranks,φ_1;ls_name="Regular",write_vtk=false)

  φ_2(x,y) = (x-0.5)^2+(y-0.5)^2-(0.401)^2
  φ_2(x,y,z) = (x-0.5)^2+(y-0.5)^2+(z-0.5)^2-(0.401)^2
  run_single_ls_tests(ranks,φ_2;ls_name="Sphere",write_vtk=false)

  φ1_2(x,y) = (x-0.5)^2+(y-0.5)^2-(0.401)^2
  φ2_2(x,y) = cos(2π*x)*cos(2π*y)-0.11
  φ1_2(x,y,z) = (x-0.5)^2+(y-0.4)^2+(z-0.5)^2-(0.401)^2
  φ2_2(x,y,z) = cos(2π*x)*cos(2π*y)*cos(2π*z)-0.11
  run_multi_ls_tests(ranks,φ1_2,φ2_2;ls_name="Regular",write_vtk=false)

  true
end

with_mpi() do distribute
  parts = (2,2)
  ranks = distribute(LinearIndices((prod(parts),)))
  run_test(ranks)
end

end
