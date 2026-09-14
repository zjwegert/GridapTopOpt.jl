module DifferentiableCutPolyTriangulationsTests

using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.ReferenceFEs, Gridap.CellData, Gridap.Fields, Gridap.Arrays, Gridap.Helpers, Gridap.FESpaces
using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using FiniteDiff
using Test

using GridapTopOpt

function generate_model(D,n,simplex_bgmodel)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  base_model = UnstructuredDiscreteModel((CartesianDiscreteModel(domain,cell_partition)))
  if simplex_bgmodel
    ref_model = refine(base_model, refinement_method = "barycentric")
    model = ref_model.model
    return model
  else
    return base_model
  end
end

function run_single_ls_tests(φ;ls_name,write_vtk)
  for D in (2,3)
    # Currently only simplex background models supported for multi-level-set cutting
    # Technically, these tests (simplex_bgmodel = false) pass in 3D, but there are degenerate cases with cutter.
    for simplex_bgmodel ∈ (true,)
      n = (D == 3) ? 11 : 21;
      bgmodel = generate_model(D,n,simplex_bgmodel)
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

      function fdm_compute(φ)
        φh1 = FEFunction(V_φ,φ)
        geo1 = DiscreteGeometryFromFEFunction(φh1,bgmodel)
        cutgeo = cut(PolytopalLevelSetCutter(),bgmodel,geo1)
        Ω = Triangulation(cutgeo)
        dΩ = Measure(Ω,2*order)
        sum(_J(dΩ)(nothing))
      end
      dJ_FD = FiniteDiff.finite_difference_gradient(fdm_compute,get_free_dof_values(φh1))

      @test maximum(abs,dJ_FD - dj)/maximum(abs,dJ_FD) < 1e-6

      if write_vtk
        writevtk(Triangulation(bgmodel),"results/Simplex$(simplex_bgmodel)_D$(D)_$(ls_name)_gradient_Ω",cellfields=["dj"=>FEFunction(V_φ,dj)])
        writevtk(Ω,"results/Simplex$(simplex_bgmodel)_D$(D)_$(ls_name)_Ω")
      end

      ###
      φh1 = interpolate(x->φ(x...),V_φ)
      geo1 = DiscreteGeometryFromFEFunction(φh1,bgmodel)

      cutgeo = cut(PolytopalLevelSetCutter(),bgmodel,geo1)

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

      function fdm_compute_2(φ)
        φh1 = FEFunction(V_φ,φ)
        geo1 = DiscreteGeometryFromFEFunction(φh1,bgmodel)
        cutgeo = cut(PolytopalLevelSetCutter(),bgmodel,geo1)
        Γ = EmbeddedBoundary(cutgeo)
        dΓ = Measure(Γ,2*order)
        sum(J_2(Γ,dΓ)(nothing))
      end
      dJ_2_FD = FiniteDiff.finite_difference_gradient(fdm_compute_2,get_free_dof_values(φh1))

      @test maximum(abs,dj_2 - dJ_2_FD)/maximum(abs,dJ_2_FD) < 1e-6

      if write_vtk
        writevtk(Triangulation(bgmodel),"results/Simplex$(simplex_bgmodel)_D$(D)_$(ls_name)_gradient_Γ",cellfields=["dj"=>FEFunction(V_φ,dj_2)])
        writevtk(Γ,"results/Simplex$(simplex_bgmodel)_D$(D)_$(ls_name)_Γ",cellfields=["n"=>get_normal_vector(Γ)])
      end
    end
  end
end

function run_multi_ls_tests(φ1,φ2;ls_name,write_vtk)
  for D in (2,3)
    # Currently only simplex background models supported for multi-level-set cutting
    # Technically, these tests (simplex_bgmodel = false) pass in 3D, but there are degenerate cases with cutter.
    for simplex_bgmodel ∈ (true,)
      if D == 3
        continue # It takes a very long time to run this test in 3D
      end
      n = 21;
      bgmodel = generate_model(D,n,simplex_bgmodel)
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

      function fdm_compute(φ)
        φh1 = FEFunction(V_φ1,φ)
        geo1 = DiscreteGeometryFromFEFunction(φh1,bgmodel,name="φ1")
        geo2 = DiscreteGeometryFromFEFunction(φh2,bgmodel,name="φ2")
        setdiff_geo1_geo2 = setdiff(geo1,geo2,name="Ω1")
        setdiff_geo2_geo1 = setdiff(geo2,geo1,name="Ω2")
        intersect_geo1_geo2 = intersect(geo1,geo2,name="Ω3")
        outside_geo1_geo2 = !(union(geo1,geo2),name="Ω4")
        _all = union(union(union(setdiff_geo1_geo2,setdiff_geo2_geo1),intersect_geo1_geo2),outside_geo1_geo2)
        cutgeo = cut(PolytopalLevelSetCutter(),bgmodel,_all)
        Ω = Triangulation(cutgeo,"Ω2")
        dΩ = Measure(Ω,2*order)
        sum(_J(dΩ)(nothing))
      end
      dJ_FD = FiniteDiff.finite_difference_gradient(fdm_compute,get_free_dof_values(φh1))

      @test maximum(abs,dJ_FD - dj)/maximum(abs,dJ_FD) < 1e-6

      if write_vtk
        writevtk(Triangulation(bgmodel),"results/Simplex$(simplex_bgmodel)_D$(D)_$(ls_name)_gradient_Ω2",cellfields=["dj"=>FEFunction(V_φ1,dj)])
        writevtk(Ω2,"results/Simplex$(simplex_bgmodel)_D$(D)_$(ls_name)_Ω2")
      end

      #######
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

      function fdm_compute_2(φ)
        φh1 = FEFunction(V_φ1,φ)
        geo1 = DiscreteGeometryFromFEFunction(φh1,bgmodel,name="φ1")
        geo2 = DiscreteGeometryFromFEFunction(φh2,bgmodel,name="φ2")
        setdiff_geo1_geo2 = setdiff(geo1,geo2,name="Ω1")
        setdiff_geo2_geo1 = setdiff(geo2,geo1,name="Ω2")
        intersect_geo1_geo2 = intersect(geo1,geo2,name="Ω3")
        outside_geo1_geo2 = !(union(geo1,geo2),name="Ω4")
        _all = union(union(union(setdiff_geo1_geo2,setdiff_geo2_geo1),intersect_geo1_geo2),outside_geo1_geo2)
        cutgeo = cut(PolytopalLevelSetCutter(),bgmodel,_all)
        Γ23 = EmbeddedBoundary(cutgeo,"Ω2","Ω3")
        dΓ23 = Measure(Γ23,2*order)
        sum(J_2(Γ23,dΓ23)(nothing))
      end
      dJ_2_FD = FiniteDiff.finite_difference_gradient(fdm_compute_2,get_free_dof_values(φh1))

      @test maximum(abs,dj_2 - dJ_2_FD)/maximum(abs,dJ_2_FD) < 1e-6

      if write_vtk
        writevtk(Triangulation(bgmodel),"results/Simplex$(simplex_bgmodel)_D$(D)_$(ls_name)_gradient_Γ23",cellfields=["dj"=>FEFunction(V_φ1,dj_2)])
        writevtk(Γ23,"results/Simplex$(simplex_bgmodel)_D$(D)_$(ls_name)_Γ23",cellfields=["n"=>get_normal_vector(Γ23)])
      end
    end
  end
end

function run_test()
  φ_1(x,y) = cos(2π*x)*cos(2π*y)-0.11
  φ_1(x,y,z) = cos(2π*x)*cos(2π*y)*cos(2π*z)-0.11
  run_single_ls_tests(φ_1;ls_name="Regular",write_vtk=false)

  φ_2(x,y) = (x-0.5)^2+(y-0.5)^2-(0.401)^2
  φ_2(x,y,z) = (x-0.5)^2+(y-0.5)^2+(z-0.5)^2-(0.401)^2
  run_single_ls_tests(φ_2;ls_name="Sphere",write_vtk=false)

  φ1_1(x,y) = (x-0.5)^2+(y-0.4)^2-(0.401)^2
  φ2_1(x,y) = (x-0.5)^2+(y-0.6)^2-(0.401)^2
  φ1_1(x,y,z) = (x-0.5)^2+(y-0.4)^2+(z-0.5)^2-(0.401)^2
  φ2_1(x,y,z) = (x-0.5)^2+(y-0.6)^2+(z-0.5)^2-(0.401)^2
  run_multi_ls_tests(φ1_1,φ2_1;ls_name="Sphere",write_vtk=false)

  φ1_2(x,y) = (x-0.5)^2+(y-0.5)^2-(0.401)^2
  φ2_2(x,y) = cos(2π*x)*cos(2π*y)-0.11
  φ1_2(x,y,z) = (x-0.5)^2+(y-0.4)^2+(z-0.5)^2-(0.401)^2
  φ2_2(x,y,z) = cos(2π*x)*cos(2π*y)*cos(2π*z)-0.11
  run_multi_ls_tests(φ1_2,φ2_2;ls_name="Regular",write_vtk=false)
end

function cell_measures_test()
  model = generate_model(2,21,true)
  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  V_φs = MultiFieldFESpace([TestFESpace(model,reffe) for _ in 1:2])

  φ1(x,y) = (x-0.5)^2+(y-0.5)^2-(0.401)^2
  φ2(x,y) = cos(2π*x)*cos(2π*y)-0.11

  φh1 = interpolate(x->φ1(x...),V_φs[1])
  φh2 = interpolate(x->φ2(x...),V_φs[2])

  function compute_geo(φh1, φh2)
    geo1 = DiscreteGeometryFromFEFunction(φh1,model,name="φ1")
    geo2 = DiscreteGeometryFromFEFunction(φh2,model,name="φ2")
    setdiff_geo1_geo2 = setdiff(geo1,geo2,name="Ω1")
    setdiff_geo2_geo1 = setdiff(geo2,geo1,name="Ω2")
    intersect_geo1_geo2 = intersect(geo1,geo2,name="Ω3")
    outside_geo1_geo2 = !(union(geo1,geo2),name="Ω4")

    _all = union(union(union(setdiff_geo1_geo2,setdiff_geo2_geo1),intersect_geo1_geo2),outside_geo1_geo2)
    cutgeo = cut(PolytopalLevelSetCutter(),model,_all)
    (;
      cutgeo,
      Ω1 = Triangulation(cutgeo,PHYSICAL,"Ω1"),
      Ω2 = Triangulation(cutgeo,PHYSICAL,"Ω2"),
      Ω3 = Triangulation(cutgeo,PHYSICAL,"Ω3"),
      Ω4 = Triangulation(cutgeo,PHYSICAL,"Ω4"),
      Γ12 = EmbeddedBoundary(cutgeo,"Ω1","Ω2"),
      Γ13 = EmbeddedBoundary(cutgeo,"Ω1","Ω3"),
      Γ14 = EmbeddedBoundary(cutgeo,"Ω1","Ω4"),
      Γ23 = EmbeddedBoundary(cutgeo,"Ω2","Ω3"),
      Γ24 = EmbeddedBoundary(cutgeo,"Ω2","Ω4"),
      Γ34 = EmbeddedBoundary(cutgeo,"Ω3","Ω4")
    )
  end
  cutgeo,Ω1,Ω2,Ω3,Ω4,Γ12,Γ13,Γ14,Γ23,Γ24,Γ34 = compute_geo(φh1, φh2)

  # Ω2 measure
  diffable_Ω2 = DifferentiableTriangulation(Ω2,cutgeo,PHYSICAL,"Ω2")
  diffable_Γ23 = DifferentiableEmbeddedBoundary(Γ23,cutgeo,"Ω2","Ω3")
  dΓ23 = Measure(diffable_Γ23,2*order)
  Ω_bg = Triangulation(model)
  function _J(φh)
    meas_K1 = get_cell_measure(diffable_Ω2, Ω_bg, diffable_Γ23)
    κ1 = CellField( meas_K1, Ω_bg)
    ∫( κ1 )dΓ23
  end

  dJ = gradient(_J,φh1)
  dj = assemble_vector(dJ,V_φs[1])

  function fdm_compute(φ)
    _φh1 = FEFunction(V_φs[1],φ)
    cutgeo,Ω1,Ω2,Ω3,Ω4,Γ12,Γ13,Γ14,Γ23,Γ24,Γ34 = compute_geo(_φh1, φh2)
    meas_K1 = get_cell_measure(Ω2, Ω_bg)
    κ1 = CellField( meas_K1, Ω_bg)
    _dΓ23 = Measure(Γ23,2*order)
    sum(∫( κ1 )_dΓ23)
  end
  dJ_fd = FiniteDiff.finite_difference_gradient(fdm_compute,get_free_dof_values(φh1))

  @test maximum(abs,dj-dJ_fd)/maximum(abs,dj) < 1e-6

  # Γ23 measure
  diffable_Γ23 = DifferentiableEmbeddedBoundary(Γ23,cutgeo,"Ω2","Ω3")
  dΓ23 = Measure(diffable_Γ23,2*order)
  Ω_bg = Triangulation(model)
  function _J(φh)
    meas_K1 = get_cell_measure(diffable_Γ23, Ω_bg)
    κ1 = CellField( meas_K1, Ω_bg)
    ∫( κ1 )dΓ23
  end

  dJ = gradient(_J,φh1)
  dj = assemble_vector(dJ,V_φs[1])

  function fdm_compute(φ)
    _φh1 = FEFunction(V_φs[1],φ)
    cutgeo,Ω1,Ω2,Ω3,Ω4,Γ12,Γ13,Γ14,Γ23,Γ24,Γ34 = compute_geo(_φh1, φh2)
    meas_K1 = get_cell_measure(Γ23, Ω_bg)
    κ1 = CellField( meas_K1, Ω_bg)
    _dΓ23 = Measure(Γ23,2*order)
    sum(∫( κ1 )_dΓ23)
  end
  dJ_fd = FiniteDiff.finite_difference_gradient(fdm_compute,get_free_dof_values(φh1))
  @test maximum(abs,dj-dJ_fd)/maximum(abs,dj) < 1e-6
end

run_test()
cell_measures_test()

end