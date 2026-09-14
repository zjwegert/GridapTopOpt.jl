# This module tests AD against analytic expressions
module DifferentiableCutPolyTriangulationsTests_Analytic1

include("DifferentiableCutPolyTriangulationsTests_Utils.jl")

################################################################
#             Test suite: Analytic methods above               #
################################################################

function run_test(d)
  if d == 2
     @info "Running 2D test"
    φ1 = (x,y) -> (x-0.5)^2+(y-0.425)^2-(0.35)^2
    φ2 = (x,y) -> (x-0.5)^2+(y-0.6)^2-(0.35)^2
    bgmodel = generate_model(2,10)
  elseif d == 3
    @info "Running 3D test"
    φ1 = (x,y,z) -> (x-0.5)^2+(y-0.425)^2+(z-0.5)^2-(0.35)^2
    φ2 = (x,y,z) -> (x-0.5)^2+(y-0.6)^2+(z-0.5)^2-(0.35)^2
    bgmodel = generate_model(3,5)
  else
    error("Unsupported dimension: $d")
  end

  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  V0 = TestFESpace(bgmodel,reffe)
  V_φ1 = TestFESpace(bgmodel,reffe)
  V_φ2 = TestFESpace(bgmodel,reffe)
  φh1 = interpolate(x->φ1(x...),V_φ1)
  φh2 = interpolate(x->φ2(x...),V_φ2)

  for φh in (φh1,φh2)
      x_φ = get_free_dof_values(φh)
      idx = findall(isapprox(0.0;atol=10^-10),x_φ)
      !isempty(idx) && @info "Correcting level values!"
      x_φ[idx] .+= 100*eps(eltype(x_φ))
  end

  function compute_geo(φh1,φh2)
      geo1 = DiscreteGeometryFromFEFunction(φh1,bgmodel,name="φ1")
      geo2 = DiscreteGeometryFromFEFunction(φh2,bgmodel,name="φ2")
      setdiff_geo1_geo2 = setdiff(geo1,geo2,name="Ω1")
      setdiff_geo2_geo1 = setdiff(geo2,geo1,name="Ω2")
      intersect_geo1_geo2 = intersect(geo1,geo2,name="Ω3")
      outside_geo1_geo2 = !(union(geo1,geo2),name="Ω4")
      _all = union(union(union(setdiff_geo1_geo2,setdiff_geo2_geo1),intersect_geo1_geo2),outside_geo1_geo2)
      return cut(PolytopalLevelSetCutter(),bgmodel,_all)
  end

  cutgeo = compute_geo(φh1,φh2)
  Γφ1 = EmbeddedBoundary(cutgeo,"φ1")
  Γφ2 = EmbeddedBoundary(cutgeo,"φ2")
  Γ = EmbeddedBoundary(cutgeo,"Ω1","Ω3")
  # writevtk(Triangulation(cutgeo,"Ω1"),"results/Shape_calc/Ω1_$(d)D")
  # writevtk(Triangulation(cutgeo,"Ω3"),"results/Shape_calc/Ω3_$(d)D")

  model_Γφ1 = get_active_model(EmbeddedBoundary(cutgeo,"φ1"))
  model_Γφ2 = get_active_model(EmbeddedBoundary(cutgeo,"φ2"))

  Gridap.Geometry.test_discrete_model(model_Γφ1)
  Gridap.Geometry.test_discrete_model(model_Γφ2)

  # writevtk(Γφ1,"results/Shape_calc/D_phi1_$(d)D")
  # writevtk(Γφ2,"results/Shape_calc/D_phi2_$(d)D")

  # Test IntersectionTriangulation
  Σ = IntersectionTriangulation(Γφ1, Γφ2)
  Gridap.Geometry.test_triangulation(Σ)
  @info "IntersectionTriangulation created" num_cells(Σ) num_point_dims(Σ) num_cell_dims(Σ)

  # Test normal vectors (one per boundary)
  n_Σ_1 = get_normal_vector(Σ, 1)
  n_Σ_2 = get_normal_vector(Σ, 2)
  @info "Normal vectors obtained" typeof(n_Σ_1) typeof(n_Σ_2)

  # Test tangent vectors (conormals)
  t_Σ_1 = get_tangent_vector(Σ, 1)
  t_Σ_2 = get_tangent_vector(Σ, 2)
  @info "Tangent vectors obtained" typeof(t_Σ_1) typeof(t_Σ_2)

  # Test assembly
  dΣ = Measure(Σ, 2)
  testf(q) = ∫(q)dΣ
  testf_vec = assemble_vector(testf,V_φ1)
  testfh = FEFunction(V_φ1, testf_vec)
  # if d == 3
  #   v1 = testfh(Point(1-0.164286,0.5125,0.5))
  #   v2 = testfh(Point(0.164286,0.5125,0.5))
  #   @info "Assembly sym rel error (3D):" abs(v1-v2)/abs(v1)
  # elseif d == 2
  #   v1 = testfh(Point(0.2,0.5))
  #   v2 = testfh(Point(0.8,0.5))
  #   @info "Assembly sym rel error (2D):" abs(v1-v2)/abs(v1)
  # end
end

run_test(2)
run_test(3)

println("
################################################################
#         Test suite: FD/AD vs analytic results (2LSF)         #
################################################################
")

for d in (2,3)
  if d == 2
    @info "----------Running 2D test"
    φ1 = (x,y) -> (x-0.5)^2+(y-0.425)^2-(0.35)^2
    φ2 = (x,y) -> (x-0.5)^2+(y-0.6)^2-(0.35)^2
    bgmodel = generate_model(2,10)
  else
    @info "----------Running 3D test"
    φ1 = (x,y,z) -> (x-0.5)^2+(y-0.425)^2+(z-0.5)^2-(0.35)^2
    φ2 = (x,y,z) -> (x-0.5)^2+(y-0.6)^2+(z-0.5)^2-(0.35)^2
    bgmodel = generate_model(3,5)
  end
  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  V0 = TestFESpace(bgmodel,reffe)
  V_φ1 = TestFESpace(bgmodel,reffe)
  V_φ2 = TestFESpace(bgmodel,reffe)
  V_φi = MultiFieldFESpace([V_φ1, V_φ2])
  φhi = interpolate([x->φ1(x...), x->φ2(x...)], V_φi)
  φh1 = φhi[1]
  φh2 = φhi[2]

  for φh in (φh1,φh2)
      x_φ = get_free_dof_values(φh)
      idx = findall(isapprox(0.0;atol=10^-10),x_φ)
      !isempty(idx) && @info "Correcting level values!"
      x_φ[idx] .+= 100*eps(eltype(x_φ))
  end

  function compute_geo(φh1,φh2)
      geo1 = DiscreteGeometryFromFEFunction(φh1,bgmodel,name="φ1")
      geo2 = DiscreteGeometryFromFEFunction(φh2,bgmodel,name="φ2")
      setdiff_geo1_geo2 = setdiff(geo1,geo2,name="Ω1")
      setdiff_geo2_geo1 = setdiff(geo2,geo1,name="Ω2")
      intersect_geo1_geo2 = intersect(geo1,geo2,name="Ω3")
      outside_geo1_geo2 = !(union(geo1,geo2),name="Ω4")
      _all = union(union(union(setdiff_geo1_geo2,setdiff_geo2_geo1),intersect_geo1_geo2),outside_geo1_geo2)
      return cut(PolytopalLevelSetCutter(),bgmodel,_all)
  end

  cutgeo = compute_geo(φh1,φh2)

  f(x) = sin(x[1])*sin(x[2])
  fh = interpolate(f,V0)
  # -- Shape derivative (Case a: (φ1,φ2)↦∫_{D^(i) ∩ D^(j)} f)
  function F1(φ1,V_φ1,φ2,V_φ2,Ω_name)
    φh1 = FEFunction(V_φ1, φ1)
    φh2 = FEFunction(V_φ2, φ2)
    cutgeo = compute_geo(φh1,φh2)
    Di_cap_Dj = Triangulation(cutgeo,Ω_name)
    dΩ = Measure(Di_cap_Dj,order*2)
    return sum(∫(fh)dΩ)
  end
  function F1_ad(φ1,V_φ1,φ2,V_φ2,Ω_name)
    φh1 = FEFunction(V_φ1, φ1)
    φh2 = FEFunction(V_φ2, φ2)
    cutgeo = compute_geo(φh1,φh2)
    Di_cap_Dj = DifferentiableTriangulation(cutgeo,Ω_name)
    dΩ = Measure(Di_cap_Dj,order*2)
    return φ->∫(fh)dΩ
  end
  function F1_analytic(φhi,Ωi_name,Ωj_name,coeff=-1)
    ∂Di_cap_Dj = EmbeddedBoundary(cutgeo,Ωi_name,Ωj_name)
    n = get_normal_vector(∂Di_cap_Dj)
    # n = ∇(φhi)/(norm ∘ (∇(φhi))) # using due to bug in get_normal_vector for non-empty ∂Di ∩ ∂Dj ∩ K
    dΓ = Measure(∂Di_cap_Dj,order*2)
    return w->∫(coeff*fh*w/abs(n⋅∇(φhi)))dΓ
  end

  _φ1 = get_free_dof_values(φh1)
  _φ2 = get_free_dof_values(φh2)

  cases = Dict(
    # φ1 -> Ω1(φ1,φ2) = D^(1)(φ1) ∩ D^(2)ᶜ(φ2) => Γ = ∂D^(1) ∩ D^(2)ᶜ = ∂Ω1 ∩ ∂Ω4
    # φ2 -> Ω1(φ1,φ2) = D^(1)(φ1) ∩ D^(2)ᶜ(φ2) => Γ = ∂D^(2) ∩ D^(1) = ∂Ω1 ∩ ∂Ω3
    "Ω1" => (("Ω1", "Ω4", -1),("Ω1", "Ω3", 1)),
    # φ1 -> Ω2(φ1,φ2) = D^(1)(φ1)ᶜ ∩ D^(2)(φ2) => Γ = ∂D^(1) ∩ D^(2) = ∂Ω2 ∩ ∂Ω3
    # φ2 -> Ω2(φ1,φ2) = D^(1)(φ1)ᶜ ∩ D^(2)(φ2) => Γ = ∂D^(2) ∩ D^(1)ᶜ = ∂Ω2 ∩ ∂Ω4
    "Ω2" => (("Ω2", "Ω3", 1),("Ω2", "Ω4", -1)),
    # φ1 -> Ω3(φ1,φ2) = D^(1)(φ1) ∩ D^(2)(φ2) => Γ = ∂D^(1) ∩ D^(2) = ∂Ω2 ∩ ∂Ω3
    # φ2 -> Ω3(φ1,φ2) = D^(1)(φ1) ∩ D^(2)(φ2) => Γ = ∂D^(2) ∩ D^(1) = ∂Ω1 ∩ ∂Ω3
    "Ω3" => (("Ω2", "Ω3", -1),("Ω1", "Ω3", -1)),
    # φ1 -> Ω4(φ1,φ2) = D^(1)(φ1)ᶜ ∩ D^(2)(φ2)ᶜ => Γ = ∂D^(1) ∩ D^(2)ᶜ = ∂Ω1 ∩ ∂Ω4
    # φ2 -> Ω4(φ1,φ2) = D^(1)(φ1)ᶜ ∩ D^(2)(φ2)ᶜ => Γ = ∂D^(2) ∩ D^(1)ᶜ = ∂Ω2 ∩ ∂Ω4
    "Ω4" => (("Ω1", "Ω4", 1),("Ω2", "Ω4", 1))
  )

  for case in keys(cases)
    _f_ad = F1_ad(_φ1, V_φ1, _φ2, V_φ2, case)
    @info "Testing case φ1 -> $case(φ1,φ2)"
    dF_analytic_vec1 = assemble_vector(F1_analytic(φh1, cases[case][1]...),V_φ1)
    dF_ad_vec1 = assemble_vector(gradient(_f_ad,φh1),V_φ1)
    dF_fdm1 = FiniteDiff.finite_difference_gradient(φ -> F1(φ, V_φ1, _φ2, V_φ2, case), _φ1)
    err_fdm = maximum(abs,dF_fdm1 - dF_analytic_vec1) / maximum(abs, dF_analytic_vec1)
    err_ad = maximum(abs, dF_analytic_vec1 - dF_ad_vec1) / maximum(abs, dF_ad_vec1)
    @info "   AD vs FDM:" err_fdm
    @test err_fdm < 1e-7
    @info "   Analytic vs AD:" err_ad
    @test err_ad < 1e-14
    @info "Testing case φ2 -> $case(φ1,φ2)"
    dF_analytic_vec2 = assemble_vector(F1_analytic(φh2, cases[case][2]...),V_φ2)
    dF_ad_vec2 = assemble_vector(gradient(_f_ad,φh2),V_φ2)
    dF_fdm2 = FiniteDiff.finite_difference_gradient(φ -> F1(_φ1, V_φ1, φ, V_φ2, case), _φ2)
    err_fdm2 = maximum(abs,dF_fdm2 - dF_analytic_vec2) / maximum(abs, dF_analytic_vec2)
    err_ad2 = maximum(abs, dF_analytic_vec2 - dF_ad_vec2) / maximum(abs, dF_ad_vec2)
    @info "   AD vs FDM:" err_fdm2
    @test err_fdm2 < 1e-7
    @info "   Analytic vs AD:" err_ad2
    @test err_ad2 < 1e-14
    @info "Testing case [φ1, φ2] -> $case(φ1,φ2)"
    dF_ad_vec_mf_mono = assemble_vector(gradient(_f_ad,φhi;ad_type=:monolithic),V_φi)
    dF_ad_vec_mf_split = assemble_vector(gradient(_f_ad,φhi;ad_type=:split),V_φi)
    err_ad_mono = maximum(abs, [dF_analytic_vec1; dF_analytic_vec2] - dF_ad_vec_mf_mono) / maximum(abs, dF_ad_vec_mf_mono)
    err_ad_split = maximum(abs, [dF_analytic_vec1; dF_analytic_vec2] - dF_ad_vec_mf_split) / maximum(abs, dF_ad_vec_mf_split)
    @info "   Analytic vs AD (mono):" err_ad_mono
    @test err_ad_mono < 1e-14
    @info "   Analytic vs AD (split):" err_ad_split
    @test err_ad_split < 1e-14
  end

  f(x) = sin(x[1])*sin(x[2])
  fh = interpolate(f,V0)
  println(" #### Case b: φ↦∫_{∂D^(i) ∩ D^(j)(φ)} f")
  function F2(φ1,V_φ1,φ2,V_φ2,Ωi_name,Ωj_name)
    φh1 = FEFunction(V_φ1, φ1)
    φh2 = FEFunction(V_φ2, φ2)
    cutgeo = compute_geo(φh1,φh2)
    Γ = EmbeddedBoundary(cutgeo,Ωi_name,Ωj_name)
    dΓ = Measure(Γ,order*2)
    return sum(∫(fh)dΓ)
  end
  function F2_ad(φ1,V_φ1,φ2,V_φ2,Ωi_name,Ωj_name)
    φh1 = FEFunction(V_φ1, φ1)
    φh2 = FEFunction(V_φ2, φ2)
    cutgeo = compute_geo(φh1,φh2)
    Γ = DifferentiableEmbeddedBoundary(cutgeo,Ωi_name,Ωj_name)
    dΓ = Measure(Γ,order*2)
    return φ -> ∫(fh)dΓ
  end
  function F2_analytic(φhi,j,coeff=-1)
    Γφ1 = EmbeddedBoundary(cutgeo,"φ1")
    Γφ2 = EmbeddedBoundary(cutgeo,"φ2")
    Σ = IntersectionTriangulation(Γφ1, Γφ2)
    dΣ = Measure(Σ, order*2)
    n_Di_in_∂Dj = get_tangent_vector(Σ, j)
    ∇ˢφ_Σ = Operation(abs)(n_Di_in_∂Dj ⋅ ∇(φhi))
    return w -> ∫(coeff*(fh*w)/∇ˢφ_Σ)dΣ
  end

  println(" ####  φ2 -> ∂D^(1)(φ1) ∩ D^(2)(φ2)  = Γ23(φ1,φ2) => ∂Ω2 ∩ ∂Ω3")
  dF_analytic_vec = assemble_vector(F2_analytic(φh2,1,-1),V_φ2);
  dF_ad_vec = assemble_vector(gradient(F2_ad(_φ1, V_φ1, _φ2, V_φ2, "Ω2","Ω3"),φh2),V_φ2);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F2(_φ1, V_φ1, φ, V_φ2, "Ω2","Ω3"), _φ2);
  err_fdm = maximum(abs,dF_fdm - dF_analytic_vec) / maximum(abs, dF_analytic_vec)
  err_ad = maximum(abs,dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_ad_vec)
  @info "   Analytic vs FD:" err_fdm
  @info "   Analytic vs AD:" err_ad
  @test err_fdm < 1e-7
  @test err_ad < 1e-14
  println(" ####  φ2 -> ∂D^(1)(φ1) ∩ D^(2)(φ2)ᶜ = Γ14(φ1,φ2) => ∂Ω1 ∩ ∂Ω4")
  dF_analytic_vec = assemble_vector(F2_analytic(φh2,1,1),V_φ2);
  dF_ad_vec = assemble_vector(gradient(F2_ad(_φ1, V_φ1, _φ2, V_φ2, "Ω1","Ω4"),φh2),V_φ2);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F2(_φ1, V_φ1, φ, V_φ2, "Ω1","Ω4"), _φ2);
  err_fdm = maximum(abs,dF_fdm - dF_analytic_vec) / maximum(abs, dF_analytic_vec)
  err_ad = maximum(abs,dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_ad_vec)
  @info "   Analytic vs FD:" err_fdm
  @info "   Analytic vs AD:" err_ad
  @test err_fdm < 1e-7
  @test err_ad < 1e-14
  println(" ####  φ1 -> ∂D^(2)(φ2) ∩ D^(1)(φ1)  = Γ13(φ1,φ2) => ∂Ω1 ∩ ∂Ω3")
  dF_analytic_vec = assemble_vector(F2_analytic(φh1,2,-1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F2_ad(_φ1, V_φ1, _φ2, V_φ2, "Ω1","Ω3"),φh1),V_φ1);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F2(φ, V_φ1, _φ2, V_φ2, "Ω1","Ω3"), _φ1);
  err_fdm = maximum(abs,dF_fdm - dF_analytic_vec) / maximum(abs, dF_analytic_vec)
  err_ad = maximum(abs,dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_ad_vec)
  @info "   Analytic vs FD:" err_fdm
  @info "   Analytic vs AD:" err_ad
  @test err_fdm < 1e-7
  @test err_ad < 1e-14
  println(" ####  φ1 -> ∂D^(2)(φ2) ∩ D^(1)(φ1)ᶜ = Γ24(φ1,φ2) => ∂Ω2 ∩ ∂Ω4")
  dF_analytic_vec = assemble_vector(F2_analytic(φh1,2,1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F2_ad(_φ1, V_φ1, _φ2, V_φ2, "Ω2","Ω4"),φh1),V_φ1);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F2(φ, V_φ1, _φ2, V_φ2, "Ω2","Ω4"), _φ1);
  err_fdm = maximum(abs,dF_fdm - dF_analytic_vec) / maximum(abs, dF_analytic_vec)
  err_ad = maximum(abs,dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_ad_vec)
  @info "   Analytic vs FD:" err_fdm
  @info "   Analytic vs AD:" err_ad
  @test err_fdm < 1e-7
  @test err_ad < 1e-14

  println(" #### Case c: φ↦∫_{∂D^(i)(φ) ∩ D^(j)} f")
  function F3(φ1,V_φ1,φ2,V_φ2,Ωi_name,Ωj_name)
    φh1 = FEFunction(V_φ1, φ1)
    φh2 = FEFunction(V_φ2, φ2)
    cutgeo = compute_geo(φh1,φh2)
    Γ = EmbeddedBoundary(cutgeo,Ωi_name,Ωj_name)
    dΓ = Measure(Γ,order*2)
    return sum(∫(fh)dΓ)
  end
  function F3_ad(φ1,V_φ1,φ2,V_φ2,Ωi_name,Ωj_name)
    φh1 = FEFunction(V_φ1, φ1)
    φh2 = FEFunction(V_φ2, φ2)
    cutgeo = compute_geo(φh1,φh2)
    Γ = DifferentiableEmbeddedBoundary(cutgeo,Ωi_name,Ωj_name)
    dΓ = Measure(Γ,order*2)
    return φ -> ∫(fh)dΓ
  end
  function F3_analytic(φhi,j,Ωm_name,Ωn_name,c=1)
    # Γ
    Γ = EmbeddedBoundary(cutgeo,Ωm_name,Ωn_name)
    dΓ = Measure(Γ,order*2)
    n_Γ = get_normal_vector(Γ)
    # Λ
    Λ = Skeleton(Γ)
    dΛ = Measure(Λ,order*2)
    n_S_Λ = get_normal_vector(Λ)
    n_Γ_on_Λ = get_subfacet_normal_vector(Λ)
    n_k = get_ghost_normal_vector(Λ)
    ∇ˢφ_Λ = Operation(abs)(n_S_Λ ⋅ ∇(φhi).plus)
    # Σᴰ
    Σᴰ = Boundary(Γ)
    dΣᴰ = Measure(Σᴰ,2*order)
    n_S_Σᴰ = get_normal_vector(Σᴰ)
    ∇ˢφ_Σᴰ = Operation(abs)(n_S_Σᴰ ⋅ ∇(φhi))
    n_Γ_on_Σᴰ = get_subfacet_normal_vector(Σᴰ)
    n_Σᴰ = get_ghost_normal_vector(Σᴰ)
    # Σ
    Γφ1 = EmbeddedBoundary(cutgeo,"φ1")
    Γφ2 = EmbeddedBoundary(cutgeo,"φ2")
    Σ = IntersectionTriangulation(Γφ1, Γφ2)
    dΣ = Measure(Σ, order*2)
    n_Di_in_∂Dj = get_tangent_vector(Σ, j)
    n_Σ_1 = get_normal_vector(Σ, 1)
    n_Σ_2 = get_normal_vector(Σ, 2)
    ∇ˢφ_Σ = Operation(abs)(n_Di_in_∂Dj ⋅ ∇(φhi))
    return w -> ∫(-(∇(fh)⋅n_Γ)*w/abs(n_Γ⋅∇(φhi)))dΓ +
      ∫((jump(fh*(n_Γ_on_Λ⋅n_k)) * mean(w) / ∇ˢφ_Λ))dΛ +
      ∫((n_Σᴰ ⋅ n_Γ_on_Σᴰ)*fh*w/∇ˢφ_Σᴰ)dΣᴰ +
      ∫(c*(n_Σ_1 ⋅ n_Σ_2)*fh*w/∇ˢφ_Σ)dΣ
  end

  println(" ####  φ2 -> D^(1)(φ1) ∩ ∂D^(2)(φ2)  = Γ13(φ1,φ2), Normal to D^(2) follows ∇φ2, so c = 1")
  dF_analytic_vec = assemble_vector(F3_analytic(φh2,1,"Ω3","Ω1",1),V_φ2);
  dF_ad_vec = assemble_vector(gradient(F3_ad(_φ1,V_φ1,_φ2,V_φ2,"Ω1","Ω3"),φh2),V_φ2);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F3(_φ1,V_φ1,φ,V_φ2,"Ω1","Ω3"), _φ2);
  err_fdm = maximum(abs,dF_fdm - dF_analytic_vec) / maximum(abs, dF_analytic_vec)
  err_ad = maximum(abs,dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_ad_vec)
  @info "   Analytic vs FD:" err_fdm
  @info "   Analytic vs AD:" err_ad
  @test err_fdm < 1e-6
  @test err_ad < 1e-14
  println(" ####  φ2 -> D^(1)(φ1)ᶜ ∩ ∂D^(2)(φ2) = Γ24(φ1,φ2), Normal to D^(2)ᶜ points in opposite direciton to ∇φ2, so c = -1")
  dF_analytic_vec = assemble_vector(F3_analytic(φh2,1,"Ω2","Ω4",-1),V_φ2);
  dF_ad_vec = assemble_vector(gradient(F3_ad(_φ1,V_φ1,_φ2,V_φ2,"Ω2","Ω4"),φh2),V_φ2);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F3(_φ1,V_φ1,φ,V_φ2,"Ω2","Ω4"), _φ2);
  err_fdm = maximum(abs,dF_fdm - dF_analytic_vec) / maximum(abs, dF_analytic_vec)
  err_ad = maximum(abs,dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_ad_vec)
  @info "   Analytic vs FD:" err_fdm
  @info "   Analytic vs AD:" err_ad
  @test err_fdm < 1e-6
  @test err_ad < 1e-13
  println(" ####  φ1 -> ∂D^(1)(φ1) ∩ D^(2)(φ2)  = Γ23(φ1,φ2), Normal to D^(1) follows ∇φ1, so c = 1")
  dF_analytic_vec = assemble_vector(F3_analytic(φh1,2,"Ω3","Ω2",1),V_φ2);
  dF_ad_vec = assemble_vector(gradient(F3_ad(_φ1,V_φ1,_φ2,V_φ2,"Ω2","Ω3"),φh1),V_φ1);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F3(φ,V_φ1,_φ2,V_φ2,"Ω2","Ω3"), _φ1);
  err_fdm = maximum(abs,dF_fdm - dF_analytic_vec) / maximum(abs, dF_analytic_vec)
  err_ad = maximum(abs,dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_ad_vec)
  @info "   Analytic vs FD:" err_fdm
  @info "   Analytic vs AD:" err_ad
  @test err_fdm < 1e-7
  @test err_ad < 1e-13
  println(" ####  φ1 -> ∂D^(1)(φ1) ∩ D^(2)(φ2)ᶜ = Γ14(φ1,φ2), Normal to D^(1)ᶜ points in opposite direciton to ∇φ1, so c = -1")
  dF_analytic_vec = assemble_vector(F3_analytic(φh1,2,"Ω1","Ω4",-1),V_φ2);
  dF_ad_vec = assemble_vector(gradient(F3_ad(_φ1,V_φ1,_φ2,V_φ2,"Ω1","Ω4"),φh1),V_φ1);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F3(φ,V_φ1,_φ2,V_φ2,"Ω1","Ω4"), _φ1);
  err_fdm = maximum(abs,dF_fdm - dF_analytic_vec) / maximum(abs, dF_analytic_vec)
  err_ad = maximum(abs,dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_ad_vec)
  @info "   Analytic vs FD:" err_fdm
  @info "   Analytic vs AD:" err_ad
  @test err_fdm < 1e-7
  @test err_ad < 1e-13
end

end