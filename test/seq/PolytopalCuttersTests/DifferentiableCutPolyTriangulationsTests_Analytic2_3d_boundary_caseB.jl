# This module tests AD against analytic expressions
module DifferentiableCutPolyTriangulationsTests_Analytic2_3d

include("DifferentiableCutPolyTriangulationsTests_Utils.jl")

println("
################################################################
#         Test suite: FD/AD vs analytic results (3LSF)         #
################################################################
")

for d in (3,)
  if d == 2
    @info "----------Running 2D test"
    φ1 = (x,y) -> (x-0.5)^2+(y-0.3)^2-(0.41)^2
    φ2 = (x,y) -> (x-0.5)^2+(y-0.6)^2-(0.35)^2
    φ3 = (x,y) -> -cos(4π*(x-0.5))*cos(4π*y) + 0.25
    bgmodel = generate_model(2,10)
  else
    @info "----------Running 3D test"
    φ1 = (x,y,z) -> (z-0.5)^2+(x-0.5)^2+(y-0.3)^2-(0.41)^2
    φ2 = (x,y,z) -> 0.3(z-0.5)^2+(x-0.5)^2+(y-0.7)^2-(0.29)^2
    φ3 = (x,y,z) -> -cos(2π*(x-0.5))*cos(2π*y)*cos(2π*z) + 0.24
    bgmodel = generate_model(3,5)
  end

  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  V0 = TestFESpace(bgmodel,reffe)
  V_φ1 = TestFESpace(bgmodel,reffe)
  V_φ2 = TestFESpace(bgmodel,reffe)
  V_φ3 = TestFESpace(bgmodel,reffe)
  V_φi = MultiFieldFESpace([V_φ1, V_φ2, V_φ3])
  φhi = interpolate([x->φ1(x...), x->φ2(x...), x->φ3(x...)], V_φi)
  φh1 = φhi[1]
  φh2 = φhi[2]
  φh3 = φhi[3]

  for φh in (φh1,φh2,φh3)
      x_φ = get_free_dof_values(φh)
      idx = findall(isapprox(0.0;atol=10^-10),x_φ)
      !isempty(idx) && @info "Correcting level values!"
      x_φ[idx] .+= 100*eps(eltype(x_φ))
  end

  function compute_geo(φh1,φh2,φh3)
      geo1 = DiscreteGeometryFromFEFunction(φh1,bgmodel,name="φ1")
      geo2 = DiscreteGeometryFromFEFunction(φh2,bgmodel,name="φ2")
      geo3 = DiscreteGeometryFromFEFunction(φh3,bgmodel,name="φ3")
      Ω1 = intersect(geo1 ∩ !geo2,!geo3,name="Ω1")
      Ω2 = intersect(geo1 ∩ geo2,!geo3,name="Ω2")
      Ω3 = intersect(geo1 ∩ !geo2,geo3,name="Ω3")
      Ω4 = intersect(geo1 ∩ geo2,geo3,name="Ω4")
      Ω5 = intersect(!geo1 ∩ !geo2,!geo3,name="Ω5")
      Ω6 = intersect(!geo1 ∩ geo2,!geo3,name="Ω6")
      Ω7 = intersect(!geo1 ∩ !geo2,geo3,name="Ω7")
      Ω8 = intersect(!geo1 ∩ geo2,geo3,name="Ω8")
      _all = Ω1 ∪ Ω2 ∪ Ω3 ∪ Ω4 ∪ Ω5 ∪ Ω6 ∪ Ω7 ∪ Ω8
      return cut(bgmodel,_all)
  end

  cutgeo = compute_geo(φh1,φh2,φh3)
  _φ1 = get_free_dof_values(φh1)
  _φ2 = get_free_dof_values(φh2)
  _φ3 = get_free_dof_values(φh3)

  f(x) = sin(x[1])*sin(x[2])
  fh = interpolate(f,V0)
  println(" #### Case b: φ↦∫_{∂D^(i) ∩ D^(j)(φ) ∩ D^(k)(φ)} f")
  function F5(φ1,V_φ1,φ2,V_φ2,φ3,V_φ3,Ωi_name,Ωj_name)
    φh1 = FEFunction(V_φ1, φ1)
    φh2 = FEFunction(V_φ2, φ2)
    φh3 = FEFunction(V_φ3, φ3)
    cutgeo = compute_geo(φh1,φh2,φh3)
    Γ = EmbeddedBoundary(cutgeo,Ωi_name,Ωj_name)
    dΓ = Measure(Γ,order*3)
    return sum(∫(fh)dΓ)
  end
  function F5_ad(φ1,V_φ1,φ2,V_φ2,φ3,V_φ3,Ωi_name,Ωj_name)
    φh1 = FEFunction(V_φ1, φ1)
    φh2 = FEFunction(V_φ2, φ2)
    φh3 = FEFunction(V_φ3, φ3)
    cutgeo = compute_geo(φh1,φh2,φh3)
    Γ = DifferentiableEmbeddedBoundary(cutgeo,Ωi_name,Ωj_name)
    dΓ = Measure(Γ,order*2)
    return _ -> ∫(fh)dΓ
  end
  function F5_analytic(φhi,i,j,c,Dφk,coeff=-1)
    Σ = RestrictedIntersectionTriangulation(cutgeo,"φ$i","φ$j",Dφk)
    dΣ = Measure(Σ, order*3)
    n_Di_in_∂Dj = get_tangent_vector(Σ, c)
    ∇ˢφ_Σ = Operation(abs)(n_Di_in_∂Dj ⋅ ∇(φhi))
    return w -> ∫(coeff*(fh*w)/∇ˢφ_Σ)dΣ
  end

  Dφ1 = get_geometry(cutgeo.geo,"φ1")
  Dφ2 = get_geometry(cutgeo.geo,"φ2")
  Dφ3 = get_geometry(cutgeo.geo,"φ3")

  # writevtk(Triangulation(cutgeo,Dφ1), "Dφ1")
  # writevtk(Triangulation(cutgeo,Dφ2), "Dφ2")
  # writevtk(Triangulation(cutgeo,Dφ3), "Dφ3")

  println("#### φ2 -> ∂D^(1)(φ1) ∩ D^(2)(φ2) ∩ D^(3)(φ3)  = Γ48(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh2,1,2,1,Dφ3,-1),V_φ2);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω4","Ω8"),φh2),V_φ2);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(_φ1, V_φ1, φ, V_φ2, _φ3, V_φ3, "Ω4","Ω8"), _φ2);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ3 -> ∂D^(1)(φ1) ∩ D^(2)(φ2) ∩ D^(3)(φ3)  = Γ48(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh3,1,3,1,Dφ2,-1),V_φ3);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω4","Ω8"),φh3),V_φ3);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(_φ1, V_φ1, _φ2, V_φ2, φ, V_φ3, "Ω4","Ω8"), _φ3);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ2 -> D^(1)(φ1) ∩ D^(2)(φ2)ᶜ ∩ ∂D^(3)(φ3) = Γ13(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh2,3,2,1,Dφ1,1),V_φ2);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω1","Ω3"),φh2),V_φ2);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(_φ1, V_φ1, φ, V_φ2, _φ3, V_φ3, "Ω1","Ω3"), _φ2);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ1 -> D^(1)(φ1) ∩ D^(2)(φ2)ᶜ ∩ ∂D^(3)(φ3) = Γ13(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh1,3,1,1,!Dφ2,-1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω1","Ω3"),φh1),V_φ1);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(φ, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω1","Ω3"), _φ1);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-5
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ1 -> D^(1)(φ1) ∩ ∂D^(2)(φ2) ∩ D^(3)ᶜ(φ3) = Γ12(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh1,2,1,1,!Dφ3,-1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω1","Ω2"),φh1),V_φ1);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(φ, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω1","Ω2"), _φ1);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ3 -> D^(1)(φ1) ∩ ∂D^(2)(φ2) ∩ D^(3)ᶜ(φ3) = Γ12(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh3,2,3,1,Dφ1,1),V_φ3);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω1","Ω2"),φh3),V_φ3);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(_φ1, V_φ1, _φ2, V_φ2, φ, V_φ3, "Ω1","Ω2"), _φ3);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ2 -> ∂D^(1)(φ1) ∩ D^(2)ᶜ(φ2) ∩ D^(3)ᶜ(φ3) = Γ15(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh2,1,2,1,!Dφ3,1),V_φ2);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω1","Ω5"),φh2),V_φ2);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(_φ1, V_φ1, φ, V_φ2, _φ3, V_φ3, "Ω1","Ω5"), _φ2);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ3 -> ∂D^(1)(φ1) ∩ D^(2)ᶜ(φ2) ∩ D^(3)ᶜ(φ3) = Γ15(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh3,1,3,1,!Dφ2,1),V_φ3);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω1","Ω5"),φh3),V_φ3);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(_φ1, V_φ1, _φ2, V_φ2, φ, V_φ3, "Ω1","Ω5"), _φ3);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ2 -> D^(1)(φ1) ∩ D^(2)(φ2) ∩ ∂D^(3)(φ3) = Γ24(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh2,3,2,1,Dφ1,-1),V_φ2);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω2","Ω4"),φh2),V_φ2);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(_φ1, V_φ1, φ, V_φ2, _φ3, V_φ3, "Ω2","Ω4"), _φ2);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ1 -> D^(1)(φ1) ∩ D^(2)(φ2) ∩ ∂D^(3)(φ3) = Γ24(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh1,3,1,1,Dφ2,-1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω2","Ω4"),φh1),V_φ1);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(φ, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω2","Ω4"), _φ1);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ1 -> D^(1)ᶜ(φ1) ∩ D^(2)(φ2) ∩ ∂D^(3)(φ3) = Γ68(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh1,3,1,1,Dφ2,1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω6","Ω8"),φh1),V_φ1);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(φ, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω6","Ω8"), _φ1);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ2 -> D^(1)ᶜ(φ1) ∩ D^(2)(φ2) ∩ ∂D^(3)(φ3) = Γ68(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh2,3,2,1,!Dφ1,-1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω6","Ω8"),φh2),V_φ2);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(_φ1, V_φ1, φ, V_φ2, _φ3, V_φ3, "Ω6","Ω8"), _φ2);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ1 -> D^(1)ᶜ(φ1) ∩ D^(2)ᶜ(φ2) ∩ ∂D^(3)(φ3) = Γ57(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh1,3,1,1,!Dφ2,1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω5","Ω7"),φh1),V_φ1);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(φ, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω5","Ω7"), _φ1);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ2 -> D^(1)ᶜ(φ1) ∩ D^(2)ᶜ(φ2) ∩ ∂D^(3)(φ3) = Γ57(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh2,3,2,1,!Dφ1,1),V_φ2);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω5","Ω7"),φh2),V_φ2);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(_φ1, V_φ1, φ, V_φ2, _φ3, V_φ3, "Ω5","Ω7"), _φ2);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ1 -> D^(1)(φ1) ∩ ∂D^(2)(φ2) ∩ D^(3)(φ3)  = Γ43(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh1,2,1,1,Dφ3,-1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω4","Ω3"),φh1),V_φ1);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(φ, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω4","Ω3"), _φ1);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ3 -> D^(1)(φ1) ∩ ∂D^(2)(φ2) ∩ D^(3)(φ3)  = Γ43(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh3,2,3,1,Dφ1,-1),V_φ3);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω4","Ω3"),φh3),V_φ3);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(_φ1, V_φ1, _φ2, V_φ2, φ, V_φ3, "Ω4","Ω3"), _φ3);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ1 -> D^(1)ᶜ(φ1) ∩ ∂D^(2)(φ2) ∩ D^(3)(φ3) = Γ78(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh1,2,1,1,Dφ3,1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω7","Ω8"),φh1),V_φ1);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(φ, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω7","Ω8"), _φ1);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ3 -> D^(1)ᶜ(φ1) ∩ ∂D^(2)(φ2) ∩ D^(3)(φ3) = Γ78(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh3,2,3,1,!Dφ1,-1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω7","Ω8"),φh3),V_φ3);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(_φ1, V_φ1, _φ2, V_φ2, φ, V_φ3, "Ω7","Ω8"), _φ3);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ1 -> D^(1)ᶜ(φ1) ∩ ∂D^(2)(φ2) ∩ D^(3)ᶜ(φ3) = Γ56(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh1,2,1,1,!Dφ3,1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω5","Ω6"),φh1),V_φ1);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(φ, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω5","Ω6"), _φ1);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ3 -> D^(1)ᶜ(φ1) ∩ ∂D^(2)(φ2) ∩ D^(3)ᶜ(φ3) = Γ56(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh3,2,3,1,!Dφ1,1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω5","Ω6"),φh3),V_φ3);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(_φ1, V_φ1, _φ2, V_φ2, φ, V_φ3, "Ω5","Ω6"), _φ3);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ2 -> ∂D^(1)(φ1) ∩ D^(2)ᶜ(φ2) ∩ D^(3)(φ3) = Γ37(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh2,1,2,1,Dφ3,1),V_φ2);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω3","Ω7"),φh2),V_φ2);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(_φ1, V_φ1, φ, V_φ2, _φ3, V_φ3, "Ω3","Ω7"), _φ2);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ3 -> ∂D^(1)(φ1) ∩ D^(2)ᶜ(φ2) ∩ D^(3)(φ3) = Γ37(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh3,1,3,1,!Dφ2,-1),V_φ3);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω3","Ω7"),φh3),V_φ3);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(_φ1, V_φ1, _φ2, V_φ2, φ, V_φ3, "Ω3","Ω7"), _φ3);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ2 -> ∂D^(1)(φ1) ∩ D^(2)(φ2) ∩ D^(3)ᶜ(φ3) = Γ26(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh2,1,2,1,!Dφ3,-1),V_φ2);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω2","Ω6"),φh2),V_φ2);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(_φ1, V_φ1, φ, V_φ2, _φ3, V_φ3, "Ω2","Ω6"), _φ2);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ3 -> ∂D^(1)(φ1) ∩ D^(2)(φ2) ∩ D^(3)ᶜ(φ3) = Γ26(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F5_analytic(φh3,1,3,1,Dφ2,1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω2","Ω6"),φh3),V_φ3);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(_φ1, V_φ1, _φ2, V_φ2, φ, V_φ3, "Ω2","Ω6"), _φ3);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
end

end