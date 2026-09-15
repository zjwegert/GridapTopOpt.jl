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

  f(x) = sin(x[1])*sin(x[2])
  fh = interpolate(f,V0)

  println(" #### Case c: φ↦∫_{∂D^(i)(φ) ∩ D^(j)} f")
  function F6(φ1,V_φ1,φ2,V_φ2,φ3,V_φ3,Ωi_name,Ωj_name)
    φh1 = FEFunction(V_φ1, φ1)
    φh2 = FEFunction(V_φ2, φ2)
    φh3 = FEFunction(V_φ3, φ3)
    cutgeo = compute_geo(φh1,φh2,φh3)
    Γ = EmbeddedBoundary(cutgeo,Ωi_name,Ωj_name)
    dΓ = Measure(Γ,order*2)
    return sum(∫(fh)dΓ)
  end
  function F6_ad(φ1,V_φ1,φ2,V_φ2,φ3,V_φ3,Ωi_name,Ωj_name)
    φh1 = FEFunction(V_φ1, φ1)
    φh2 = FEFunction(V_φ2, φ2)
    φh3 = FEFunction(V_φ3, φ3)
    cutgeo = compute_geo(φh1,φh2,φh3)
    Γ = DifferentiableEmbeddedBoundary(cutgeo,Ωi_name,Ωj_name)
    dΓ = Measure(Γ,order*2)
    return _ -> ∫(fh)dΓ
  end
  function F6_analytic(k,φhs,j,l,Ωm_name,Ωn_name,cj=1,cl=1)
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
    ∇ˢφ_Λ = Operation(abs)(n_S_Λ ⋅ ∇(φhs[k]).plus)
    # Σᴰ
    Σᴰ = Boundary(Γ)
    dΣᴰ = Measure(Σᴰ,2*order)
    n_S_Σᴰ = get_normal_vector(Σᴰ)
    ∇ˢφ_Σᴰ = Operation(abs)(n_S_Σᴰ ⋅ ∇(φhs[k]))
    n_Γ_on_Σᴰ = get_subfacet_normal_vector(Σᴰ)
    n_Σᴰ = get_ghost_normal_vector(Σᴰ)
    # Σj
    Γφ1 = EmbeddedBoundary(cutgeo,"φ$k")
    Γφ2 = EmbeddedBoundary(cutgeo,"φ$j")
    # Σj = IntersectionTriangulation(Γφ1, Γφ2)
    Dφk = cl == 1 ? get_geometry(cutgeo.geo,"φ$l") : !get_geometry(cutgeo.geo,"φ$l")
    Σj = RestrictedIntersectionTriangulation(cutgeo,"φ$k","φ$j",Dφk)
    dΣj = Measure(Σj, order*2)
    n_Di_in_∂Dj = get_tangent_vector(Σj, 2)
    n_Σ_1j = get_normal_vector(Σj, 1)
    n_Σ_2j = get_normal_vector(Σj, 2)
    ∇ˢφ_Σj = Operation(abs)(n_Di_in_∂Dj ⋅ ∇(φhs[k]))
    # χ1(φ) = (cj * φ < 0) ? 1 : 0
    # Σl
    Γφ1 = EmbeddedBoundary(cutgeo,"φ$k")
    Γφ2 = EmbeddedBoundary(cutgeo,"φ$l")
    # Σl = IntersectionTriangulation(Γφ1, Γφ2)
    Dφj = cj == 1 ? get_geometry(cutgeo.geo,"φ$j") : !get_geometry(cutgeo.geo,"φ$j")
    Σl = RestrictedIntersectionTriangulation(cutgeo,"φ$k","φ$l",Dφj)
    dΣl = Measure(Σl, order*2)
    n_Di_in_∂Dl = get_tangent_vector(Σl, 2)
    n_Σ_1l = get_normal_vector(Σl, 1)
    n_Σ_2l = get_normal_vector(Σl, 2)
    ∇ˢφ_Σl = Operation(abs)(n_Di_in_∂Dl ⋅ ∇(φhs[k]))
    # χ2(φ) = (cl * φ < 0) ? 1 : 0
    return w -> ∫(-(∇(fh)⋅n_Γ)*w/abs(n_Γ⋅∇(φhs[k])))dΓ +
      ∫((jump(fh*(n_Γ_on_Λ⋅n_k)) * mean(w) / ∇ˢφ_Λ))dΛ +
      ∫((n_Σᴰ ⋅ n_Γ_on_Σᴰ)*fh*w/∇ˢφ_Σᴰ)dΣᴰ +
      ∫(cj*(n_Σ_1j ⋅ n_Σ_2j)*fh*w/∇ˢφ_Σj)dΣj +
      ∫(cl*(n_Σ_1l ⋅ n_Σ_2l)*fh*w/∇ˢφ_Σl)dΣl
      # ∫(cj*(χ2 ∘ (φhs[l]))*(n_Σ_1j ⋅ n_Σ_2j)*fh*w/∇ˢφ_Σj)dΣj +
      # ∫(cl*(χ1 ∘ (φhs[j]))*(n_Σ_1l ⋅ n_Σ_2l)*fh*w/∇ˢφ_Σl)dΣl
  end

  println(" ####  φ3 -> D^(1)(φ1) ∩ D^(2)(φ2) ∩ ∂D^(3)(φ3) = Γ24(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F6_analytic(3,[φh1,φh2,φh3],1,2,"Ω4","Ω2",1,1),V_φ3);
  dF_ad_vec = assemble_vector(gradient(F6_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω2","Ω4"),φh3),V_φ3);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F6(_φ1, V_φ1, _φ2, V_φ2, φ, V_φ3, "Ω2","Ω4"), _φ3);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ3 -> D^(1)(φ1) ∩ D^(2)(φ2)ᶜ ∩ ∂D^(3)(φ3) = Γ13(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F6_analytic(3,[φh1,φh2,φh3],1,2,"Ω3","Ω1",1,-1),V_φ3);
  dF_ad_vec = assemble_vector(gradient(F6_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω1","Ω3"),φh3),V_φ3);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F6(_φ1, V_φ1, _φ2, V_φ2, φ, V_φ3, "Ω1","Ω3"), _φ3);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println(" ####  φ3 -> D^(1)ᶜ(φ1) ∩ D^(2)(φ2) ∩ ∂D^(3)(φ3) = Γ68(φ1,φ2,φ3)")
  dF_ad_vec = assemble_vector(gradient(F6_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω6","Ω8"),φh3),V_φ3);
  dF_analytic_vec = assemble_vector(F6_analytic(3,[φh1,φh2,φh3],1,2,"Ω8","Ω6",-1,1),V_φ3);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F6(_φ1, V_φ1, _φ2, V_φ2, φ, V_φ3, "Ω6","Ω8"), _φ3);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println(" ####  φ3 -> D^(1)ᶜ(φ1) ∩ D^(2)ᶜ(φ2) ∩ ∂D^(3)(φ3) = Γ57(φ1,φ2,φ3)")
  dF_ad_vec = assemble_vector(gradient(F6_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω5","Ω7"),φh3),V_φ3);
  dF_analytic_vec = assemble_vector(F6_analytic(3,[φh1,φh2,φh3],1,2,"Ω7","Ω5",-1,-1),V_φ3);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F6(_φ1, V_φ1, _φ2, V_φ2, φ, V_φ3, "Ω5","Ω7"), _φ3);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ2 -> D^(1)(φ1) ∩ ∂D^(2)(φ2) ∩ D^(3)(φ3)  = Γ43(φ1,φ2,φ3) => ∂Ω4 ∩ ∂Ω3")
  dF_analytic_vec = assemble_vector(F6_analytic(2,[φh1,φh2,φh3],1,3,"Ω4","Ω3",1,1),V_φ2);
  dF_ad_vec = assemble_vector(gradient(F6_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω4","Ω3"),φh2),V_φ2);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F6(_φ1, V_φ1, φ, V_φ2, _φ3, V_φ3, "Ω4","Ω3"), _φ2);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ3 -> D^(1)(φ1) ∩ ∂D^(2)(φ2) ∩ D^(3)ᶜ(φ3) = Γ12(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F6_analytic(2,[φh1,φh2,φh3],1,3,"Ω2","Ω1",1,-1),V_φ2);
  dF_ad_vec = assemble_vector(gradient(F6_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω1","Ω2"),φh2),V_φ2);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F6(_φ1, V_φ1, φ, V_φ2, _φ3, V_φ3, "Ω1","Ω2"), _φ2);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ3 -> D^(1)ᶜ(φ1) ∩ ∂D^(2)(φ2) ∩ D^(3)(φ3) = Γ78(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F6_analytic(2,[φh1,φh2,φh3],1,3,"Ω8","Ω7",-1,1),V_φ2);
  dF_ad_vec = assemble_vector(gradient(F6_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω7","Ω8"),φh2),V_φ2);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F6(_φ1, V_φ1, φ, V_φ2, _φ3, V_φ3, "Ω7","Ω8"), _φ2);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-12
  println("#### φ3 -> D^(1)ᶜ(φ1) ∩ ∂D^(2)(φ2) ∩ D^(3)ᶜ(φ3) = Γ56(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F6_analytic(2,[φh1,φh2,φh3],1,3,"Ω6","Ω5",-1,-1),V_φ2);
  dF_ad_vec = assemble_vector(gradient(F6_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω5","Ω6"),φh2),V_φ2);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F6(_φ1, V_φ1, φ, V_φ2, _φ3, V_φ3, "Ω5","Ω6"), _φ2);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-12
  println("#### φ1 -> ∂D^(1)(φ1) ∩ D^(2)(φ2) ∩ D^(3)(φ3)  = Γ48(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F6_analytic(1,[φh1,φh2,φh3],2,3,"Ω4","Ω8",1,1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F6_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω4","Ω8"),φh1),V_φ1);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F6(φ, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω4","Ω8"), _φ1);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ1 -> ∂D^(1)(φ1) ∩ D^(2)ᶜ(φ2) ∩ D^(3)(φ3) = Γ37(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F6_analytic(1,[φh1,φh2,φh3],2,3,"Ω3","Ω7",-1,1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F6_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω3","Ω7"),φh1),V_φ1);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F6(φ, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω3","Ω7"), _φ1);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ1 -> ∂D^(1)(φ1) ∩ D^(2)(φ2) ∩ D^(3)ᶜ(φ3) = Γ26(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F6_analytic(1,[φh1,φh2,φh3],2,3,"Ω2","Ω6",1,-1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F6_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω2","Ω6"),φh1),V_φ1);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F6(φ, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω2","Ω6"), _φ1);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
  println("#### φ1 -> ∂D^(1)(φ1) ∩ D^(2)ᶜ(φ2) ∩ D^(3)ᶜ(φ3) = Γ15(φ1,φ2,φ3)")
  dF_analytic_vec = assemble_vector(F6_analytic(1,[φh1,φh2,φh3],2,3,"Ω1","Ω5",-1,-1),V_φ1);
  dF_ad_vec = assemble_vector(gradient(F5_ad(_φ1, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω1","Ω5"),φh1),V_φ1);
  dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F5(φ, V_φ1, _φ2, V_φ2, _φ3, V_φ3, "Ω1","Ω5"), _φ1);
  err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs,dF_fdm - dF_ad_vec)
  err_ad = maximum(abs, dF_analytic_vec) > 0 ? maximum(abs, dF_analytic_vec - dF_ad_vec) / maximum(abs, dF_analytic_vec) : maximum(abs, dF_analytic_vec - dF_ad_vec)
  @info "   AD vs FDM:" err_fdm
  @test err_fdm < 1e-6
  @info "   Analytic vs AD:" err_ad
  @test err_ad < 1e-13
end

end