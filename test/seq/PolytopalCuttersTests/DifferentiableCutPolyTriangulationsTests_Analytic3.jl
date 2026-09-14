# This module tests AD against analytic expressions
module DifferentiableCutPolyTriangulationsTests_Analytic3

include("DifferentiableCutPolyTriangulationsTests_Utils.jl")

println("
################################################################
#        Test suite: AD vs analytic results (3LSF, rand)       #
################################################################
")

using Random

for d in (2,3)
  if d == 2
    @info "----------Running 2D test"
    bgmodel = generate_model(2,10)
  else
    @info "----------Running 3D test"
    bgmodel = generate_model(3,5)
  end

  seed1 = Random.MersenneTwister(1001)
  seed2 = Random.MersenneTwister(1002)
  seed3 = Random.MersenneTwister(1003)

  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  V0 = TestFESpace(bgmodel,reffe)
  V_φ1 = TestFESpace(bgmodel,reffe)
  V_φ2 = TestFESpace(bgmodel,reffe)
  V_φ3 = TestFESpace(bgmodel,reffe)
  V_φi = MultiFieldFESpace([V_φ1, V_φ2, V_φ3])
  φhi = interpolate([x->1-2rand(seed1), x->1-2rand(seed2), x->1-2rand(seed3)], V_φi)
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

  # for i = 1:8
  #   writevtk(Triangulation(cutgeo,"Ω$i"),"Ω$i")
  # end

  f(x) = sin(x[1])*sin(x[2])
  fh = interpolate(f,V0)
  println(" #### Case a: φ↦∫_{D^(i)(φ) ∩ D^(j)} f")
  function F4(φ1,V_φ1,φ2,V_φ2,φ3,V_φ3,Ω_name)
    φh1 = FEFunction(V_φ1, φ1)
    φh2 = FEFunction(V_φ2, φ2)
    φh3 = FEFunction(V_φ3, φ3)
    cutgeo = compute_geo(φh1,φh2,φh3)
    Di_cap_Dj = Triangulation(cutgeo,Ω_name)
    dΩ = Measure(Di_cap_Dj,order*2)
    return sum(∫(fh)dΩ)
  end
  function F4_ad(φ1,V_φ1,φ2,V_φ2,φ3,V_φ3,Ω_name)
    φh1 = FEFunction(V_φ1, φ1)
    φh2 = FEFunction(V_φ2, φ2)
    φh3 = FEFunction(V_φ3, φ3)
    cutgeo = compute_geo(φh1,φh2,φh3)
    Di_cap_Dj = DifferentiableTriangulation(cutgeo,Ω_name)
    dΩ = Measure(Di_cap_Dj,order*2)
    return φ -> ∫(fh)dΩ
  end
  function F4_analytic(φhi,Ωi_name,Ωj_name,coeff=-1)
    ∂Di_cap_Dj = EmbeddedBoundary(cutgeo,Ωi_name,Ωj_name)
    n = ∇(φhi)/(norm ∘ (∇(φhi))) #get_normal_vector(∂Di_cap_Dj)
    dΓ = Measure(∂Di_cap_Dj,order*2)
    return w->∫(coeff*fh*w/abs(n⋅∇(φhi)))dΓ
  end

  _φ1 = get_free_dof_values(φh1)
  _φ2 = get_free_dof_values(φh2)
  _φ3 = get_free_dof_values(φh3)

  cases = Dict(
    # Ω1(φ1,φ2,φ3) = D^(1)(φ1) ∩ D^(2)ᶜ(φ2) ∩ D^(3)ᶜ(φ3)
    "Ω1" => (("Ω1", "Ω5", -1),("Ω1", "Ω2", 1),("Ω1", "Ω3", 1)),
    # Ω2(φ1,φ2,φ3) = D^(1)(φ1) ∩ D^(2)(φ2) ∩ D^(3)ᶜ(φ3)
    "Ω2" => (("Ω2", "Ω6", -1),("Ω2", "Ω1", -1),("Ω2", "Ω4", 1)),
    # Ω3(φ1,φ2,φ3) = D^(1)(φ1) ∩ D^(2)ᶜ(φ2) ∩ D^(3)(φ3)
    "Ω3" => (("Ω3", "Ω7", -1),("Ω3", "Ω4", 1),("Ω1", "Ω3", -1)),
    # Ω4(φ1,φ2,φ3) = D^(1)(φ1) ∩ D^(2)(φ2) ∩ D^(3)(φ3)
    "Ω4" => (("Ω4", "Ω8", -1),("Ω3", "Ω4", -1),("Ω4", "Ω2", -1)),
    # Ω5(φ1,φ2,φ3) = D^(1)ᶜ(φ1) ∩ D^(2)ᶜ(φ2) ∩ D^(3)ᶜ(φ3)
    "Ω5" => (("Ω5", "Ω1", 1),("Ω5", "Ω6", 1),("Ω5", "Ω7", 1)),
    # Ω6(φ1,φ2,φ3) = D^(1)ᶜ(φ1) ∩ D^(2)(φ2) ∩ D^(3)ᶜ(φ3)
    "Ω6" => (("Ω6", "Ω2", 1),("Ω6", "Ω5", -1),("Ω6", "Ω8", 1)),
    # Ω7(φ1,φ2,φ3) = D^(1)ᶜ(φ1) ∩ D^(2)ᶜ(φ2) ∩ D^(3)(φ3)
    "Ω7" => (("Ω7", "Ω3", 1),("Ω7", "Ω8", 1),("Ω7", "Ω5", -1)),
    # Ω8(φ1,φ2,φ3) = D^(1)ᶜ(φ1) ∩ D^(2)(φ2) ∩ D^(3)(φ3)
    "Ω8" => (("Ω8", "Ω4", 1),("Ω8", "Ω7", -1),("Ω8", "Ω6", -1)),
  )
  for case in keys(cases)
    _f_ad = F4_ad(_φ1,V_φ1,_φ2,V_φ2,_φ3,V_φ3,case)
    @info "Testing case φ1 -> $case(φ1,φ2,φ3)"
    dF_ad_vec = assemble_vector(gradient(_f_ad,φh1),V_φ1)
    dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F4(φ, V_φ1, _φ2, V_φ2, _φ3, V_φ3, case), _φ1)
    err_fdm = maximum(abs, dF_ad_vec) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_ad_vec) : maximum(abs,dF_fdm - dF_ad_vec)
    @info "   AD vs FDM:" err_fdm
    @test err_fdm < 1e-6
    @info "Testing case φ2 -> $case(φ1,φ2,φ3)"
    dF_ad_vec = assemble_vector(gradient(_f_ad,φh2),V_φ2)
    dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F4(_φ1, V_φ1, φ, V_φ2, _φ3, V_φ3, case), _φ2)
    err_fdm = maximum(abs, dF_ad_vec) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_ad_vec) : maximum(abs,dF_fdm - dF_ad_vec)
    @info "   AD vs FDM:" err_fdm
    @test err_fdm < 1e-7
    @info "Testing case φ3 -> $case(φ1,φ2,φ3)"
    dF_ad_vec = assemble_vector(gradient(_f_ad,φh3),V_φ3)
    dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F4(_φ1, V_φ1, _φ2, V_φ2, φ, V_φ3, case), _φ3)
    err_fdm = maximum(abs, dF_ad_vec) > 0 ? maximum(abs,dF_fdm - dF_ad_vec) / maximum(abs, dF_ad_vec) : maximum(abs,dF_fdm - dF_ad_vec)
    @info "   AD vs FDM:" err_fdm
    @test err_fdm < 1e-7
  end

  function F7(φi,V_φi,Ωi_name,Ωj_name)
    φhi = FEFunction(V_φi, φi)
    cutgeo = compute_geo(φhi...)
    Γ = EmbeddedBoundary(cutgeo,Ωi_name,Ωj_name)
    dΓ = Measure(Γ,order*2)
    return sum(∫(fh)dΓ)
  end
  function F7_ad(φi,V_φi,Ωi_name,Ωj_name)
    φhi = FEFunction(V_φi, φi)
    cutgeo = compute_geo(φhi...)
    Γ = DifferentiableEmbeddedBoundary(cutgeo,Ωi_name,Ωj_name)
    dΓ = Measure(Γ,order*2)
    return _ -> ∫(fh)dΓ
  end

  for i = 1:8
    for j = i+1:8
      println(" #### Case b & c: φ -> ∫_{Γ_$i$j} f -- AD VS FDM ONLY")
      _φi = get_free_dof_values(φhi)
      dF_ad_vec = assemble_vector(gradient(F7_ad(_φi, V_φi, "Ω$i","Ω$j"),φhi),V_φi);
      dF_fdm = FiniteDiff.finite_difference_gradient(φ -> F7(φ, V_φi, "Ω$i","Ω$j"), _φi);
      err_fdm = maximum(abs, dF_fdm) > 0 ? maximum(abs, dF_fdm - dF_ad_vec) / maximum(abs, dF_fdm) : maximum(abs, dF_fdm - dF_ad_vec)
      @info "   AD vs FDM:" err_fdm
      @test err_fdm < 1e-6
    end
  end
end

end