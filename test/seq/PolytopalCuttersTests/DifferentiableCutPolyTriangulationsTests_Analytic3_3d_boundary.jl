# This module tests AD against analytic expressions
module DifferentiableCutPolyTriangulationsTests_Analytic3_3d

include("DifferentiableCutPolyTriangulationsTests_Utils.jl")

println("
################################################################
#        Test suite: AD vs analytic results (3LSF, rand)       #
################################################################
")

using Random

for d in (3,)
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