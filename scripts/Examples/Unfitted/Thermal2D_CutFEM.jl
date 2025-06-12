using Gridap,GridapTopOpt, GridapSolvers
using Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters

using GridapTopOpt: StateParamMap

"""
  (Serial) Minimum thermal compliance with a CutFEM formulation based on Burman et al. (2015)
    [10.1002/nme.4823] & automatic shape differentiation in 2D.

  Optimisation problem:
      Min J(Ω) = ∫ ∇(u)⋅∇(u) dΩ
       Ω
    s.t., Vol(Ω) = 0.3,
          ⎡u∈V=H¹(Ω;u(Γ_D)=0),
          ⎣∫ ∇(u)⋅∇(v) dΩ + j(u,v) + i(u,v) = ∫ v dΓ_N, ∀v∈V.

  In the above, j(u,v) is the ghost penalty term over the ghost skeleton Γg
  with outward normal n_Γg, and i(u,v) enforces zero temperature within the
  isolated volumes marked by χ. There are given by
      j(u,v) = ∫ γh[[∇(u)⋅n_Γg]][[(∇(v)⋅n_Γg]] dΓg, &
      i(u,v) = ∫ χuv dΩ.
"""
function main()
  path="./results/Unfitted_Thermal2D/"
  mkpath(path)
  # Params
  n = 50
  order = 1
  γ = 0.1
  max_steps = 10
  vf = 0.3
  α_coeff = 2
  iter_mod = 1

  # Model and some refinement
  _model = CartesianDiscreteModel((0,1,0,1),(n,n))
  base_model = UnstructuredDiscreteModel(_model)
  ref_model = refine(base_model, refinement_method = "barycentric")
  ref_model = refine(ref_model)
  ref_model = refine(ref_model)
  model = ref_model.model
  h = minimum(get_element_diameters(model))
  hₕ = get_element_diameter_field(model)
  f_Γ_D(x) = (x[1]-0.5)^2 + (x[2]-0.5)^2 <= 0.05^2
  f_Γ_N(x) = ((x[1] ≈ 0 || x[1] ≈ 1) && (0.2 <= x[2] <= 0.3 + eps() || 0.7 - eps() <= x[2] <= 0.8)) ||
    ((x[2] ≈ 0 || x[2] ≈ 1) && (0.2 <= x[1] <= 0.3 + eps() || 0.7 - eps() <= x[1] <= 0.8))
  update_labels!(1,model,f_Γ_D,"Gamma_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")
  writevtk(model,path*"model")

  ## Triangulations and measures
  Ω = Triangulation(model)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΩ = Measure(Ω,2*order)
  dΓ_N = Measure(Γ_N,2*order)
  vol_D = sum(∫(1)dΩ)

  ## Levet-set function space and derivative regularisation space
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_D","Gamma_N"])
  U_reg = TrialFESpace(V_reg)
  V_φ = TestFESpace(model,reffe_scalar)

  ## Levet-set function
  f1 = (x,y) -> -cos(6π*(x-1/12))*cos(6π*(y-1/12))-0.5
  f2 = (x,y) -> -cos(6π*(x-3/12))*cos(6π*(y-1/12))-0.5
  f3 = (x,y) -> (x-0.5)^2 + (y-0.5)^2 - 0.06^2
  f((x,y)) = min(max(f1(x,y),f2(x,y)),f3(x,y))

  φh = interpolate(f,V_φ)
  Ωs = EmbeddedCollection(model,φh) do cutgeo,_,_
    Ωin = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
    Γ = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
    Γg = GhostSkeleton(cutgeo)
    Ωact = Triangulation(cutgeo,ACTIVE)
    (;
      :Ωin  => Ωin,
      :dΩin => Measure(Ωin,2*order),
      :Γg   => Γg,
      :dΓg  => Measure(Γg,2*order),
      :n_Γg => get_normal_vector(Γg),
      :Γ    => Γ,
      :dΓ   => Measure(Γ,2*order),
      :n_Γ  => get_normal_vector(Γ), # Note, need to recompute inside obj/constraints to compute derivs
      :Ωact => Ωact,
      :χ => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_D"])
    )
  end

  ## Weak form
  const γg = 0.1
  a(u,v,φ) = ∫(∇(v)⋅∇(u))Ωs.dΩin +
    ∫((γg*mean(hₕ))*jump(Ωs.n_Γg⋅∇(v))*jump(Ωs.n_Γg⋅∇(u)))Ωs.dΓg +
    ∫(Ωs.χ*v*u)Ωs.dΩin
  l(v,φ) = ∫(v)dΓ_N

  ## Optimisation functionals
  J(u,φ) = ∫(∇(u)⋅∇(u))Ωs.dΩin
  Vol(u,φ) = ∫(1/vol_D)Ωs.dΩin - ∫(vf/vol_D)dΩ
  dVol(q,u,φ) = ∫(-1/vol_D*q/(abs(Ωs.n_Γ ⋅ ∇(φ))))Ωs.dΓ

  ## Setup solver and FE operators
  state_collection = GridapTopOpt.EmbeddedCollection_in_φh(model,φh) do _φh
    V = TestFESpace(Ωs.Ωact,reffe_scalar;dirichlet_tags=["Gamma_D"])
    U = TrialFESpace(V,0.0)
    state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,_φh)
    (;
      :state_map => state_map,
      :J => StateParamMap(J,state_map),
      :C => map(Ci -> StateParamMap(Ci,state_map),[Vol,])
    )
  end
  pcfs = EmbeddedPDEConstrainedFunctionals(state_collection;analytic_dC=(dVol,))

  ## Evolution Method
  evo = CutFEMEvolve(V_φ,Ωs,dΩ,hₕ;max_steps,γg=0.1)
  reinit = StabilisedReinit(V_φ,Ωs,dΩ,hₕ;stabilisation_method=ArtificialViscosity(2.0))
  ls_evo = UnfittedFEEvolution(evo,reinit)
  reinit!(ls_evo,φh)

  ## Hilbertian extension-regularisation problems
  α = (α_coeff)^2*hₕ*hₕ
  a_hilb(p,q) =∫(α*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

  ## Optimiser
  converged(m) = GridapTopOpt.default_al_converged(
    m;
    L_tol = 0.01*h,
    C_tol = 0.01
  )
  optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;verbose=true,constraint_names=[:Vol],converged)
  for (it,uh,φh) in optimiser
    if iszero(it % iter_mod)
      writevtk(Ω,path*"Omega$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"χ"=>Ωs.χ])
      writevtk(Ωs.Ωin,path*"Omega_in$it",cellfields=["uh"=>uh])
    end
    write_history(path*"/history.txt",optimiser.history)
  end
  it = get_history(optimiser).niter; uh = get_state(pcfs)
  writevtk(Ω,path*"Omega$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"χ"=>Ωs.χ])
  writevtk(Ωs.Ωin,path*"Omega_in$it",cellfields=["uh"=>uh])
end

main()