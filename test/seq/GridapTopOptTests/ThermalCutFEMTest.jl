module ThermalCutFEMTest
using Test
using Gridap, Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt, GridapSolvers

using GridapTopOpt: StateParamMap

function main(AD_case)
  # Params
  n = 10            # Initial mesh size (pre-refinement)
  max_steps = 10/n  # Time-steps for evolution equation
  vf = 0.3          # Volume fraction
  α_coeff = 2       # Regularisation coefficient extension-regularisation

  # Model and some refinement
  _model = CartesianDiscreteModel((0,1,0,1),(n,n))
  base_model = UnstructuredDiscreteModel(_model)
  ref_model = refine(base_model, refinement_method = "barycentric")
  #ref_model = refine(ref_model)
  #ref_model = refine(ref_model)
  model = get_model(ref_model)
  h = minimum(get_element_diameters(model))
  hₕ = get_element_diameter_field(model)
  f_Γ_D(x) = (x[1]-0.5)^2 + (x[2]-0.5)^2 <= 0.05^2
  f_Γ_N(x) = ((x[1] ≈ 0 || x[1] ≈ 1) && (0.2 <= x[2] <= 0.3 + eps() || 0.7 - eps() <= x[2] <= 0.8)) ||
    ((x[2] ≈ 0 || x[2] ≈ 1) && (0.2 <= x[1] <= 0.3 + eps() || 0.7 - eps() <= x[1] <= 0.8))
  update_labels!(1,model,f_Γ_D,"Omega_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")

  ## Levet-set function space and derivative regularisation space
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Omega_D","Gamma_N"])
  U_reg = TrialFESpace(V_reg)
  V_φ = TestFESpace(model,reffe_scalar)

  ## Level-set function
  f1 = (x,y) -> -cos(6π*(x-1/12))*cos(6π*(y-1/12))-0.5
  f2 = (x,y) -> -cos(6π*(x-3/12))*cos(6π*(y-1/12))-0.5
  f3 = (x,y) -> (x-0.5)^2 + (y-0.5)^2 - 0.06^2
  f((x,y)) = min(max(f1(x,y),f2(x,y)),f3(x,y))
  φh = interpolate(f,V_φ)

  # Check LS
  GridapTopOpt.correct_ls!(φh)

  ## Triangulations and measures
  Ω_bg = Triangulation(model)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΩ_bg = Measure(Ω_bg,2)
  dΓ_N = Measure(Γ_N,2)
  vol_D = sum(∫(1)dΩ_bg)

  Ωs = EmbeddedCollection(model,φh) do cutgeo,cutgeo_facets,_φh
    Ωin = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
    Γ = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
    Γg = GhostSkeleton(cutgeo)
    Ωact = Triangulation(cutgeo,ACTIVE)
    # Isolated volumes
    φ_cell_values = get_cell_dof_values(_φh)
    χ,_ = get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Omega_D",])
    (;
      :Ωin  => Ωin,
      :dΩin => Measure(Ωin,2),
      :Γg   => Γg,
      :dΓg  => Measure(Γg,2),
      :n_Γg => get_normal_vector(Γg),
      :Γ    => Γ,
      :dΓ   => Measure(Γ,2),
      :n_Γ  => get_normal_vector(Γ),
      :Ωact => Ωact,
      :χ => χ
    )
  end

  ## Weak form
  γg = 0.1
  a(u,v,φ) = ∫(∇(v)⋅∇(u))Ωs.dΩin +
    ∫((γg*mean(hₕ))*jump(Ωs.n_Γg⋅∇(v))*jump(Ωs.n_Γg⋅∇(u)))Ωs.dΓg +
    ∫(Ωs.χ*v*u)Ωs.dΩin
  l(v,φ) = ∫(v)dΓ_N

  ## Optimisation functionals
  J(u,φ) = ∫(∇(u)⋅∇(u))Ωs.dΩin
  Vol(u,φ) = ∫(1/vol_D)Ωs.dΩin - ∫(vf/vol_D)dΩ_bg
  dVol(q,u,φ) = ∫(-1/vol_D*q/(abs(Ωs.n_Γ ⋅ ∇(φ))))Ωs.dΓ

  ## FE operators
  state_collection = EmbeddedCollection_in_φh(model,φh) do _φh
    update_collection!(Ωs,_φh)
    V = TestFESpace(Ωs.Ωact,reffe_scalar;dirichlet_tags=["Omega_D"])
    U = TrialFESpace(V,0.0)
    state_map = AffineFEStateMap(a,l,U,V,V_φ,_φh)
    (;
      :state_map => state_map,
      :J => StateParamMap(J,state_map),
      :C => map(Ci -> StateParamMap(Ci,state_map),[Vol,])
    )
  end

  pcfs = if AD_case == :with_ad
    EmbeddedPDEConstrainedFunctionals(state_collection;analytic_dC=(dVol,))
  elseif AD_case == :custom_pcf
    function φ_to_jc(φ)
      u = state_collection.state_map(φ)
      j = state_collection.J(u,φ)
      c = map(constrainti -> constrainti(u,φ),state_collection.C)
      [j,c...]
    end
    CustomEmbeddedPDEConstrainedFunctionals(φ_to_jc,1,state_collection)
  else
    @error "AD case not defined"
  end

  ## Evolution Method
  evo = CutFEMEvolver(V_φ,Ωs,dΩ_bg,hₕ;max_steps,γg=0.1)
  reinit = StabilisedReinitialiser(V_φ,Ωs,dΩ_bg,hₕ;stabilisation_method=ArtificialViscosity(2.0))
  ls_evo = LevelSetEvolution(evo,reinit)
  reinit!(ls_evo,φh)

  ## Hilbertian extension-regularisation problems
  α = (α_coeff)^2*hₕ*hₕ
  a_hilb(p,q) =∫(α*∇(p)⋅∇(q) + p*q)dΩ_bg;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

  ## Optimiser
  optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;verbose=true,constraint_names=[:Vol])

  # Do a few iterations
  vars, state = iterate(optimiser)
  vars, state = iterate(optimiser,state)
  true
end

@test main(:with_ad)
@test main(:custom_pcf)

end