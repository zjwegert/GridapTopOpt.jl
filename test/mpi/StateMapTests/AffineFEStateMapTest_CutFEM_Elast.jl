module AffineFEStateMapTest_CutFEM_Elast

using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt
using PartitionedArrays, GridapDistributed

using FiniteDiff,Test

function main(_model,isserial)
  # Params
  vf = 0.3
  D = 2

  # Load mesh
  base_model = UnstructuredDiscreteModel(_model)
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = get_model(ref_model)
  f_Γ_D(x) = x[1] ≈ 0.0
  f_Γ_N(x) = (x[1] ≈ 1 && 0.4 - eps() <= x[2] <= 0.6 + eps())
  update_labels!(1,model,f_Γ_D,"Gamma_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")

  # Get triangulation and element size
  Ω_bg = Triangulation(model)
  hₕ = get_element_diameter_field(model)

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)

  f((x,y),q,r) = - cos(q*π*x)*cos(q*π*y)/q - r/q
  lsf(x) = f(x,4,0.1)
  φh = interpolate(lsf,V_φ)

  # Check LS
  GridapTopOpt.correct_ls!(φh)

  # Setup integration meshes and measures
  order = 1
  degree = 2*(order+1)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΓ_N = Measure(Γ_N,degree)
  dΩ_bg = Measure(Ω_bg,degree)
  Ω_data = EmbeddedCollection(model,φh) do cutgeo,cutgeo_facets,_φh
    Ω = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
    Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
    Γg = GhostSkeleton(cutgeo)
    Ω_act = Triangulation(cutgeo,ACTIVE)
    # Isolated volumes
    φ_cell_values = isserial ? get_cell_dof_values(_φh) : map(get_cell_dof_values,local_views(_φh))
    ψ,_ = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Gamma_D"])
    (;
      :Ω_act => Ω_act,
      :Ω     => Ω,
      :dΩ    => Measure(Ω,degree),
      :Γg    => Γg,
      :dΓg   => Measure(Γg,degree),
      :n_Γg  => get_normal_vector(Γg),
      :Γ     => Γ,
      :dΓ    => Measure(Γ,degree),
      :n_Γ   => get_normal_vector(Γ), # Note, need to recompute inside obj/constraints to compute derivs
      :ψ     => ψ
    )
  end

  # Setup spaces
  uin((x,y)) = 0.3VectorValue(-y,x)

  reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
  function build_spaces(Ω_act)
    V = TestFESpace(Ω_act,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_D","Gamma_N"])
    U = TrialFESpace(V,[VectorValue(0.0,0.0),uin])
    return U,V
  end

  ### Weak form
  # Material parameters
  function lame_parameters(E,ν)
    λ = (E*ν)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    (λ, μ)
  end
  λs, μs = lame_parameters(1.0,0.3)
  # Stabilization
  α_Gd = 1e-7
  k_d = 1.0
  γ_Gd(h) = α_Gd*(λs + μs)*h^3
  # Terms
  σ(ε) = λs*tr(ε)*one(ε) + 2*μs*ε
  a_s_Ω(d,s) = ε(s) ⊙ (σ ∘ ε(d)) # Elasticity
  j_s_k(d,s) = mean(γ_Gd ∘ hₕ)*(jump(Ω_data.n_Γg ⋅ ∇(s)) ⋅ jump(Ω_data.n_Γg ⋅ ∇(d)))
  v_s_ψ(d,s) = (k_d*Ω_data.ψ)*(d⋅s) # Isolated volume term
  _g = VectorValue(0.0,0.0)

  a(d,s,φ) = ∫(a_s_Ω(d,s) + v_s_ψ(d,s))Ω_data.dΩ + ∫(j_s_k(d,s))Ω_data.dΓg
  l(s,φ) = ∫(s⋅_g)dΓ_N

  ## Optimisation functionals
  vol_D = sum(∫(1)dΩ_bg)
  J_comp(d,φ) = ∫(ε(d) ⊙ (σ ∘ ε(d)))Ω_data.dΩ
  Vol(d,φ) = ∫(1/vol_D)Ω_data.dΩ - ∫(vf/vol_D)dΩ_bg
  dVol(q,d,φ) = ∫(-1/vol_D*q/(abs(Ω_data.n_Γ ⋅ ∇(φ))))Ω_data.dΓ

  ## Setup solver and FE operators
  state_collection = GridapTopOpt.EmbeddedCollection_in_φh(model,φh) do _φh
    update_collection!(Ω_data,_φh)
    U,V = build_spaces(Ω_data.Ω_act)
    state_map = AffineFEStateMap(a,l,U,V,V_φ)
    (;
      :state_map => state_map,
      :J => GridapTopOpt.StateParamMap(J_comp,state_map),
      :C => map(Ci -> GridapTopOpt.StateParamMap(Ci,state_map),[Vol,])
    )
  end

  pcfs = EmbeddedPDEConstrainedFunctionals(state_collection;analytic_dC=(dVol,))
  _,_,_dF,_ = evaluate!(pcfs,φh)

  if !isserial

    function custom_embedded_φ_to_j(φ)
      u = state_collection.state_map(φ)
      state_collection.J(u,φ)
    end

    cpcfs = CustomEmbeddedPDEConstrainedFunctionals(custom_embedded_φ_to_j,0,state_collection)
    _,_,cdF,_ = evaluate!(cpcfs,φh)
    @test cdF ≈ _dF

    function custom_embedded_φ_to_j_v2(φ)
      u = state_collection.state_map(φ)
      [state_collection.J(u,φ),state_collection.C[1](u,φ)]
    end

    cpcfs = CustomEmbeddedPDEConstrainedFunctionals(custom_embedded_φ_to_j_v2,1,state_collection)
    _,_,cdF,cdC = evaluate!(cpcfs,φh)
    @test cdF ≈ _dF

    function analytic_dVol!(dV,φ)
      φh = FEFunction(V_φ,φ)
      dh = get_state(state_collection.state_map)
      _dC(q) = dVol(q,dh,φh)
      Gridap.FESpaces.assemble_vector!(_dC,dV,V_φ)
    end
    cpcfs = CustomEmbeddedPDEConstrainedFunctionals(custom_embedded_φ_to_j_v2,1,state_collection;analytic_dC=[analytic_dVol!,])
    _,_,cdF,cdC_2 = evaluate!(cpcfs,φh)
    @test cdF ≈ _dF
    @test cdC[1] ≈ cdC_2[1]
  end

  return _dF,V_φ
end

function run_test(distribute,mesh_partition)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  model_serial = CartesianDiscreteModel((0,1,0,1),(7,7));
  dF_serial,V_deriv_serial = main(model_serial,true);

  model = GridapTopOpt.ordered_distributed_model_from_serial_model(ranks,model_serial);
  dF,V_deriv = main(model,false);

  @test length(dF_serial) == length(dF)
  @test norm(dF_serial) ≈ norm(dF)

  dFh = FEFunction(V_deriv,dF)
  dFh_serial = FEFunction(V_deriv_serial,dF_serial)
  deriv_test = GridapTopOpt.test_serial_and_distributed_fields(dFh,V_deriv,dFh_serial,V_deriv_serial)

  map_main(deriv_test) do deriv_test
    @test deriv_test
    nothing
  end
end

with_mpi() do distribute
  run_test(distribute,(2,2))
end

end