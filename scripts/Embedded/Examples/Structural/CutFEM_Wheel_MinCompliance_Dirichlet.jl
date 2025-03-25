using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt

using GridapDistributed,PartitionedArrays,GridapPETSc

if isassigned(ARGS,1)
  global γg_evo =  parse(Float64,ARGS[1])
else
  global γg_evo =  0.01
end

MUMPSSolver() = PETScLinearSolver(petsc_mumps_setup)

function petsc_mumps_setup(ksp)
  pc       = Ref{GridapPETSc.PETSC.PC}()
  mumpsmat = Ref{GridapPETSc.PETSC.Mat}()

  @check_error_code GridapPETSc.PETSC.KSPSetFromOptions(ksp[])

  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPPREONLY)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCLU)
  @check_error_code GridapPETSc.PETSC.PCFactorSetMatSolverType(pc[],GridapPETSc.PETSC.MATSOLVERMUMPS)
  @check_error_code GridapPETSc.PETSC.PCFactorSetUpMatSolverType(pc[])
  @check_error_code GridapPETSc.PETSC.PCFactorGetMatrix(pc[],mumpsmat)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  4, 1)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 28, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 14, 50000)
  @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
end

CGAMGSolver(;kwargs...) = PETScLinearSolver(gamg_ksp_setup(;kwargs...))

function gamg_ksp_setup(;rtol=10^-8,maxits=100)

  function ksp_setup(ksp)
    pc = Ref{GridapPETSc.PETSC.PC}()

    rtol = PetscScalar(rtol)
    atol = GridapPETSc.PETSC.PETSC_DEFAULT
    dtol = GridapPETSc.PETSC.PETSC_DEFAULT
    maxits = PetscInt(maxits)

    @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPCG)
    @check_error_code GridapPETSc.PETSC.KSPSetTolerances(ksp[], rtol, atol, dtol, maxits)
    @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
    @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCGAMG)
    @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
  end

  return ksp_setup
end

function main(ranks)
  # Params
  vf = 0.3
  γ_evo = 0.1
  max_steps = 10
  α_coeff = γ_evo*max_steps
  iter_mod = 1
  D = 3

  # Output path
  path = "./results/CutFEM_Wheel_MinCompliance_Dirichlet_gammag_$(γg_evo)_vf_$(vf)/"
  files_path = path*"data/"
  i_am_main(ranks) && mkpath(files_path)

  # Load mesh
  model = GmshDiscreteModel(ranks,(@__DIR__)*"/Meshes/wheel.msh")
  model = UnstructuredDiscreteModel(model)

  # Add non-designable region
  f_NonDesign(x) = 0.1 < sqrt(x[1]^2+x[2]^2) <= 0.2 + eps()
  update_labels!(1,model,f_NonDesign,"Omega_NonDesign")
  writevtk(model,path*"model")

  # Get triangulation and element size
  Ω_bg = Triangulation(model)
  hₕ = get_element_diameter_field(model)
  hmin = minimum(get_element_diameters(model))

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N","Omega_NonDesign"])
  U_reg = TrialFESpace(V_reg)

  f((x,y,z),q,r) = - cos(q*π*x)*cos(q*π*y)*cos(q*π*z)/q - r/q
  g((x,y,z)) = sqrt(x^2 + y^2) - 0.2
  lsf(x) = min(f(x,4,0.1),g(x))
  φh = interpolate(lsf,V_φ)
  φh_nondesign = interpolate(g,V_φ)

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
    φ_cell_values = map(get_cell_dof_values,local_views(_φh))
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
      :n_Γ        => get_normal_vector(Γ), # Note, need to recompute inside obj/constraints to compute derivs
      :ψ     => ψ
    )
  end
  writevtk(get_triangulation(φh),path*"initial_islands",cellfields=["φh"=>φh,"ψ"=>Ω_data.ψ])
  writevtk(Ω_data.Ω,path*"Omega_initial")

  # Setup spaces
  uin((x,y,z)) = 0.1VectorValue(-y,x,0.0)

  reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
  function build_spaces(Ω_act)
    V = TestFESpace(Ω_act,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_D","Gamma_N"])
    U = TrialFESpace(V,[VectorValue(0.0,0.0,0.0),uin])
    return U,V
  end
  _U_init = build_spaces(Ω_data.Ω_act)[1]
  i_am_main(ranks) && println("Number of free dofs: ",num_free_dofs(_U_init))

  ### Weak form
  # Material parameters
  function lame_parameters(E,ν)
    λ = (E*ν)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    (λ, μ)
  end
  λs, μs = lame_parameters(10^4,0.3)
  # Stabilization
  α_Gd = 1e-7
  k_d = 1.0
  γ_Gd(h) = α_Gd*(λs + μs)*h^3
  # Terms
  σ(ε) = λs*tr(ε)*one(ε) + 2*μs*ε
  a_s_Ω(d,s) = ε(s) ⊙ (σ ∘ ε(d)) # Elasticity
  j_s_k(d,s) = mean(γ_Gd ∘ hₕ)*(jump(Ω_data.n_Γg ⋅ ∇(s)) ⋅ jump(Ω_data.n_Γg ⋅ ∇(d)))
  v_s_ψ(d,s) = (k_d*Ω_data.ψ)*(d⋅s) # Isolated volume term
  _g = VectorValue(0.0,0.0,0.0)

  a(d,s,φ) = ∫(a_s_Ω(d,s) + v_s_ψ(d,s))Ω_data.dΩ + ∫(j_s_k(d,s))Ω_data.dΓg
  l(s,φ) = ∫(s⋅_g)dΓ_N

  ## Optimisation functionals
  vol_D = sum(∫(1)dΩ_bg)
  iso_vol_frac(φ) = ∫(Ω_data.ψ/vol_D)Ω_data.dΩ
  J_comp(d,φ) = ∫(ε(d) ⊙ (σ ∘ ε(d)))Ω_data.dΩ + iso_vol_frac(φ) + ∫(35)Ω_data.dΩ
  Vol(d,φ) = ∫(1/vol_D)Ω_data.dΩ - ∫(vf/vol_D)dΩ_bg
  dVol(q,d,φ) = ∫(-1/vol_D*q/(abs(Ω_data.n_Γ ⋅ ∇(φ))))Ω_data.dΓ

  ## Setup solver and FE operators
  elast_ls = MUMPSSolver()
  state_collection = GridapTopOpt.EmbeddedCollection_in_φh(model,φh) do _φh
    update_collection!(Ω_data,_φh)
    U,V = build_spaces(Ω_data.Ω_act)
    # elast_ls = ElasticitySolver(U;rtol=1.e-8,maxits=200)
    state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,_φh;ls=elast_ls,adjoint_ls=elast_ls)
    (;
      :state_map => state_map,
      :J => GridapTopOpt.StateParamMap(J_comp,state_map),
      :C => map(Ci -> GridapTopOpt.StateParamMap(Ci,state_map),GridapTopOpt.StateParamMap[])#,[Vol,])
    )
  end

  pcf = EmbeddedPDEConstrainedFunctionals(state_collection)#;analytic_dC=(dVol,))

  ## Evolution Method
  evolve_ls = MUMPSSolver()
  evolve_nls = NewtonSolver(evolve_ls;maxiter=1,verbose=i_am_main(ranks))
  reinit_nls = NewtonSolver(MUMPSSolver();maxiter=20,rtol=1.e-14,verbose=i_am_main(ranks))

  evo = CutFEMEvolve(V_φ,Ω_data,dΩ_bg,hₕ;max_steps,γg=γg_evo,ode_ls=evolve_ls,ode_nl=evolve_nls)
  reinit1 = StabilisedReinit(V_φ,Ω_data,dΩ_bg,hₕ;stabilisation_method=ArtificialViscosity(0.5),nls=reinit_nls)
  reinit2 = StabilisedReinit(V_φ,Ω_data,dΩ_bg,hₕ;stabilisation_method=GridapTopOpt.InteriorPenalty(V_φ,γg=0.1),nls=reinit_nls)
  reinit = GridapTopOpt.MultiStageStabilisedReinit([reinit1,reinit2])
  ls_evo = UnfittedFEEvolution(evo,reinit)

  ## Hilbertian extension-regularisation problems
  hilb_ls = CGAMGSolver()
  _α(hₕ) = (α_coeff*hₕ)^2
  a_hilb(p,q) =∫((_α ∘ hₕ)*∇(p)⋅∇(q) + p*q)dΩ_bg;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg;ls=hilb_ls)

  ## Optimiser
  converged(m) = GridapTopOpt.default_al_converged(
    m;
    L_tol = 0.01hmin,
    C_tol = 0.01
  )
  optimiser = AugmentedLagrangian(pcf,ls_evo,vel_ext,φh;
    γ=γ_evo,verbose=i_am_main(ranks),converged)#,constraint_names=[:Vol],converged)
  for (it,uh,φh) in optimiser
    GC.gc()
    if iszero(it % iter_mod)
      writevtk(Ω_bg,path*"Omega_act_$it",
        cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ψ"=>Ω_data.ψ])
      writevtk(Ω_data.Ω,path*"Omega_in_$it",cellfields=["uh"=>uh])
    end
    write_history(path*"/history.txt",optimiser.history;ranks)

    isolated_vol = sum(iso_vol_frac(φh))
    i_am_main(ranks) && println(" --- Isolated volume: ",isolated_vol)

    # Geometric operation to re-add the non-designable region # TODO: Move to a function
    _φ = get_free_dof_values(φh)
    _φ_nondesign = get_free_dof_values(φh_nondesign)
    map(local_views(_φ),local_views(_φ_nondesign)) do φ,φ_nondesign
      φ .= min.(φ,φ_nondesign)
    end
    consistent!(_φ) |> wait
    reinit!(ls_evo,φh)
  end
  it = get_history(optimiser).niter; uh = get_state(pcf)
  writevtk(Ω_bg,path*"Omega_act_$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ψ"=>Ω_data.ψ])
  writevtk(Ω_data.Ω,path*"Omega_in_$it",cellfields=["uh"=>uh])
end

with_mpi() do distribute
  ncpus = 48
  ranks = distribute(LinearIndices((ncpus,)))
  petsc_options = "-ksp_converged_reason -ksp_error_if_not_converged true"
  GridapPETSc.with(;args=split(petsc_options)) do
    main(ranks)
  end
end