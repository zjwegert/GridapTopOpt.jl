using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt

using GridapDistributed,PartitionedArrays,GridapPETSc

if isassigned(ARGS,1)
  global γg_evo =  parse(Float64,ARGS[1])
else
  global γg_evo =  0.05
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
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 14, 50)
  # @check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[],  1, 0.00001) # relative thresh
  # @check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[], 3, 1.0e-6) # absolute thresh
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
  path = "./results/FCM_Wheel_MinCompliance_gammag_$(γg_evo)/"
  files_path = path*"data/"
  i_am_main(ranks) && mkpath(files_path)

  γ_evo = 0.1
  vf = 0.3
  iter_mod = 1
  D = 3

  model = GmshDiscreteModel(ranks,(@__DIR__)*"/Meshes/wheel.msh")
  writevtk(model,path*"model")

  Ω_bg = Triangulation(model)
  hₕ = get_element_diameter_field(model)
  hmin = minimum(get_element_diameters(model))

  max_steps = 0.2/hmin
  α_coeff = γ_evo*max_steps

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N",])
  U_reg = TrialFESpace(V_reg)

  f((x,y,z),q,r) = - cos(q*π*x)*cos(q*π*y)*cos(q*π*z)/q - r/q
  φh = interpolate(x->f(x,4,0.1),V_φ)

  # Ensure values at DoFs are non-zero to satify assumptions for derivatives
  _φ = get_free_dof_values(φh)
  map(local_views(_φ)) do φ
    idx = findall(isapprox(0.0;atol=1e-10),φ)
    if !isempty(idx)
      i_am_main(ranks) && println("    Correcting level values at $(length(idx)) nodes")
    end
    φ[idx] .+= 1e-10
  end
  consistent!(_φ) |> wait

  # Setup integration meshes and measures
  order = 1
  degree = 2*(order+1)

  dΩ_bg = Measure(Ω_bg,degree)
  Ω_data = EmbeddedCollection(model,φh) do cutgeo,_,_
    Ω = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
    Ω_out = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
    Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
    Γg = GhostSkeleton(cutgeo)
    Ω_act = Triangulation(cutgeo,ACTIVE)
    (;
      :Ω_act  => Ω_act,
      :Ω      => Ω,
      :dΩ     => Measure(Ω,degree),
      :dΩ_out => Measure(Ω_out,degree),
      :Γg     => Γg,
      :dΓg    => Measure(Γg,degree),
      :n_Γg   => get_normal_vector(Γg),
      :Γ      => Γ,
      :dΓ     => Measure(Γ,degree),
      :n_Γ        => get_normal_vector(Γ), # Note, need to recompute inside obj/constraints to compute derivs
      :ψ      => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_D","Gamma_N"];groups=((CUT,IN),OUT)),
    )
  end
  writevtk(get_triangulation(φh),path*"initial_islands",cellfields=["φh"=>φh,"ψ"=>Ω_data.ψ])
  writevtk(Ω_data.Ω,path*"Omega_initial")

  # Setup spaces
  uin((x,y,z)) = 10VectorValue(-y,x,0.0)

  reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
  V = TestFESpace(model,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_D","Gamma_N"])
  U = TrialFESpace(V,[VectorValue(0.0,0.0,0.0),uin])
  i_am_main(ranks) && println("Number of free dofs: ",num_free_dofs(U))

  ### Weak form
  # Material parameters
  function lame_parameters(E,ν)
    λ = (E*ν)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    (λ, μ)
  end
  λs, μs = lame_parameters(1.0,0.3)
  ϵ = (λs + μs)*1e-7
  # Terms
  σ(ε) = λs*tr(ε)*one(ε) + 2*μs*ε
  a_s_Ω(d,s) = ε(s) ⊙ (σ ∘ ε(d)) # Elasticity
  e_s_Ω(d,s) = ϵ*a_s_Ω(d,s) # Ersatz

  a(d,s,φ) = ∫(a_s_Ω(d,s))Ω_data.dΩ + ∫(e_s_Ω(d,s))Ω_data.dΩ_out

  _g = VectorValue(0.0,0.0,0.0)
  l(s,φ) = ∫(s⋅_g)dΩ_bg

  ## Optimisation functionals
  vol_D = sum(∫(1)dΩ_bg)
  J_comp(d,φ) = a(d,d,φ)
  Vol(d,φ) = ∫(1/vol_D)Ω_data.dΩ - ∫(vf/vol_D)dΩ_bg
  dVol(q,d,φ) = ∫(-1/vol_D*q/(abs(Ω_data.n_Γ ⋅ ∇(φ))))Ω_data.dΓ

  ## Setup solver and FE operators
  elast_ls = ElasticitySolver(U;rtol=1.e-8,maxits=200)
  state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh;ls=elast_ls,adjoint_ls=elast_ls)
  pcf = PDEConstrainedFunctionals(J_comp,[Vol],state_map;analytic_dC=(dVol,))

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
  _α(hₕ) = (α_coeff*hₕ)^2
  a_hilb(p,q) =∫((_α ∘ hₕ)*∇(p)⋅∇(q) + p*q)dΩ_bg;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg;ls=CGAMGSolver())

  ## Optimiser
  converged(m) = GridapTopOpt.default_al_converged(
    m;
    L_tol = 0.01hmin,
    C_tol = 0.01
  )
  optimiser = AugmentedLagrangian(pcf,ls_evo,vel_ext,φh;
    γ=γ_evo,verbose=i_am_main(ranks),constraint_names=[:Vol],converged)
  for (it,uh,φh) in optimiser
    GC.gc()
    if iszero(it % iter_mod)
      writevtk(Ω_bg,path*"Omega_act_$it",
        cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ψ"=>Ω_data.ψ])
      writevtk(Ω_data.Ω,path*"Omega_in_$it",cellfields=["uh"=>uh])
    end
    write_history(path*"/history.txt",optimiser.history;ranks)
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
