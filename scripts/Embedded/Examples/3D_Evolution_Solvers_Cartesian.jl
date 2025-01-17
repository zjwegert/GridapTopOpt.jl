using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt

using GridapDistributed,PartitionedArrays,GridapPETSc

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
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 1)
  # @check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[],  1, 0.00001) # relative thresh
  # @check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[], 3, 1.0e-6) # absolute thresh
  @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
end

CGAMGSolver(;rtol=10^-8,maxits=100) = PETScLinearSolver(cg_gamg_ksp_setup(rtol,maxits))

function cg_gamg_ksp_setup(rtol,maxits)

  function ksp_setup(ksp)
    pc = Ref{GridapPETSc.PETSC.PC}()

    @check_error_code GridapPETSc.PETSC.KSPSetFromOptions(ksp[])

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

GMRESAMGSolver(;rtol=10^-8,maxits=100) = PETScLinearSolver(gmres_amg_ksp_setup(rtol,maxits))

function gmres_amg_ksp_setup(rtol,maxits)

  function ksp_setup(ksp)
    pc = Ref{GridapPETSc.PETSC.PC}()

    @check_error_code GridapPETSc.PETSC.KSPSetFromOptions(ksp[])

    rtol = PetscScalar(rtol)
    atol = GridapPETSc.PETSC.PETSC_DEFAULT
    dtol = GridapPETSc.PETSC.PETSC_DEFAULT
    maxits = PetscInt(maxits)

    @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPGMRES)
    @check_error_code GridapPETSc.PETSC.KSPSetTolerances(ksp[], rtol, atol, dtol, maxits)
    @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
    @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCGAMG)
    @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
  end

  return ksp_setup
end

function main(ranks,mesh_partition)
  path = "./results/MPI_test_evo_cartesian/"
  i_am_main(ranks) && mkpath(path)

  n = 50
  _model = CartesianDiscreteModel(ranks,mesh_partition,(0,1,0,1,0,1),(n,n,n))
  base_model = UnstructuredDiscreteModel(_model)
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)
  h = 1/n;
  max_steps = floor(Int,n/10)
  writevtk(model,path*"model")

  Ω_act = Triangulation(model)

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)

  # _e = 5e-3
  # f0((x,y,z),a,b) = max(2/a*abs(x-x0),1/(b/2+1)*abs(y-b/2+1),2/(H-2cw)*abs(z-H/2))-1
  # f1((x,y,z),q,r) = - cos(q*π*x)*cos(q*π*y)*cos(q*π*z)/q - r/q
  # fin(x) = f0(x,l*(1+5_e),a*(1+5_e))
  # fsolid(x) = min(f0(x,l*(1+_e),b*(1+_e)),f0(x,w*(1+_e),a*(1+_e)))
  # fholes((x,y,z),q,r) = max(f1((x,y,z),q,r),f1((x-1/q,y,z),q,r))
  # # lsf(x) = min(min(max(fin(x),fholes(x,5,0.5)),fsolid(x)),0.1)
  # lsf(x) = min(max(fin(x),fholes(x,5,0.5)),fsolid(x))
  # # lsf((x,y,z)) = ((x-x0)^2 + (y-a/2)^2 + (z-0.5)^2) - (a/4)^2

  lsf = initial_lsf(4,0.2)
  φh = interpolate(lsf,V_φ)
  writevtk(get_triangulation(φh),path*"initial_lsf",cellfields=["φ"=>φh])

  # Setup integration meshes and measures
  order = 1
  degree = 2*(order+1)

  dΩ_act = Measure(Ω_act,degree)
  Ω = EmbeddedCollection(model,φh) do cutgeo,_
    Ωs = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
    Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
    Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
    Γg = GhostSkeleton(cutgeo)
    (;
      :Ωs      => Ωs,
      :dΩs     => Measure(Ωs,degree),
      :Ωf      => Ωf,
      :dΩf     => Measure(Ωf,degree),
      :Γg      => Γg,
      :dΓg     => Measure(Γg,degree),
      :n_Γg    => get_normal_vector(Γg),
      :Γ       => Γ,
      :dΓ      => Measure(Γ,degree)
    )
  end

  ## Evolution Method
  evolve_ls = GMRESAMGSolver()
  reinit1_nls = NewtonSolver(GMRESAMGSolver();maxiter=10,rtol=1.e-14,verbose=i_am_main(ranks))
  reinit2_nls = NewtonSolver(GMRESAMGSolver();maxiter=10,rtol=1.e-14,verbose=i_am_main(ranks))

  evo = CutFEMEvolve(V_φ,Ω,dΩ_act,h;max_steps,γg=0.01,ode_ls=evolve_ls)
  reinit1 = StabilisedReinit(V_φ,Ω,dΩ_act,h;stabilisation_method=ArtificialViscosity(1.0),nls=reinit1_nls)
  reinit2 = StabilisedReinit(V_φ,Ω,dΩ_act,h;stabilisation_method=GridapTopOpt.InteriorPenalty(V_φ,γg=1.0),nls=reinit2_nls)
  reinit = GridapTopOpt.MultiStageStabilisedReinit([reinit1,reinit2])
  ls_evo = UnfittedFEEvolution(evo,reinit)

  writevtk(Ω_act,path*"Omega_act_0",
    cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh))])
  writevtk(Ω.Ωs,path*"Omega_s_0")

  reinit!(ls_evo,φh)

  writevtk(Ω_act,path*"Omega_act_1",
    cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh))])
  writevtk(Ω.Ωs,path*"Omega_s_1")

  # velh = interpolate(x->-1,V_φ)
  # evolve!(ls_evo,φh,velh,0.1)

  # writevtk(Ω_act,path*"Omega_act_2",
  #   cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh))])
  # writevtk(Ω.Ωs,path*"Omega_s_2")
  nothing
end

with_mpi() do distribute
  mesh_partition = (1,2,2)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  petsc_options = "-ksp_converged_reason -ksp_error_if_not_converged true -ksp_gmres_modifiedgramschmidt"
  GridapPETSc.with(;args=split(petsc_options)) do
    main(ranks,mesh_partition)
  end
end
