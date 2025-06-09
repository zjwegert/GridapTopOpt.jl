using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapPETSc

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


function main(ranks,mesh_partition,n,solver)
  path = "./results/Testing PETSc with empty parts/"
  i_am_main(ranks) && mkpath(path)

  domain = (0,1,0,1)
  cell_partition = (n,n)
  base_model = UnstructuredDiscreteModel(CartesianDiscreteModel(ranks,mesh_partition,domain,cell_partition))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)
  writevtk(model,path*"model")

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)

  lsf(x) = sqrt(x[1]^2+(x[2]-0.5)^2)-0.3
  φh = interpolate(lsf,V_φ)
  writevtk(get_triangulation(φh),path*"initial_lsf",cellfields=["φh"=>φh])

  # Cut
  order = 1
  degree = 2*(order+1)

  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)

  Ω = Triangulation(cutgeo,PHYSICAL)
  Ω_act = Triangulation(cutgeo,ACTIVE)
  dΩ = Measure(Ω,degree)

  Γ  = EmbeddedBoundary(cutgeo)
  dΓ  = Measure(Γ,degree)

  Γg = GhostSkeleton(cutgeo)
  n_Γg = get_normal_vector(Γg)
  dΓg = Measure(Γg,degree)

  writevtk(Ω,path*"omega")

  # Weak form

  γg = 0.1
  h = 1/n
  g = 1
  a(u,v) =
    ∫( ∇(v)⋅∇(u) )dΩ +
    ∫( (γg*h)*(jump(n_Γg⋅∇(v))⋅jump(n_Γg⋅∇(u))) )dΓg

  l(v) = ∫( g*v )dΓ

  reffe = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(Ω_act,reffe,conformity=:H1,dirichlet_tags=["boundary",])
  U = TrialFESpace(V)

  op = AffineFEOperator(a,l,U,V)

  uh = solve(solver(),op)
  writevtk(Ω,path*"omega_sol_$solver",cellfields=["uh"=>uh])
  return uh
end

with_mpi() do distribute
  mesh_partition = (2,2)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  petsc_options = "-ksp_converged_reason -ksp_error_if_not_converged true"
  GridapPETSc.with(;args=split(petsc_options)) do
    uh1 = main(ranks,mesh_partition,30,LUSolver)
    uh2 = main(ranks,mesh_partition,30,MUMPSSolver)
    uh3 = main(ranks,mesh_partition,30,CGAMGSolver)
    i_am_main(ranks) && println("norm(uh_LU - uh_MUMPS,Inf) = ",maximum(abs,get_free_dof_values(uh1) - get_free_dof_values(uh2)))
    i_am_main(ranks) && println("norm(uh_LU - uh_CGAMG,Inf) = ",maximum(abs,get_free_dof_values(uh1) - get_free_dof_values(uh3)))
  end
end
