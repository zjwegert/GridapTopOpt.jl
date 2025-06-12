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

function main(ranks,mesh_partition)
  path = "./results/Staggered_FCM_3d_FSI_NavierStokes/"
  i_am_main(ranks) && mkpath(path)
  D = 3

  _model = CartesianDiscreteModel(ranks,mesh_partition,(0,4,0,1,0,1),(40,10,10))
  base_model = UnstructuredDiscreteModel(_model)
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)
  # model = GmshDiscreteModel(ranks,(@__DIR__)*"/fsi/gmsh/mesh_low_res_3d.msh")

  f_Γ_Left(x) = x[3] == 0.0
  f_Γ_Right(x) = x[3] == 1.0
  f_Γ_Top(x) = x[2] == 1.0
  f_Γ_Bottom(x) = x[2] == 0.0
  f_Γ_TopCorners(x) = x[2] == 1.0 && (x[3] == 0.0 || x[3] == 1.0)
  f_Γ_D(x) = x[1] == 0.0

  update_labels!(1,model,f_Γ_Left,"Gamma_Left")
  update_labels!(2,model,f_Γ_Right,"Gamma_Right")
  update_labels!(3,model,f_Γ_Top,"Gamma_Top")
  update_labels!(4,model,f_Γ_Bottom,"Gamma_Bottom")
  update_labels!(5,model,f_Γ_TopCorners,"Gamma_TopCorners")
  update_labels!(6,model,f_Γ_D,"Gamma_f_D")
  writevtk(model,path*"model")

  Ω_act = Triangulation(model)
  hₕ = get_element_diameter_field(model)

  order = 1
  degree = 2order
  dΩ_act = Measure(Ω_act,degree)
  # Setup spaces
  uin(x) = VectorValue(x[2],x[2],0.0)

  reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
  reffe_p = ReferenceFE(lagrangian,Float64,order,space=:P)

  # Test spaces
  V = TestFESpace(Ω_act,reffe_u,conformity=:H1,
    dirichlet_tags=["Gamma_f_D","Gamma_Bottom","Gamma_Top",
      "Gamma_Left","Gamma_Right","Gamma_TopCorners"],
    dirichlet_masks=[(true,true,true),(true,true,true),
      (false,true,false),(false,false,true),(false,false,true),(false,true,true)])
  Q = TestFESpace(Ω_act,reffe_p,conformity=:H1)

  # Trial spaces
  U = TrialFESpace(V,[uin,[VectorValue(0.0,0.0,0.0) for _ = 1:5]...])
  P = TrialFESpace(Q)

  # Multifield spaces
  UP = MultiFieldFESpace([U,P])
  VQ = MultiFieldFESpace([V,Q])

  ### Weak form
  # Properties
  Re = 60 # Reynolds number
  ρ = 1.0 # Density
  cl = 1.0 # Characteristic length
  u0_max = 1.0
  μ = ρ*cl*u0_max/Re # Viscosity
  ν = μ/ρ # Kinematic viscosity

  # Stabilization parameters
  α_PSPG = 1/3
  τ_PSPG(h,u) = α_PSPG*((2norm(u)/h)^2 + 9*(4ν/h^2)^2)^-0.5 # (Eqn. 31, Peterson et al., 2018)

  # Terms
  a_Ω(u,v) = μ*(∇(u) ⊙ ∇(v))
  b_Ω(v,p) = - (∇ ⋅ v)*p
  c_Ω(p,q,u) = (τ_PSPG ∘ (hₕ,u))*1/ρ*∇(p) ⋅ ∇(q)

  a_fluid((u,p),(v,q)) =
    ∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p) )dΩ_act

  a_PSPG((u,p),(v,q)) = ∫( -c_Ω(p,q,u) )dΩ_act
  jac_PSPG((u,p),(du,dp),(v,q)) = ∫( -c_Ω(dp,q,u) )dΩ_act

  conv(u,∇u) = ρ*(∇u') ⋅ u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v) = ∫( v ⋅ (conv∘(u,∇(u))) )dΩ_act
  dc(u,du,v) = ∫( v ⋅ (dconv∘(du,∇(du),u,∇(u))) )dΩ_act

  res_fluid((u,p),(v,q)) = a_fluid((u,p),(v,q)) + a_PSPG((u,p),(v,q)) + c(u,v)
  jac_fluid((u,p),(du,dp),(v,q)) = a_fluid((du,dp),(v,q)) + jac_PSPG((u,p),(du,dp),(v,q)) + dc(u,du,v)

  fluid_ls = MUMPSSolver()
  fluid_nls = NewtonSolver(fluid_ls;maxiter=10,rtol=1.e-8,verbose=i_am_main(ranks))

  _op = FEOperator(res_fluid,jac_fluid,UP,VQ)
  uh,ph = solve(fluid_nls,_op)
  writevtk(get_triangulation(model),path*"fluid",cellfields=["uh"=>uh,"ph"=>ph])
  nothing
end

with_mpi() do distribute
  mesh_partition = (2,2,2)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  petsc_options = "-ksp_monitor -ksp_error_if_not_converged true"
  GridapPETSc.with(;args=split(petsc_options)) do
    main(ranks,mesh_partition)
  end
end