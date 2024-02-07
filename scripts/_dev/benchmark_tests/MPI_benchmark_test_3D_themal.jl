using Gridap, Gridap.MultiField, GridapDistributed, GridapPETSc, GridapSolvers, 
  PartitionedArrays, LevelSetTopOpt, SparseMatricesCSR

"""
  (MPI) Minimum thermal compliance with Lagrangian method in 3D.

  Optimisation problem:
      Min J(Ω) = ∫ D*∇(u)⋅∇(u) + ξ dΩ
      Ω
    s.t., ⎡u∈V=H¹(Ω;u(Γ_D)=0),
          ⎣∫ D*∇(u)⋅∇(v) dΩ = ∫ v dΓ_N, ∀v∈V.
"""
function main(mesh_partition,distribute,el_size)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  ## Parameters
  order = 1
  xmax=ymax=zmax=1.0
  prop_Γ_N = 0.4
  prop_Γ_D = 0.2
  dom = (0,xmax,0,ymax,0,zmax)
  γ = 0.1
  γ_reinit = 0.5
  max_steps = floor(Int,minimum(el_size)/3)
  tol = 1/(order^2*10)*prod(inv,minimum(el_size))
  D = 1
  η_coeff = 2
  α_coeff = 4
  path = dirname(dirname(@__DIR__))*"/results/MPI_main_3d"

  ## FE Setup
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
  el_size = get_el_size(model)
  f_Γ_D(x) = (x[1] ≈ 0.0) && (x[2] <= ymax*prop_Γ_D + eps() || x[2] >= ymax-ymax*prop_Γ_D - eps()) &&
    (x[3] <= zmax*prop_Γ_D + eps() || x[3] >= zmax-zmax*prop_Γ_D - eps())
  f_Γ_N(x) = (x[1] ≈ xmax) && (ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= ymax/2+ymax*prop_Γ_N/4 + eps()) &&
    (zmax/2-zmax*prop_Γ_N/4 - eps() <= x[3] <= zmax/2+zmax*prop_Γ_N/4 + eps())
  update_labels!(1,model,f_Γ_D,"Gamma_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")

  ## Triangulations and measures
  Ω = Triangulation(model)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΩ = Measure(Ω,2order)
  dΓ_N = Measure(Γ_N,2order)

  ## Spaces
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,0.0)
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)

  ## Create FE functions
  φh = interpolate(initial_lsf(4,0.2),V_φ);

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_size))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  a(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*D*∇(u)⋅∇(v))dΩ
  l(v,φ,dΩ,dΓ_N) = ∫(v)dΓ_N

  ## Optimisation functionals
  ξ = 0.1
  J(u,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*D*∇(u)⋅∇(u) + ξ*(ρ ∘ φ))dΩ
  dJ(q,u,φ,dΩ,dΓ_N) = ∫((ξ-D*∇(u)⋅∇(u))*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  stencil = AdvectionStencil(FirstOrderStencil(3,Float64),model,V_φ,tol,max_steps)
  reinit!(stencil,φh,γ_reinit)

  ## Setup solver and FE operators
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  solver = PETScLinearSolver()
  
  state_map = AffineFEStateMap(
    a,l,U,V,V_φ,U_reg,φh,dΩ,dΓ_N;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
    ls = solver,adjoint_ls = solver
  )
  pcfs = PDEConstrainedFunctionals(J,state_map,analytic_dJ=dJ)

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_size)
  a_hilb(p,q) = ∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(
    a_hilb, U_reg, V_reg;
    assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls = solver
  )

  ## Optimiser
  make_dir(path;ranks=ranks)
  return AugmentedLagrangian(pcfs,stencil,vel_ext,φh;γ,γ_reinit)
end

with_mpi() do distribute
  mesh_partition = (3,3,2)
  el_size = (100,100,100)
  all_solver_options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
    -ksp_converged_reason -ksp_rtol 1.0e-12"

  GridapPETSc.with(args=split(all_solver_options)) do
    opt = main(mesh_partition,distribute,el_size)
    ## Benchmark optimiser
    bopt = benchmark_optimizer(opt, 1, nothing)
    ## Benchmark forward problem
    bfwd = benchmark_forward_problem(opt.problem.state_map, opt.φ0, nothing)
    ## Benchmark advection
    v = get_free_dof_values(interpolate(FEFunction(LevelSetTopOpt.get_deriv_space(opt.problem.state_map),
      opt.problem.dJ),LevelSetTopOpt.get_aux_space(opt.problem.state_map)))
    badv = benchmark_advection(opt.stencil, get_free_dof_values(opt.φ0), v, 0.1, nothing)
    ## Benchmark reinitialisation
    brinit = benchmark_reinitialisation(opt.stencil, get_free_dof_values(opt.φ0), 0.1, nothing)
    ## Benchmark velocity extension
    bvelext = benchmark_velocity_extension(opt.vel_ext, opt.problem.dJ, nothing)
    ## Printing
    ranks = distribute(LinearIndices((prod(mesh_partition),)))
    if i_am_main(ranks)
      println("bopt => $bopt")
      println("bfwd => $bfwd")
      println("badv => $badv")
      println("brinit => $brinit")
      println("bvelext => $bvelext")
    end
    nothing
  end
end;

