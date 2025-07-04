using Gridap, Gridap.MultiField, GridapDistributed, GridapPETSc, GridapSolvers,
  PartitionedArrays, GridapTopOpt, SparseMatricesCSR

using GridapTopOpt: get_deriv_space, get_aux_space,benchmark_optimizer,
  benchmark_forward_problem,benchmark_advection,benchmark_reinitialisation,
  benchmark_velocity_extension,benchmark_hilbertian_projection_map,benchmark_single_iteration

using GridapSolvers: NewtonSolver

global NAME = ARGS[1]
global WRITE_DIR = ARGS[2]
global PROB_TYPE = ARGS[3]
global BMARK_TYPE = ARGS[4]
global Nx = parse(Int,ARGS[5])
global Ny = parse(Int,ARGS[6])
global Nz = parse(Int,ARGS[7])
global N_EL = parse(Int,ARGS[8])
global ORDER = parse(Int,ARGS[9])
global VERBOSE = parse(Int,ARGS[10])
global NREPS = parse(Int,ARGS[11])

function nl_elast(mesh_partition,ranks,el_size,order,verbose)
  # FE parameters
  xmax,ymax,zmax = (1.0,1.0,1.0)                  # Domain size
  dom = (0,xmax,0,ymax,0,zmax)                    # Bounding domain
  prop_Γ_N = 0.2                                  # Γ_N size parameter
  f_Γ_N(x) = (x[1] ≈ xmax) &&                     # Γ_N indicator function
    (ymax/2-ymax*prop_Γ_N/2 - eps() <= x[2] <= ymax/2+ymax*prop_Γ_N/2 + eps())&&
    (zmax/2-zmax*prop_Γ_N/2 - eps() <= x[3] <= zmax/2+zmax*prop_Γ_N/2 + eps())
  f_Γ_D(x) = (x[1] ≈ 0.0)                         # Γ_D indicator function
  # FD parameters
  γ = 0.1                                         # HJ eqn time step coeff
  γ_reinit = 0.5                                  # Reinit. eqn time step coeff
  max_steps = floor(Int,order*minimum(el_size)/5) # Max steps for advection
  tol = 1/(5order^2)/minimum(el_size)             # Reinitialisation tolerance
  # Problem parameters
  E = 1000                                        # Young's modulus
  ν = 0.3                                         # Poisson's ratio
  μ, λ = E/(2*(1 + ν)), E*ν/((1 + ν)*(1 - 2*ν))   # Lame constants
  g = VectorValue(0,0,-1)                         # Applied load on Γ_N
  vf = 0.4                                        # Volume fraction constraint
  lsf_func = initial_lsf(4,0.2)                   # Initial level set function
  # Model
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
  update_labels!(1,model,f_Γ_D,"Gamma_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")
  # Triangulation and measures
  Ω = Triangulation(model)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΩ = Measure(Ω,2*order)
  dΓ_N = Measure(Γ_N,2*order)
  # Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,VectorValue(0.0,0.0,0.0))
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)
  # Level set and interpolator
  φh = interpolate(lsf_func,V_φ)
  interp = SmoothErsatzMaterialInterpolation(η = 2*maximum(get_el_Δ(model)))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ
  # Weak formulation
  ## Piola-Kirchhoff tensor
  function S(∇u)
    Cinv = inv(C(F(∇u)))
    μ*(one(∇u)-Cinv) + λ*log(J(F(∇u)))*Cinv
  end
  ## Derivative of Green strain
  dE(∇du,∇u) = 0.5*( ∇du⋅F(∇u) + (∇du⋅F(∇u))' )
  ## Right Cauchy-Green deformation tensor
  C(F) = (F')⋅F
  ## Deformation gradient tensor
  F(∇u) = one(∇u) + ∇u'
  ## Volume change
  J(F) = sqrt(det(C(F)))
  ## Residual
  res(u,v,φ) = ∫( (I ∘ φ)*((dE ∘ (∇(v),∇(u))) ⊙ (S ∘ ∇(u))) )*dΩ - ∫(g⋅v)dΓ_N
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  lin_solver = ElasticitySolver(V)
  nl_solver = NewtonSolver(lin_solver;maxiter=50,rtol=10^-8,verbose)
  state_map = NonlinearFEStateMap(
    res,U,V,V_φ,φh;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,V_φ,V_φ),
    nls = nl_solver, adjoint_ls = lin_solver
  )
  # Objective and constraints
  J(u,φ) = ∫((I ∘ φ)*((dE ∘ (∇(u),∇(u))) ⊙ (S ∘ ∇(u))))dΩ
  vol_D = sum(∫(1)dΩ)
  C1(u,φ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
  dC1(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  pcfs = PDEConstrainedFunctionals(J,[C1],state_map,analytic_dC=[dC1])
  # Velocity extension
  α = 4max_steps*γ*maximum(get_el_Δ(model))
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ
  vel_ext = VelocityExtension(
    a_hilb, U_reg, V_reg;
    assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls = PETScLinearSolver()
  )
  # Finite difference scheme
  scheme = FirstOrderStencil(length(el_size),Float64)
  ls_evo = HamiltonJacobiEvolution(scheme,model,V_φ,tol,max_steps)
  # Optimiser
  return AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;γ,γ_reinit,verbose)
end

function therm(mesh_partition,ranks,el_size,order,verbose)
  # FE parameters
  xmax,ymax,zmax = (1.0,1.0,1.0)                  # Domain size
  dom = (0,xmax,0,ymax,0,zmax)                    # Bounding domain
  prop_Γ_N = 0.2                                  # Γ_N size parameter
  prop_Γ_D = 0.2                                  # Γ_D size parameter
  f_Γ_N(x) = (x[1] ≈ xmax) &&                     # Γ_N indicator function
    (ymax/2-ymax*prop_Γ_N/2 - eps() <= x[2] <= ymax/2+ymax*prop_Γ_N/2 + eps())&&
    (zmax/2-zmax*prop_Γ_N/2 - eps() <= x[3] <= zmax/2+zmax*prop_Γ_N/2 + eps())
  f_Γ_D(x) = (x[1] ≈ 0.0) &&                      # Γ_D indicator function
    (x[2] <= ymax*prop_Γ_D + eps() || x[2] >= ymax-ymax*prop_Γ_D - eps()) &&
    (x[3] <= zmax*prop_Γ_D + eps() || x[3] >= zmax-zmax*prop_Γ_D - eps())
  # FD parameters
  γ = 0.1                                         # HJ eqn time step coeff
  γ_reinit = 0.5                                  # Reinit. eqn time step coeff
  max_steps = floor(Int,order*minimum(el_size)/5) # Max steps for advection
  tol = 1/(5order^2)/minimum(el_size)             # Reinitialisation tolerance
  # Problem parameters
  κ = 1                                           # Diffusivity
  g = 1                                           # Heat flow in
  vf = 0.4                                        # Volume fraction constraint
  lsf_func = initial_lsf(4,0.2)                   # Initial level set function
  # Model
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
  update_labels!(1,model,f_Γ_D,"Gamma_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")
  # Triangulation and measures
  Ω = Triangulation(model)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΩ = Measure(Ω,2*order)
  dΓ_N = Measure(Γ_N,2*order)
  # Spaces
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,0.0)
  V_φ = TestFESpace(model,reffe)
  V_reg = TestFESpace(model,reffe;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)
  # Level set and interpolator
  φh = interpolate(lsf_func,V_φ)
  interp = SmoothErsatzMaterialInterpolation(η = 2*maximum(get_el_Δ(model)))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ
  # Weak formulation
  a(u,v,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(v))dΩ
  l(v,φ) = ∫(g*v)dΓ_N
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  solver = PETScLinearSolver()
  state_map = AffineFEStateMap(
    a,l,U,V,V_φ,φh;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,V_φ,V_φ),
    ls = solver,adjoint_ls = solver
  )
  # Objective and constraints
  J(u,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(u))dΩ
  dJ(q,u,φ) = ∫(κ*∇(u)⋅∇(u)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  vol_D = sum(∫(1)dΩ)
  C1(u,φ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
  dC1(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  pcfs = PDEConstrainedFunctionals(J,[C1],state_map,
    analytic_dJ=dJ,analytic_dC=[dC1])
  # Velocity extension
  α = 4max_steps*γ*maximum(get_el_Δ(model))
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ
  vel_ext = VelocityExtension(
    a_hilb, U_reg, V_reg;
    assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls = solver
  )
  # Finite difference scheme
  scheme = FirstOrderStencil(length(el_size),Float64)
  ls_evo = HamiltonJacobiEvolution(scheme,model,V_φ,tol,max_steps)
  # Optimiser
  return AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;γ,γ_reinit,verbose)
end

function elast(mesh_partition,ranks,el_size,order,verbose)
  # FE parameters
  xmax,ymax,zmax = (1.0,1.0,1.0)                  # Domain size
  dom = (0,xmax,0,ymax,0,zmax)                    # Bounding domain
  prop_Γ_N = 0.2                                  # Γ_N size parameter
  f_Γ_N(x) = (x[1] ≈ xmax) &&                     # Γ_N indicator function
    (ymax/2-ymax*prop_Γ_N/2 - eps() <= x[2] <= ymax/2+ymax*prop_Γ_N/2 + eps())&&
    (zmax/2-zmax*prop_Γ_N/2 - eps() <= x[3] <= zmax/2+zmax*prop_Γ_N/2 + eps())
  f_Γ_D(x) = (x[1] ≈ 0.0)                         # Γ_D indicator function
  # FD parameters
  γ = 0.1                                         # HJ eqn time step coeff
  γ_reinit = 0.5                                  # Reinit. eqn time step coeff
  max_steps = floor(Int,order*minimum(el_size)/5) # Max steps for advection
  tol = 1/(5order^2)/minimum(el_size)             # Reinitialisation tolerance
  # Problem parameters
  C = isotropic_elast_tensor(3,1.,0.3)            # Stiffness tensor
  g = VectorValue(0,0,-1)                         # Applied load on Γ_N
  vf = 0.4                                        # Volume fraction constraint
  lsf_func = initial_lsf(4,0.2)                   # Initial level set function
  # Model
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
  update_labels!(1,model,f_Γ_D,"Gamma_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")
  # Triangulation and measures
  Ω = Triangulation(model)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΩ = Measure(Ω,2*order)
  dΓ_N = Measure(Γ_N,2*order)
  # Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,VectorValue(0.0,0.0,0.0))
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)
  # Level set and interpolator
  φh = interpolate(lsf_func,V_φ)
  interp = SmoothErsatzMaterialInterpolation(η = 2*maximum(get_el_Δ(model)))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ
  # Weak formulation
  a(u,v,φ) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(v)))dΩ
  l(v,φ) = ∫(v⋅g)dΓ_N
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  solver = ElasticitySolver(V)
  state_map = AffineFEStateMap(
    a,l,U,V,V_φ,φh;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,V_φ,V_φ),
    ls = solver,adjoint_ls = solver
  )
  # Objective and constraints
  J(u,φ) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(u)))dΩ
  dJ(q,u,φ) = ∫((C ⊙ ε(u) ⊙ ε(u))*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  vol_D = sum(∫(1)dΩ)
  C1(u,φ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
  dC1(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  pcfs = PDEConstrainedFunctionals(J,[C1],state_map,
    analytic_dJ=dJ,analytic_dC=[dC1])
  # Velocity extension
  α = 4max_steps*γ*maximum(get_el_Δ(model))
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ
  vel_ext = VelocityExtension(
    a_hilb, U_reg, V_reg;
    assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls = PETScLinearSolver()
  )
  # Finite difference scheme
  scheme = FirstOrderStencil(length(el_size),Float64)
  ls_evo = HamiltonJacobiEvolution(scheme,model,V_φ,tol,max_steps)
  # Optimiser
  return AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;γ,γ_reinit,verbose)
end

function inverter_HPM(mesh_partition,ranks,el_size,order,verbose)
  ## Parameters
  dom = (0,1,0,1,0,1)
  γ = 0.05
  γ_reinit = 0.5
  max_steps = floor(Int,minimum(el_size)/3)
  tol = 1/(2order^2)/minimum(el_size)
  C = isotropic_elast_tensor(3,1.0,0.3)
  η_coeff = 2
  α_coeff = 4
  vf=0.4
  δₓ=0.5
  ks = 0.01
  g = VectorValue(1,0,0)

  ## FE Setup
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
  el_Δ = get_el_Δ(model)
  f_Γ_in(x) = (x[1] ≈ 0.0) && (0.4 - eps() <= x[2] <= 0.6 + eps()) &&
    (0.4 - eps() <= x[3] <= 0.6 + eps())
  f_Γ_out(x) = (x[1] ≈ 1.0) && (0.4 - eps() <= x[2] <= 0.6 + eps()) &&
    (0.4 - eps() <= x[3] <= 0.6 + eps())
  f_Γ_out_ext(x) = ~f_Γ_out(x) && (0.9 <= x[1] <= 1.0) && (0.3 - eps() <= x[2] <= 0.7 + eps()) &&
    (0.3 - eps() <= x[3] <= 0.7 + eps())
  f_Γ_D(x) = (x[1] ≈ 0.0)  && (x[2] <= 0.1 || x[2] >= 0.9) && (x[3] <= 0.1 || x[3] >= 0.9)
  update_labels!(1,model,f_Γ_in,"Gamma_in")
  update_labels!(2,model,f_Γ_out,"Gamma_out")
  update_labels!(3,model,f_Γ_out_ext,"Gamma_out_ext")
  update_labels!(4,model,f_Γ_D,"Gamma_D")

  ## Triangulations and measures
  Ω = Triangulation(model)
  Γ_in = BoundaryTriangulation(model,tags="Gamma_in")
  Γ_out = BoundaryTriangulation(model,tags="Gamma_out")
  dΩ = Measure(Ω,2*order)
  dΓ_in = Measure(Γ_in,2*order)
  dΓ_out = Measure(Γ_out,2*order)
  vol_D = sum(∫(1)dΩ)
  vol_Γ_in = sum(∫(1)dΓ_in)
  vol_Γ_out = sum(∫(1)dΓ_out)

  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,VectorValue(0.0,0.0,0.0))
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_in","Gamma_out","Gamma_out_ext"])
  U_reg = TrialFESpace(V_reg,[0,0,0])

  ## Create FE functions
  φh = interpolate(initial_lsf(4,0.1),V_φ);

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  a(u,v,φ) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(v)))dΩ + ∫(ks*(u⋅v))dΓ_out
  l(v,φ) = ∫(v⋅g)dΓ_in

  ## Optimisation functionals
  e₁ = VectorValue(1,0,0)
  J(u,φ) = ∫((u⋅e₁)/vol_Γ_in)dΓ_in
  Vol(u,φ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ;
  dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  UΓ_out(u,φ) = ∫((u⋅-e₁-δₓ)/vol_Γ_out)dΓ_out

  ## Finite difference solver
  ls_evo = HamiltonJacobiEvolution(FirstOrderStencil(3,Float64),model,V_φ,tol,max_steps)

  ## Setup solver and FE operators
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  solver = ElasticitySolver(V)

  state_map = AffineFEStateMap(
    a,l,U,V,V_φ,φh;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,V_φ,V_φ),
    ls = solver, adjoint_ls = solver
  )
  pcfs = PDEConstrainedFunctionals(J,[Vol,UΓ_out],state_map,analytic_dC=[dVol,nothing])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(
    a_hilb,U_reg,V_reg;
    assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls = PETScLinearSolver()
  )

  ## Optimiser
  return HilbertianProjection(pcfs,ls_evo,vel_ext,φh;γ,γ_reinit,verbose=verbose)
end

with_mpi() do distribute
  # Setup
  mesh_partition = (Nx,Ny,Nz)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  el_size = (N_EL,N_EL,N_EL)
  verbose = Bool(VERBOSE) ? i_am_main(ranks) : false;
  if PROB_TYPE == "NLELAST"
    options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true
      -ksp_converged_reason -ksp_rtol 1.0e-12"
    opt = nl_elast
  elseif PROB_TYPE == "THERM"
    options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true
      -ksp_converged_reason -ksp_rtol 1.0e-12"
    opt = therm
  elseif PROB_TYPE == "ELAST"
    options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true
      -ksp_converged_reason -ksp_rtol 1.0e-12"
    opt = elast
  elseif PROB_TYPE == "INVERTER_HPM"
    options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true
      -ksp_converged_reason -ksp_rtol 1.0e-12"
    opt = inverter_HPM
  else
    error("Problem not defined")
  end

  # Output
  i_am_main(ranks) && ~isdir(WRITE_DIR) && mkpath(WRITE_DIR)
  # Run
  t_start = PTimer(ranks);
  GridapPETSc.with(args=split(options)) do
    tic!(t_start;barrier=true)
    optim = opt(mesh_partition,ranks,el_size,ORDER,verbose)
    toc!(t_start,"startup")
    startup_time = map_main(t_start.data) do data
      map(x -> x.max,values(data))
    end |> PartitionedArrays.getany
    bstart = i_am_main(ranks) && [startup_time; zeros(NREPS-1)]
    ## Benchmark optimiser
    if occursin("bopt0",BMARK_TYPE)
      bopt0 = benchmark_optimizer(optim, 0, ranks; nreps=NREPS)
    else
      bopt0 = i_am_main(ranks) && zeros(NREPS)
    end
    if occursin("bopt1",BMARK_TYPE)
      bopt1 = benchmark_single_iteration(optim, ranks; nreps = NREPS)
    else
      bopt1 = i_am_main(ranks) && zeros(NREPS)
    end
    ## Benchmark reinitialisation
    if occursin("breinit",BMARK_TYPE)
      breinit = benchmark_reinitialisation(optim.ls_evolver, get_free_dof_values(optim.φ0), 0.1, ranks; nreps=NREPS)
    else
      breinit = i_am_main(ranks) && zeros(NREPS)
    end
    ## Benchmark forward problem
    if occursin("bfwd",BMARK_TYPE)
      bfwd = benchmark_forward_problem(optim.problem.state_map, optim.φ0, ranks; nreps=NREPS)
    else
      bfwd = i_am_main(ranks) && zeros(NREPS)
    end
    ## Benchmark advection
    reinit!(optim.ls_evolver,optim.φ0,optim.params.γ_reinit)
    if occursin("badv",BMARK_TYPE)
      vh = interpolate(FEFunction(get_deriv_space(optim.problem.state_map),optim.problem.dJ),
        get_aux_space(optim.problem.state_map))
      v = get_free_dof_values(vh)
      badv = benchmark_advection(optim.ls_evolver, get_free_dof_values(optim.φ0), v, 0.1, ranks; nreps=NREPS)
    else
      badv = i_am_main(ranks) && zeros(NREPS)
    end
    ## Benchmark velocity extension
    if occursin("bvelext",BMARK_TYPE)
      bvelext = benchmark_velocity_extension(optim.vel_ext, optim.problem.dJ, ranks; nreps=NREPS)
    else
      bvelext = i_am_main(ranks) && zeros(NREPS)
    end
    ## HPM
    if occursin("bhpm",BMARK_TYPE)
      @assert typeof(optim) <: HilbertianProjection
      J, C, dJ, dC = Gridap.evaluate!(optim.problem,optim.φ0)
      optim.projector,dJ,C,dC,optim.vel_ext.K
      bhpm = benchmark_hilbertian_projection_map(optim.projector,dJ,C,dC,optim.vel_ext.K,ranks)
    else
      bhpm = i_am_main(ranks) && zeros(NREPS)
    end
    ## Write results
    if i_am_main(ranks)
      open(WRITE_DIR*NAME*".txt","w") do f
        bcontent = "startup,bopt(0),bopt(1),bfwd,badv,breinit,bvelext,bhpm\n"
        for i = 1:NREPS
          bcontent *= "$(bstart[i]),$(bopt0[i]),$(bopt1[i]),$(bfwd[i]),$(badv[i]),$(breinit[i]),$(bvelext[i]),$(bhpm[i])\n"
        end
        write(f,bcontent)
      end
    end
  end
end