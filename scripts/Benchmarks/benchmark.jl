using Gridap, Gridap.MultiField, GridapDistributed, GridapPETSc, GridapSolvers, 
  PartitionedArrays, LSTO_Distributed, SparseMatricesCSR

using LSTO_Distributed: get_deriv_space, get_aux_space
using GridapSolvers: NewtonSolver

global NAME = ARGS[1]
global WRITE_DIR = ARGS[2]
global PROB_TYPE = ARGS[3]
global BMARK_TYPE = ARGS[4]
global N = parse(Int,ARGS[5])
global N_EL = parse(Int,ARGS[6])
global ORDER = parse(Int,ARGS[7])
global VERBOSE = parse(Int,ARGS[8])
global NREPS = parse(Int,ARGS[9])

function nl_elast(mesh_partition,ranks,el_size,order,verbose)
  ## Parameters
  xmax=ymax=zmax=1.0
  prop_Γ_N = 0.4
  dom = (0,xmax,0,ymax,0,zmax)
  γ = 0.05
  γ_reinit = 0.5
  max_steps = floor(Int,minimum(el_size)/3)
  tol = 1/(2order^2)*prod(inv,minimum(el_size))
  vf = 0.5
  η_coeff = 2
  α_coeff = 4

  ## FE Setup
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size)
  Δ = get_Δ(model)
  f_Γ_D(x) = (x[1] ≈ 0.0)
  f_Γ_N(x) = (x[1] ≈ xmax) && (ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= ymax/2+ymax*prop_Γ_N/4 + eps()) &&
      (zmax/2-zmax*prop_Γ_N/4 - eps() <= x[3] <= zmax/2+zmax*prop_Γ_N/4 + eps())
  update_labels!(1,model,f_Γ_D,"Gamma_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")

  ## Triangulations and measures
  Ω = Triangulation(model)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΩ = Measure(Ω,2order)
  dΓ_N = Measure(Γ_N,2order)
  vol_D = sum(∫(1)dΩ)

  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,VectorValue(0.0,0.0,0.0))
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)

  ## Create FE functions
  φh = interpolate(gen_lsf(4,0.2),V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  ## Material properties
  _E = 1000
  nu = 0.3
  μ, λ = _E/(2*(1 + nu)), _E*nu/((1 + nu)*(1 - 2*nu))
  g = VectorValue(0,0,-100)

  ## Neohookean hyperelastic material
  # Deformation gradient
  F(∇u) = one(∇u) + ∇u'
  J(F) = sqrt(det(C(F)))

  # Derivative of green Strain
  dE(∇du,∇u) = 0.5*( ∇du⋅F(∇u) + (∇du⋅F(∇u))' )

  # Right Caughy-green deformation tensor
  C(F) = (F')⋅F

  # Constitutive law (Neo hookean)
  function S(∇u)
    Cinv = inv(C(F(∇u)))
    μ*(one(∇u)-Cinv) + λ*log(J(F(∇u)))*Cinv
  end

  # Cauchy stress tensor and residual
  σ(∇u) = (1.0/J(F(∇u)))*F(∇u)⋅S(∇u)⋅(F(∇u))'
  res(u,v,φ,dΩ,dΓ_N) = ∫( (I ∘ φ)*((dE∘(∇(v),∇(u))) ⊙ (S∘∇(u))) )*dΩ - ∫(g⋅v)dΓ_N

  ## Optimisation functionals
  Obj(u,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*((dE∘(∇(u),∇(u))) ⊙ (S∘∇(u))))dΩ
  Vol(u,φ,dΩ,dΓ_N) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
  dVol(q,u,φ,dΩ,dΓ_N) = ∫(1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver
  stencil = AdvectionStencil(FirstOrderStencil(3,Float64),model,V_φ,tol,max_steps)

  ## Setup solver and FE operators
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  lin_solver = ElasticitySolver(V)
  nl_solver = NewtonSolver(lin_solver;maxiter=50,rtol=10^-8,verbose=verbose)

  state_map = NonlinearFEStateMap(
    res,U,V,V_φ,U_reg,φh,dΩ,dΓ_N;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
    nls = nl_solver, adjoint_ls = lin_solver
  )
  pcfs = PDEConstrainedFunctionals(Obj,[Vol],state_map,analytic_dC=[dVol])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(Δ)
  a_hilb(p,q) = ∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(
    a_hilb,U_reg,V_reg;
    assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls = PETScLinearSolver()
  )
  
  ## Optimiser
  return AugmentedLagrangian(pcfs,stencil,vel_ext,φh;γ,γ_reinit,verbose=verbose)
end

function therm(mesh_partition,ranks,el_size,order,verbose)
  ## Parameters
  xmax=ymax=zmax=1.0
  prop_Γ_N = 0.4
  prop_Γ_D = 0.2
  dom = (0,xmax,0,ymax,0,zmax)
  γ = 0.05
  γ_reinit = 0.5
  max_steps = floor(Int,minimum(el_size)/3)
  tol = 1/(2order^2)*prod(inv,minimum(el_size))
  D = 1
  η_coeff = 2
  α_coeff = 4

  ## FE Setup
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
  Δ = get_Δ(model)
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
  φh = interpolate(gen_lsf(4,0.2),V_φ);

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  a(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*D*∇(u)⋅∇(v))dΩ
  l(v,φ,dΩ,dΓ_N) = ∫(v)dΓ_N

  ## Optimisation functionals
  ξ = 0.1
  J(u,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*D*∇(u)⋅∇(u) + ξ*(ρ ∘ φ))dΩ
  dJ(q,u,φ,dΩ,dΓ_N) = ∫((ξ-D*∇(u)⋅∇(u))*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver
  stencil = AdvectionStencil(FirstOrderStencil(3,Float64),model,V_φ,tol,max_steps)

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
  α = α_coeff*maximum(Δ)
  a_hilb(p,q) = ∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(
    a_hilb, U_reg, V_reg;
    assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls = solver
  )

  ## Optimiser
  return AugmentedLagrangian(pcfs,stencil,vel_ext,φh;γ,γ_reinit,verbose=verbose)
end

function elast(mesh_partition,ranks,el_size,order,verbose)
  ## Parameters
  xmax=ymax=zmax=1.0
  prop_Γ_N = 0.4
  dom = (0,xmax,0,ymax,0,zmax)
  γ = 0.05
  γ_reinit = 0.5
  max_steps = floor(Int,minimum(el_size)/3)
  tol = 1/(2order^2)*prod(inv,minimum(el_size))
  C = isotropic_3d(1.,0.3)
  g = VectorValue(0,0,-1)
  η_coeff = 2
  α_coeff = 4
  vf = 0.5

  ## FE Setup
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
  Δ = get_Δ(model)
  f_Γ_D(x) = (x[1] ≈ 0.0)
  f_Γ_N(x) = (x[1] ≈ xmax) && (ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= ymax/2+ymax*prop_Γ_N/4 + eps()) &&
      (zmax/2-zmax*prop_Γ_N/4 - eps() <= x[3] <= zmax/2+zmax*prop_Γ_N/4 + eps())
  update_labels!(1,model,f_Γ_D,"Gamma_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")

  ## Triangulations and measures
  Ω = Triangulation(model)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΩ = Measure(Ω,2*order)
  dΓ_N = Measure(Γ_N,2*order)
  vol_D = sum(∫(1)dΩ)

  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,VectorValue(0.0,0.0,0.0))
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)

  ## Create FE functions
  φh = interpolate(gen_lsf(4,0.2),V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  a(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(v)))dΩ
  l(v,φ,dΩ,dΓ_N) = ∫(v⋅g)dΓ_N

  ## Optimisation functionals
  J(u,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(u)))dΩ
  dJ(q,u,φ,dΩ,dΓ_N) = ∫(( - C ⊙ ε(u) ⊙ ε(u))*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  Vol(u,φ,dΩ,dΓ_N) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
  dVol(q,u,φ,dΩ,dΓ_N) = ∫(1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver
  stencil = AdvectionStencil(FirstOrderStencil(3,Float64),model,V_φ,tol,max_steps)

  ## Setup solver and FE operators
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  solver = ElasticitySolver(V)
  
  state_map = AffineFEStateMap(
    a,l,U,V,V_φ,U_reg,φh,dΩ,dΓ_N;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
    ls = solver, adjoint_ls = solver
  )
  pcfs = PDEConstrainedFunctionals(J,[Vol],state_map;analytic_dJ=dJ,analytic_dC=[dVol])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(Δ)
  a_hilb(p,q) = ∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(
    a_hilb,U_reg,V_reg;
    assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls = PETScLinearSolver()
  )

  ## Optimiser
  return AugmentedLagrangian(pcfs,stencil,vel_ext,φh;γ,γ_reinit,verbose=verbose)
end

function inverter_HPM(mesh_partition,ranks,el_size,order,verbose)
  ## Parameters
  dom = (0,1,0,1,0,1)
  γ = 0.05
  γ_reinit = 0.5
  max_steps = floor(Int,minimum(el_size)/3)
  tol = 1/(2order^2)*prod(inv,minimum(el_size))
  C = isotropic_3d(1.0,0.3)
  η_coeff = 2
  α_coeff = 4
  vf=0.4
  δₓ=0.5
  ks = 0.01
  g = VectorValue(1,0,0)

  ## FE Setup
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
  Δ = get_Δ(model)
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
  φh = interpolate(gen_lsf(4,0.1),V_φ);

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  a(u,v,φ,dΩ,dΓ_in,dΓ_out) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(v)))dΩ + ∫(ks*(u⋅v))dΓ_out
  l(v,φ,dΩ,dΓ_in,dΓ_out) = ∫(v⋅g)dΓ_in

  ## Optimisation functionals
  e₁ = VectorValue(1,0,0)
  J(u,φ,dΩ,dΓ_in,dΓ_out) = ∫((u⋅e₁)/vol_Γ_in)dΓ_in
  Vol(u,φ,dΩ,dΓ_in,dΓ_out) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ;
  dVol(q,u,φ,dΩ,dΓ_in,dΓ_out) = ∫(1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  UΓ_out(u,φ,dΩ,dΓ_in,dΓ_out) = ∫((u⋅-e₁-δₓ)/vol_Γ_out)dΓ_out

  ## Finite difference solver
  stencil = AdvectionStencil(FirstOrderStencil(3,Float64),model,V_φ,tol,max_steps)

  ## Setup solver and FE operators
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  solver = ElasticitySolver(V)
  
  state_map = AffineFEStateMap(
    a,l,U,V,V_φ,U_reg,φh,dΩ,dΓ_in,dΓ_out;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
    ls = solver, adjoint_ls = solver
  )
  pcfs = PDEConstrainedFunctionals(J,[Vol,UΓ_out],state_map,analytic_dC=[dVol,nothing])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(
    a_hilb,U_reg,V_reg;
    assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls = PETScLinearSolver()
  )
  
  ## Optimiser
  return HilbertianProjection(pcfs,stencil,vel_ext,φh;γ,γ_reinit,verbose=verbose)
end

with_mpi() do distribute
  # Setup
  mesh_partition = (N,N,N)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  el_size = (N_EL,N_EL,N_EL)
  verbose = Bool(VERBOSE) ? i_am_main(ranks) : false; 
  if PROB_TYPE == "NLELAST"
    options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
      -ksp_converged_reason -ksp_rtol 1.0e-12 -mat_block_size 3
      -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_coarse_sub_pc_type cholesky"
    opt = nl_elast
  elseif PROB_TYPE == "THERM" 
    options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
      -ksp_converged_reason -ksp_rtol 1.0e-12"
    opt = therm
  elseif PROB_TYPE == "ELAST"
    options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
      -ksp_converged_reason -ksp_rtol 1.0e-12 -mat_block_size 3
      -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_coarse_sub_pc_type cholesky"
    opt = elast
  elseif PROB_TYPE == "INVERTER_HPM"
    options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
      -ksp_converged_reason -ksp_rtol 1.0e-12 -mat_block_size 3
      -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_coarse_sub_pc_type cholesky"
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
      bopt1 = benchmark_optimizer(optim, 1, ranks; nreps=NREPS)
    else
      bopt1 = i_am_main(ranks) && zeros(NREPS)
    end
    ## Benchmark reinitialisation
    if occursin("breinit",BMARK_TYPE)
      breinit = benchmark_reinitialisation(optim.stencil, get_free_dof_values(optim.φ0), 0.1, ranks; nreps=NREPS)
    else
      breinit = i_am_main(ranks) && zeros(NREPS)
    end
    ## Benchmark forward problem
    reinit!(optim.stencil,optim.φ0,optim.params.γ_reinit)
    if occursin("bfwd",BMARK_TYPE)
      bfwd = benchmark_forward_problem(optim.problem.state_map, optim.φ0, ranks; nreps=NREPS)
    else
      bfwd = i_am_main(ranks) && zeros(NREPS)
    end
    ## Benchmark advection
    if occursin("badv",BMARK_TYPE)
      vh = interpolate(FEFunction(get_deriv_space(optim.problem.state_map),optim.problem.dJ),
        get_aux_space(optim.problem.state_map))
      v = get_free_dof_values(vh)
      badv = benchmark_advection(optim.stencil, get_free_dof_values(optim.φ0), v, 0.1, ranks; nreps=NREPS)
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