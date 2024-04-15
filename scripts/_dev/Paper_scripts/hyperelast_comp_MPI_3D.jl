using LevelSetTopOpt, Gridap, GridapDistributed, GridapPETSc, PartitionedArrays, 
  SparseMatricesCSR

using GridapSolvers: NewtonSolver

function main(mesh_partition,distribute)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  # FE parameters
  order = 1                                       # Finite element order
  xmax,ymax,zmax = (2.0,1.0,1.0)                  # Domain size
  dom = (0,xmax,0,ymax,0,zmax)                    # Bounding domain
  el_size = (160,80,80)                           # Mesh partition size
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
  iter_mod = 10                                   # VTK Output modulo
  path = "./results/hyperelast_comp_MPI_3D/"      # Output path
  i_am_main(ranks) && mkpath(path)                # Create path
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
  res(u,v,φ,dΩ,dΓ_N) = ∫( (I ∘ φ)*((dE ∘ (∇(v),∇(u))) ⊙ (S ∘ ∇(u))) )*dΩ - ∫(g⋅v)dΓ_N
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  lin_solver = ElasticitySolver(V)
  nl_solver = NewtonSolver(lin_solver;maxiter=50,rtol=10^-8,verbose=i_am_main(ranks))
  state_map = NonlinearFEStateMap(
    res,U,V,V_φ,U_reg,φh,dΩ,dΓ_N;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
    nls = nl_solver, adjoint_ls = lin_solver
  )
  # Objective and constraints
  J(u,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*((dE ∘ (∇(u),∇(u))) ⊙ (S ∘ ∇(u))))dΩ
  vol_D = sum(∫(1)dΩ)
  C1(u,φ,dΩ,dΓ_N) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
  dC1(q,u,φ,dΩ,dΓ_N) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
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
  optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;γ,γ_reinit,
    verbose=i_am_main(ranks))
  # Solve
  for (it,uh,φh) in optimiser
    data = ["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh]
    iszero(it % iter_mod) && writevtk(Ω,path*"out$it",cellfields=data)
    write_history(path*"/history.txt",get_history(optimiser);ranks)
  end
  # Final structure
  it = get_history(optimiser).niter; uh = get_state(pcfs)
  writevtk(Ω,path*"out$it",cellfields=["φ"=>φh,
    "H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
end

with_mpi() do distribute
  mesh_partition = (5,5,5)
  solver_options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
    -ksp_converged_reason -ksp_rtol 1.0e-12"
  GridapPETSc.with(args=split(solver_options)) do
    main(mesh_partition,distribute)
  end
end