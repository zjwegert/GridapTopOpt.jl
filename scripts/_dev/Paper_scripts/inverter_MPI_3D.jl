using LevelSetTopOpt, Gridap, GridapDistributed, GridapPETSc, PartitionedArrays, 
  SparseMatricesCSR

function main(mesh_partition,distribute)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  # FE parameters
  order = 1                                       # Finite element order
  xmax,ymax,zmax = (1.0,1.0,1.0)                  # Domain size
  dom = (0,xmax,0,ymax,0,zmax)                    # Bounding domain
  el_size = (100,100,100)                         # Mesh partition size
  f_Γ_in(x) = (x[1] ≈ 0.0) &&                     # Γ_in indicator function
    (0.4 - eps() <= x[2] <= 0.6 + eps()) && (0.4 - eps() <= x[3] <= 0.6 + eps())
  f_Γ_out(x) = (x[1] ≈ 1.0) &&                    # Γ_out indicator function
    (0.4 - eps() <= x[2] <= 0.6 + eps()) && (0.4 - eps() <= x[3] <= 0.6 + eps())
  f_Γ_D(x) = (x[1] ≈ 0.0)  &&                     # Γ_D indicator function
    (x[2] <= 0.1 || x[2] >= 0.9) && (x[3] <= 0.1 || x[3] >= 0.9)
  # FD parameters
  γ = 0.1                                         # HJ eqn time step coeff
  γ_reinit = 0.5                                  # Reinit. eqn time step coeff
  max_steps = floor(Int,order*minimum(el_size)/5) # Max steps for advection
  tol = 1/(5order^2)/minimum(el_size)             # Reinitialisation tolerance
  # Problem parameters
  C = isotropic_elast_tensor(3,1.,0.3)            # Stiffness tensor
  g = VectorValue(0,0,-1)                         # Applied load on Γ_N
  vf = 0.4                                        # Volume fraction constraint
  sphere(x,(xc,yc,zc)) = -sqrt((x[1]-xc)^2+(x[2]-yc)^2+(x[3]-zc)^2) + 0.2
  lsf_func(x) = max(initial_lsf(4,0.2)(x),        # Initial level set function
    sphere(x,(1,0,0)),sphere(x,(1,0,1)),sphere(x,(1,1,0)),sphere(x,(1,1,1)))
  lsf_func = initial_lsf(4,0.2)                   # Initial level set function
  iter_mod = 10                                   # VTK Output modulo
  path = "./results/inverter_MPI_3D/"             # Output path
  i_am_main(ranks) && mkpath(path)                # Create path
  # Model
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
  update_labels!(1,model,f_Γ_in,"Gamma_in")
  update_labels!(2,model,f_Γ_out,"Gamma_out")
  update_labels!(4,model,f_Γ_D,"Gamma_D")
  # Triangulation and measures
  Ω = Triangulation(model)
  Γ_in = BoundaryTriangulation(model,tags="Gamma_in")
  Γ_out = BoundaryTriangulation(model,tags="Gamma_out")
  dΩ = Measure(Ω,2*order)
  dΓ_in = Measure(Γ_in,2*order)
  dΓ_out = Measure(Γ_out,2*order)
  # Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,VectorValue(0.0,0.0,0.0))
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_in","Gamma_out"])
  U_reg = TrialFESpace(V_reg,0)
  # Level set and interpolator
  φh = interpolate(lsf_func,V_φ);
  interp = SmoothErsatzMaterialInterpolation(η = 2*maximum(get_el_Δ(model)))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ
  # Weak formulation
  a(u,v,φ,dΩ,dΓ_in,dΓ_out) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(v)))dΩ + ∫(ks*(u⋅v))dΓ_out
  l(v,φ,dΩ,dΓ_in,dΓ_out) = ∫(v⋅g)dΓ_in
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  solver = ElasticitySolver(V)
  state_map = AffineFEStateMap(
    a,l,U,V,V_φ,U_reg,φh,dΩ,dΓ_in,dΓ_out;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
    ls = solver,adjoint_ls = solver
  )
  # Objective and constraints
  e₁ = VectorValue(1,0,0)
  vol_Γ_in = sum(∫(1)dΓ_in)
  vol_Γ_out = sum(∫(1)dΓ_out)
  vol_D = sum(∫(1)dΩ)
  J(u,φ,dΩ,dΓ_in,dΓ_out) = ∫((u⋅e₁)/vol_Γ_in)dΓ_in
  C1(u,φ,dΩ,dΓ_in,dΓ_out) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
  dC1(q,u,φ,dΩ,dΓ_in,dΓ_out) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  C2(u,φ,dΩ,dΓ_in,dΓ_out) = ∫((u⋅-e₁-δₓ)/vol_Γ_out)dΓ_out
  pcfs = PDEConstrainedFunctionals(J,[C1,C2],state_map,
    analytic_dJ=dJ,analytic_dC=[dC1,nothing])
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
  stencil = AdvectionStencil(scheme,model,V_φ,tol,max_steps)
  # Optimiser
  optimiser = AugmentedLagrangian(pcfs,stencil,vel_ext,φh;γ,γ_reinit,
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