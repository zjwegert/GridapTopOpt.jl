using LevelSetTopOpt, Gridap, GridapDistributed, GridapPETSc, PartitionedArrays, 
  SparseMatricesCSR

function main(mesh_partition,distribute)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  # FE parameters
  order = 1                                       # Finite element order
  xmax,ymax,zmax = (1.0,1.0,1.0)                  # Domain size
  dom = (0,xmax,0,ymax,0,zmax)                    # Bounding domain
  el_size = (100,100,100)                         # Mesh partition size
  f_Γ_D(x) = iszero(x)                            # Γ_D indicator function
  # FD parameters
  γ = 0.1                                         # HJ eqn time step coeff
  γ_reinit = 0.5                                  # Reinit. eqn time step coeff
  max_steps = floor(Int,minimum(el_size)/10)      # Max steps for advection
  tol = 1/(5order^2)/minimum(el_size)             # Reinitialisation tolerance
  # Problem parameters
  C = isotropic_elast_tensor(3,1.,0.3)            # Stiffness tensor
  g = VectorValue(0,0,-1)                         # Applied load on Γ_N
  vf = 0.4                                        # Volume fraction constraint
  lsf_func(x) = cos(2π*x[1]) + cos(2π*x[2]) +     # Initial level set function
    cos(2π*x[3])
  iter_mod = 10                                   # VTK Output modulo
  path = "./results/inverse_hom_MPI_3D/"          # Output path
  i_am_main(ranks) && mkpath(path)                # Create path
  # Model
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size,isperiodic=(true,true,true));
  update_labels!(1,model,f_Γ_D,"origin")
  # Triangulation and measures
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*order)
  # Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["origin"])
  U = TrialFESpace(V,VectorValue(0.0,0.0,0.0))
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar)
  U_reg = TrialFESpace(V_reg)
  # Level set and interpolator
  φh = interpolate(lsf_func,V_φ)
  interp = SmoothErsatzMaterialInterpolation(η = 2*maximum(get_el_Δ(model)))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ
  # Weak formulation
  εᴹ = (TensorValue(1.,0.,0.,0.,0.,0.,0.,0.,0.),
        TensorValue(0.,0.,0.,0.,1.,0.,0.,0.,0.),
        TensorValue(0.,0.,0.,0.,0.,0.,0.,0.,1.),
        TensorValue(0.,0.,0.,0.,0.,1/2,0.,1/2,0.),
        TensorValue(0.,0.,1/2,0.,0.,0.,1/2,0.,0.),
        TensorValue(0.,1/2,0.,1/2,0.,0.,0.,0.,0.))
  a(u,v,φ,dΩ) = ∫((I ∘ φ) * C ⊙ ε(u) ⊙ ε(v))dΩ
  l = [(v,φ,dΩ) -> ∫(-(I ∘ φ) * C ⊙ εᴹ[i] ⊙ ε(v))dΩ for i in 1:6]
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  solver = ElasticitySolver(V)
  state_map = RepeatingAffineFEStateMap(
    6,a,l,U,V,V_φ,U_reg,φh,dΩ;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
    ls = solver, adjoint_ls = solver
  )
  # Objective and constraints
  Cᴴ(r,s,u,φ,dΩ) = ∫((I ∘ φ)*(C ⊙ (ε(u[r])+εᴹ[r]) ⊙ εᴹ[s]))dΩ
  dCᴴ(r,s,q,u,φ,dΩ) = ∫(-q*(C ⊙ (ε(u[r])+εᴹ[r]) ⊙ (ε(u[s])+εᴹ[s]))*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  κ(u,φ,dΩ) = -1/9*(Cᴴ(1,1,u,φ,dΩ)+Cᴴ(2,2,u,φ,dΩ)+Cᴴ(3,3,u,φ,dΩ)+
    2*(Cᴴ(1,2,u,φ,dΩ)+Cᴴ(1,3,u,φ,dΩ)+Cᴴ(2,3,u,φ,dΩ)))
  dκ(q,u,φ,dΩ) = -1/9*(dCᴴ(1,1,q,u,φ,dΩ)+dCᴴ(2,2,q,u,φ,dΩ)+dCᴴ(3,3,q,u,φ,dΩ)+
    2*(dCᴴ(1,2,q,u,φ,dΩ)+dCᴴ(1,3,q,u,φ,dΩ)+dCᴴ(2,3,q,u,φ,dΩ)))
  vol_D = sum(∫(1)dΩ)
  C1(u,φ,dΩ,dΓ_N) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
  dC1(q,u,φ,dΩ,dΓ_N) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  pcfs = PDEConstrainedFunctionals(κ,[C1],state_map,
    analytic_dJ=dκ,analytic_dC=[dC1])
  # Velocity extension
  α = 4*maximum(get_el_Δ(model))
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
    data = ["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh))]
    iszero(it % iter_mod) && writevtk(Ω,path*"out$it",cellfields=data)
    write_history(path*"/history.txt",get_history(optimiser);ranks)
  end
  # Final structure
  it = get_history(optimiser).niter; uh = get_state(pcfs)
  writevtk(Ω,path*"out$it",cellfields=["φ"=>φh,
    "H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh))])
end

with_mpi() do distribute
  mesh_partition = (5,5,5)
  solver_options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
    -ksp_converged_reason -ksp_rtol 1.0e-12"
  GridapPETSc.with(args=split(solver_options)) do
    main(mesh_partition,distribute)
  end
end