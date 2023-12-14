using Gridap, Gridap.MultiField, GridapDistributed, GridapPETSc, GridapSolvers, 
  PartitionedArrays, LSTO_Distributed, SparseMatricesCSR

"""
  (MPI) Minimum thermal compliance with Lagrangian method in 3D with nonlinear diffusivity.

  Optimisation problem:
      Min J(Ω) = ∫ D(u)*∇(u)⋅∇(u) + ξ dΩ
        Ω
    s.t., ⎡u∈V=H¹(Ω;u(Γ_D)=0),
          ⎣∫ D(u)*∇(u)⋅∇(v) dΩ = ∫ v dΓ_N, ∀v∈V.
  
  In this example D(u) = D0*(exp(ξ*u))
""" 
function main(mesh_partition,distribute,el_size)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  ## Parameters
  order = 1;
  xmax=ymax=zmax=1.0
  prop_Γ_N = 0.4;
  prop_Γ_D = 0.2
  dom = (0,xmax,0,ymax,0,zmax);
  γ = 0.1;
  γ_reinit = 0.5;
  max_steps = floor(Int,minimum(el_size)/3)
  tol = 1/(order^2*10)*prod(inv,minimum(el_size))
  η_coeff = 2;
  α_coeff = 4;
  path = "./Results/MPI_main_3d_nonlinear"

  ## FE Setup
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
  Δ = get_Δ(model)
  f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= ymax*prop_Γ_D + eps() || x[2] >= ymax-ymax*prop_Γ_D - eps()) &&
    (x[3] <= zmax*prop_Γ_D + eps() || x[3] >= zmax-zmax*prop_Γ_D - eps())) ? true : false;
  f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= ymax/2+ymax*prop_Γ_N/4 + eps() &&
    zmax/2-zmax*prop_Γ_N/4 - eps() <= x[3] <= zmax/2+zmax*prop_Γ_N/4 + eps()) ? true : false;
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
  φ = get_free_dof_values(φh)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  D0 = 1;
  ξ = -1;
  D(u) = D0*(exp(ξ*u));

  res(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*(D ∘ u)*∇(u)⋅∇(v))dΩ - ∫(v)dΓ_N

  ## Optimisation functionals
  ξ = 0.1;
  J = (u,φ,dΩ,dΓ_N) -> ∫((I ∘ φ)*(D ∘ u)*∇(u)⋅∇(u) + ξ*(ρ ∘ φ))dΩ

  ## Finite difference solver and level set function
  stencil = AdvectionStencil(FirstOrderStencil(3,Float64),model,V_φ,Δ./order,max_steps,tol)
  reinit!(stencil,φ,γ_reinit)

  ## Setup solver and FE operators
  Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv=Vector{PetscScalar}
  nl_solver = PETScNonlinearSolver()
  lin_solver = PETScLinearSolver()
  
  state_map = NonlinearFEStateMap(res,U,V,V_φ,U_reg,φh,dΩ,dΓ_N;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
    nls=nl_solver,
    adjoint_ls=lin_solver)

  pcfs = PDEConstrainedFunctionals(J,state_map)

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg,;
    assem=SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls=lin_solver)
  
  ## Optimiser
  make_dir(path,ranks=ranks)
  optimiser = AugmentedLagrangian(φ,pcfs,stencil,vel_ext,interp,el_size,γ,γ_reinit);
  for history in optimiser
    it,Ji,_,_ = last(history)
    print_history(it,["J"=>Ji],ranks=ranks)
    write_history(history,path*"/history.csv",ranks=ranks)
    uhi = get_state(pcfs)
    write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh"=>uhi])
  end
  it,Ji,_,_ = last(optimiser.history)
  print_history(it,["J"=>Ji],ranks=ranks)
  write_history(optimiser.history,path*"/history.csv",ranks=ranks)
  uhi = get_state(pcfs)
  write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh"=>uhi];iter_mod=1)
end

with_mpi() do distribute
  mesh_partition = (3,3,2)
  el_size = (100,100,100)

  options = "-snes_type newtonls -snes_linesearch_type basic  -snes_linesearch_damping 1.0"*
    " -snes_rtol 1.0e-14 -snes_atol 0.0 -snes_monitor -pc_type gamg -ksp_type cg"*
    " -snes_converged_reason -ksp_converged_reason -ksp_error_if_not_converged true -ksp_rtol 1.0e-12"*
    " -mat_block_size 3 -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_coarse_sub_pc_type cholesky"
  
  GridapPETSc.with(args=split(options)) do
    main(mesh_partition,distribute,el_size)
  end
end;