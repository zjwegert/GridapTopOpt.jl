using Gridap, GridapDistributed, GridapPETSc, PartitionedArrays, LSTO_Distributed

"""
  (MPI) Inverter mechanism with augmented Lagrangian method in 3D.
  
  Ref: http://doi.org/10.1007/s00158-018-1950-2

  Optimisation problem:
      Min J(Ω) = ∫ ηᵢₙ * uₓ dΓᵢₙ + ∫ ηₒᵤₜ * uₓ dΓₒᵤₜ
        Ω
    s.t., Vol(Ω) = Vf,
          ⎡u∈V=H¹(Ω;u(Γ_D)=0)³, 
          ⎣∫ C ⊙ ε(u) ⊙ ε(v) dΩ + ∫ kₛv⋅u dΓₒᵤₜ = ∫ v⋅g dΓᵢₙ , ∀v∈V.
""" 
function main(mesh_partition,distribute,el_size)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  ## Parameters
  order = 1;
  dom = (0,1,0,1,0,1);
  γ = 0.1;
  γ_reinit = 0.5;
  max_steps = floor(Int,minimum(el_size)/3)
  tol = 1/(order^2*10)*prod(inv,minimum(el_size))
  C = isotropic_3d(1.0,0.3);
  η_coeff = 2;
  α_coeff = 4;
  path = "./Results/MPI_main_3d_inverter_mechanism"

  η_in = 2;
  η_out = 1;
  ks = 0.01;
  g = VectorValue(1,0,0);

  ## FE Setup
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
  Δ = get_Δ(model)
  f_Γ_in(x) = (x[1] ≈ 0.0) && 0.47 - eps() <= x[2] <= 0.53 + eps() && 
    0.47 - eps() <= x[3] <= 0.53 + eps() ? true : false;
  f_Γ_out(x) = (x[1] ≈ 1.0) && 0.43 - eps() <= x[2] <= 0.57 + eps() && 
    0.43 - eps() <= x[3] <= 0.57 + eps() ? true : false;
  f_Γ_D(x) = x[1] ≈ 0.0 && (x[2] <= 0.1 || x[2] >= 0.9) && 
    (x[3] <= 0.1 || x[3] >= 0.9)  ? true : false;
  update_labels!(1,model,f_Γ_in,"Gamma_in")
  update_labels!(2,model,f_Γ_out,"Gamma_out")
  update_labels!(3,model,f_Γ_D,"Gamma_D")

  ## Triangulations and measures
  Ω = Triangulation(model)
  Γ_in = BoundaryTriangulation(model,tags="Gamma_in")
  Γ_out = BoundaryTriangulation(model,tags="Gamma_out")
  dΩ = Measure(Ω,2order)
  dΓ_in = Measure(Γ_in,2order)
  dΓ_out = Measure(Γ_out,2order)
  vol_D = sum(∫(1)dΩ)

  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,VectorValue(0.0,0.0,0.0))
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_in","Gamma_out"])
  U_reg = TrialFESpace(V_reg,[0,0])

  ## Create FE functions
  φh = interpolate(gen_lsf(4,0.2),V_φ);
  φ = get_free_dof_values(φh)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  a(u,v,φ,dΩ,dΓ_in,dΓ_out) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(v)))dΩ + ∫(ks*(u⋅v))dΓ_out
  l(v,φ,dΩ,dΓ_in,dΓ_out) = ∫(v⋅g)dΓ_in
  res(u,v,φ,dΩ,dΓ_in,dΓ_out) = a(u,v,φ,dΩ,dΓ_in,dΓ_out) - l(v,φ,dΩ,dΓ_in,dΓ_out)

  ## Optimisation functionals
  J = (u,φ,dΩ,dΓ_in,dΓ_out) -> ∫(η_in*u⋅VectorValue(1,0,0))dΓ_in + 
                               ∫(η_out*u⋅VectorValue(1,0,0))dΓ_out
  Vol = (u,φ,dΩ,dΓ_in,dΓ_out) -> ∫(((ρ ∘ φ) - 0.5)/vol_D)dΩ;
  dVol = (q,u,φ,dΩ,dΓ_in,dΓ_out) -> ∫(1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  stencil = AdvectionStencil(FirstOrderStencil(2,Float64),model,V_φ,Δ./order,max_steps,tol)
  reinit!(stencil,φ,γ_reinit)

  ## Setup solver and FE operators
  Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv=Vector{PetscScalar}
  solver = ElasticitySolver(V)
  
  state_map = AffineFEStateMap(a,l,res,U,V,V_φ,U_reg,φh,dΩ,dΓ_in,dΓ_out;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
    ls=solver,
    adjoint_ls=solver)
  pcfs = PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dC=[dVol])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(Δ)
  a_hilb = (p,q,dΩ)->∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg,dΩ;
    assem=SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls=PETScLinearSolver())
  
  ## Optimiser
  make_dir(path)
  optimiser = AugmentedLagrangian(φ,pcfs,stencil,vel_ext,interp,el_size,γ,γ_reinit);
  for history in optimiser
    it,Ji,Ci,Li = last(history)
    λi = optimiser.λ; Λi = optimiser.Λ
    print_history(it,["J"=>Ji,"C"=>Ci,"L"=>Li,"λ"=>λi,"Λ"=>Λi])
    write_history(history,path*"/history.csv")
    write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh))])
  end
  it,Ji,Ci,Li = last(optimiser.history)
  λi = optimiser.λ; Λi = optimiser.Λ
  print_history(it,["J"=>Ji,"C"=>Ci,"L"=>Li,"λ"=>λi,"Λ"=>Λi])
  write_history(optimiser.history,path*"/history.csv")
  uhi = get_state(pcfs)
  write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh"=>uhi];iter_mod=1)
end

with_mpi() do distribute
  mesh_partition = (4,4,4)
  el_size = (100,100,100)
  hilb_solver_options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
    -ksp_converged_reason -ksp_rtol 1.0e-12"
  
  GridapPETSc.with(args=split(hilb_solver_options)) do
    main(mesh_partition,distribute,el_size)
  end
end;