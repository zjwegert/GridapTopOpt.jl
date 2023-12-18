using Gridap, Gridap.MultiField, GridapDistributed, GridapPETSc, GridapSolvers, 
  PartitionedArrays, LSTO_Distributed, SparseMatricesCSR
using Gridap.MultiField: BlockMultiFieldStyle

"""
  (MPI) Maximum bulk modulus inverse homogenisation with augmented Lagrangian method in 3D.

  Optimisation problem:
      Min J(Ω) = -κ(Ω)
        Ω
    s.t., Vol(Ω) = Vf,
          ⎡For unique εᴹᵢ, find uᵢ∈V=H¹ₚₑᵣ(Ω)ᵈ, 
          ⎣∫ ∑ᵢ C ⊙ ε(uᵢ) ⊙ ε(vᵢ) dΩ = ∫ -∑ᵢ C ⊙ ε⁰ᵢ ⊙ ε(vᵢ) dΩ, ∀v∈V.
""" 
function main(mesh_partition,distribute,el_size,diag_assem::Bool)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  t = PTimer(ranks)

  ## Parameters
  order = 1;
  xmax,ymax,zmax=(1.0,1.0,1.0)
  dom = (0,xmax,0,ymax,0,zmax);
  γ = 0.05;
  γ_reinit = 0.5;
  max_steps = floor(Int,minimum(el_size)/3)
  tol = 1/(order^2*10)*prod(inv,minimum(el_size))
  C = isotropic_3d(1.,0.3);
  η_coeff = 2;
  α_coeff = 4;
  path = dirname(dirname(@__DIR__))*"/results/MPI_main_3d_inverse_homenisation_ALM"

  ## FE Setup
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size,isperiodic=(true,true,true));
  Δ = get_Δ(model)
  f_Γ_D(x) = iszero(x)
  update_labels!(1,model,f_Γ_D,"origin")

  ## Triangulations and measures
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2order)
  vol_D = sum(∫(1)dΩ)

  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  _V = TestFESpace(model,reffe;dirichlet_tags=["origin"])
  _U = TrialFESpace(_V,VectorValue(0.0,0.0,0.0))
  mfs = BlockMultiFieldStyle()
  U = MultiFieldFESpace([_U,_U,_U,_U,_U,_U];style=mfs);
  V = MultiFieldFESpace([_V,_V,_V,_V,_V,_V];style=mfs);
  V_reg = V_φ = TestFESpace(model,reffe_scalar)
  U_reg = TrialFESpace(V_reg)

  ## Create FE functions
  lsf_fn = x -> cos(2π*x[1]) + cos(2π*x[2]) + cos(2π*x[3])
  φh = interpolate(lsf_fn,V_φ);
  φ = get_free_dof_values(φh)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  εᴹ = (TensorValue(1.,0.,0.,0.,0.,0.,0.,0.,0.),           # ϵᵢⱼ⁽¹¹⁾≡ϵᵢⱼ⁽¹⁾
        TensorValue(0.,0.,0.,0.,1.,0.,0.,0.,0.),           # ϵᵢⱼ⁽²²⁾≡ϵᵢⱼ⁽²⁾
        TensorValue(0.,0.,0.,0.,0.,0.,0.,0.,1.),           # ϵᵢⱼ⁽³³⁾≡ϵᵢⱼ⁽³⁾
        TensorValue(0.,0.,0.,0.,0.,1/2,0.,1/2,0.),         # ϵᵢⱼ⁽²³⁾≡ϵᵢⱼ⁽⁴⁾
        TensorValue(0.,0.,1/2,0.,0.,0.,1/2,0.,0.),         # ϵᵢⱼ⁽¹³⁾≡ϵᵢⱼ⁽⁵⁾
        TensorValue(0.,1/2,0.,1/2,0.,0.,0.,0.,0.))         # ϵᵢⱼ⁽¹²⁾≡ϵᵢⱼ⁽⁶⁾

  a(u,v,φ,dΩ) = ∫((I ∘ φ)*sum(C ⊙ ε(u[i]) ⊙ ε(v[i]) for i = 1:length(u)))dΩ
  l(v,φ,dΩ) = ∫(-(I ∘ φ)*sum(C ⊙ εᴹ[i] ⊙ ε(v[i]) for i = 1:length(v)))dΩ;
  res(u,v,φ,dΩ) = a(u,v,φ,dΩ) - l(v,φ,dΩ)

  ## Optimisation functionals
  _C(C,ε_p,ε_q) = C ⊙ ε_p ⊙ ε_q;

  _K(C,u,εᴹ) = (_C(C,ε(u[1])+εᴹ[1],εᴹ[1]) + _C(C,ε(u[2])+εᴹ[2],εᴹ[2]) + _C(C,ε(u[3])+εᴹ[3],εᴹ[3]) + 
              2(_C(C,ε(u[1])+εᴹ[1],εᴹ[2]) + _C(C,ε(u[1])+εᴹ[1],εᴹ[3]) + _C(C,ε(u[2])+εᴹ[2],εᴹ[3])))/9 

  _v_K(C,u,εᴹ) = (_C(C,ε(u[1])+εᴹ[1],ε(u[1])+εᴹ[1]) + _C(C,ε(u[2])+εᴹ[2],ε(u[2])+εᴹ[2]) + _C(C,ε(u[3])+εᴹ[3],ε(u[3])+εᴹ[3]) + 
                2(_C(C,ε(u[1])+εᴹ[1],ε(u[2])+εᴹ[2]) + _C(C,ε(u[1])+εᴹ[1],ε(u[3])+εᴹ[3]) + _C(C,ε(u[2])+εᴹ[2],ε(u[3])+εᴹ[3])))/9 

  J = (u,φ,dΩ) -> ∫(-(I ∘ φ)*_K(C,u,εᴹ))dΩ
  dJ = (q,u,φ,dΩ) -> ∫(-_v_K(C,u,εᴹ)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;
  Vol = (u,φ,dΩ) -> ∫(((ρ ∘ φ) - 0.5)/vol_D)dΩ;
  dVol = (q,u,φ,dΩ) -> ∫(1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  stencil = AdvectionStencil(FirstOrderStencil(3,Float64),model,V_φ,Δ./order,max_steps,tol)
  reinit!(stencil,φ,γ_reinit)

  ## Setup solver and FE operators
  Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv=Vector{PetscScalar}
  P = BlockDiagonalPreconditioner(map(Vi -> ElasticitySolver(Vi),V))
  solver = GridapSolvers.LinearSolvers.GMRESSolver(100;Pr=P,rtol=1.e-8,verbose=i_am_main(ranks))

  assem_U,assem_adjoint = if diag_assem
    DiagonalBlockMatrixAssembler(SparseMatrixAssembler(Tm,Tv,U,V)),
      DiagonalBlockMatrixAssembler(SparseMatrixAssembler(Tm,Tv,V,U))
  else
    SparseMatrixAssembler(Tm,Tv,V,U),
      SparseMatrixAssembler(Tm,Tv,U_reg,U_reg)
  end

  state_map = AffineFEStateMap(a,l,res,U,V,V_φ,U_reg,φh,dΩ;
    assem_U = assem_U,
    assem_adjoint = assem_adjoint,
    assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
    ls=solver,
    adjoint_ls=solver)
  pcfs = PDEConstrainedFunctionals(J,[Vol],state_map;analytic_dJ=dJ,analytic_dC=[dVol])

  J_init,C_init,dJ,dC = Gridap.evaluate!(pcfs,φ)
  u_vec = pcfs.state_map.fwd_caches[4]

  tic!(t)
  state_map(φ)
  toc!(t,"Assembly and solve")

  return pcfs,[J_init,C_init,dJ,dC],u_vec,t

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg,
    assem=SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls=PETScLinearSolver())
  
  ## Optimiser
  make_dir(path;ranks=ranks)
  optimiser = AugmentedLagrangian(φ,pcfs,stencil,vel_ext,interp,el_size,γ,γ_reinit);
  for history in optimiser
    it,Ji,Ci,Li = last(history)
    λi = optimiser.λ; Λi = optimiser.Λ
    print_history(it,["J"=>Ji,"C"=>Ci,"L"=>Li,"λ"=>λi,"Λ"=>Λi];ranks=ranks)
    write_history(history,path*"/history.csv";ranks=ranks)
    write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh))])
  end
  it,Ji,Ci,Li = last(optimiser.history)
  λi = optimiser.λ; Λi = optimiser.Λ
  print_history(it,["J"=>Ji,"C"=>Ci,"L"=>Li,"λ"=>λi,"Λ"=>Λi];ranks=ranks)
  write_history(optimiser.history,path*"/history.csv";ranks=ranks)
  write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh))];iter_mod=1)
end

# RUN: mpiexecjl --project=. -n 64 julia ./scripts/MPI/MPI_main_inverse_homenisation_ALM.jl
with_mpi() do distribute
  mesh_partition = (3,2,2)
  el_size = (40,40,40)
  hilb_solver_options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
    -ksp_converged_reason -ksp_rtol 1.0e-12 -mat_block_size 3
    -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_coarse_sub_pc_type cholesky"
  
  _PSFs,_OBJ_VALS,_U,T = GridapPETSc.with(args=split(hilb_solver_options)) do
    main(mesh_partition,distribute,el_size,false)
  end

  _PSFs_diag,_OBJ_VALS_diag,_U_diag,T_diag = GridapPETSc.with(args=split(hilb_solver_options)) do
    main(mesh_partition,distribute,el_size,true)
  end

  @show _OBJ_VALS==_OBJ_VALS_diag
  @show _U == _U_diag

  display(T)
  display(T_diag)
end;