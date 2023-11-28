using Gridap, GridapDistributed, GridapPETSc, PartitionedArrays, LSTO_Distributed, SparseMatricesCSR

"""
  (MPI) Maximum bulk modulus inverse homogenisation with augmented Lagrangian method in 3D.

  Optimisation problem:
      Min J(Ω) = -κ(Ω)
        Ω
    s.t., Vol(Ω) = Vf,
          ⎡For unique εᴹᵢ, find uᵢ∈V=H¹ₚₑᵣ(Ω)ᵈ, 
          ⎣∫ ∑ᵢ C ⊙ ε(uᵢ) ⊙ ε(vᵢ) dΩ = ∫ -∑ᵢ C ⊙ ε⁰ᵢ ⊙ ε(vᵢ) dΩ, ∀v∈V.
""" 
function main(mesh_partition,distribute)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  ## Parameters
  order = 1;
  xmax,ymax=(1.0,1.0,1.0)
  dom = (0,xmax,0,ymax,0,zmax);
  el_size = (50,50,50);
  γ = 0.05;
  γ_reinit = 0.5;
  max_steps = floor(Int,minimum(el_size)/3)
  tol = 1/(order^2*10)*prod(inv,minimum(el_size))
  C = isotropic_2d(1.,0.3);
  η_coeff = 2;
  α_coeff = 4;
  path = "./Results/MPI_main_3d_inverse_homenisation_AGM"

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
  U = MultiFieldFESpace([_U,_U,_U,_U,_U,_U]);
  V = MultiFieldFESpace([_V,_V,_V,_V,_V,_V]);
  V_reg = V_φ = TestFESpace(model,reffe_scalar)
  U_reg = TrialFESpace(V_reg)

  ## Create FE functions
  lsf_fn = x -> cos(2π*x[1]) + cos(2π*y[2]) + cos(2π*z[3])
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

  a(u,v,φ,dΩ) = ∫((I ∘ φ)*sum(C ⊙ ε(u[i]) ⊙ ε(v[i]) for i ∈ eachindex(εᴹ)))dΩ
  l(v,φ,dΩ) = ∫(-(I ∘ φ)*sum(C ⊙ εᴹ[i] ⊙ ε(v[i]) for i ∈ eachindex(εᴹ)))dΩ;
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
  solver = ElasticitySolver(Ω,U);

  state_map = AffineFEStateMap(a,l,res,U,V,V_φ,U_reg,φh,dΩ;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
    ls=solver,
    adjoint_ls=solver)
  pcfs = PDEConstrainedFunctionals(J,[Vol],state_map;analytic_dJ=dJ,analytic_dC=[dVol])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(Δ)
  a_hilb = (p,q,dΩ)->∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg,dΩ,
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

with_mpi() do distribute
  mesh_partition = (3,3,3)
  hilb_solver_options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
    -ksp_converged_reason -ksp_rtol 1.0e-12"
  
  GridapPETSc.with(args=split(hilb_solver_options)) do
    main(mesh_partition,distribute)
  end
end;