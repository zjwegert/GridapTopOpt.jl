using Gridap, Gridap.MultiField, GridapDistributed, GridapPETSc, GridapSolvers, 
  PartitionedArrays, LSTO_Distributed, SparseMatricesCSR

"""
  (MPI) Inverter mechanism with Hilbertian projection method in 3D.

  Optimisation problem:
      Min J(Ω) = ηᵢₙ*∫ u⋅e₁ dΓᵢₙ
        Ω
    s.t., Vol(Ω) = vf,
            C(Ω) = 0, 
          ⎡u∈V=H¹(Ω;u(Γ_D)=0)ᵈ, 
          ⎣∫ C ⊙ ε(u) ⊙ ε(v) dΩ + ∫ kₛv⋅u dΓₒᵤₜ = ∫ v⋅g dΓᵢₙ , ∀v∈V.
        
    where C(Ω) = ∫ -u⋅e₁-δₓ dΓₒᵤₜ. We assume symmetry in the problem to aid
     convergence.
""" 
function main(mesh_partition,distribute,el_size)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  ## Parameters
  order = 1
  dom = (0,1,0,1,0,1)
  γ = 0.05
  γ_reinit = 0.5
  max_steps = floor(Int,minimum(el_size)/3)
  tol = 1/(2order^2)*prod(inv,minimum(el_size))
  C = isotropic_3d(1.0,0.3)
  η_coeff = 2
  α_coeff = 4
  vf=0.4
  δₓ=0.75
  ks = 0.01
  g = VectorValue(1,0,0)
  path = dirname(dirname(@__DIR__))*"/results/3d_inverter_HPM"

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

  ## Finite difference solver and level set function
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
  make_dir(path;ranks=ranks)
  optimiser = HilbertianProjection(pcfs,stencil,vel_ext,φh;γ,γ_reinit,α_min=0.5,#α_min=0.7,ls_γ_max=0.05,
    verbose=i_am_main(ranks),constraint_names=[:Vol,:UΓ_out])
  for (it, uh, φh) in optimiser
    write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh"=>uh])
    write_history(path*"/history.txt",optimiser.history;ranks=ranks)
  end
  it = optimiser.history.niter; uh = get_state(optimiser.problem)
  write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh"=>uh];iter_mod=1)
end

with_mpi() do distribute
  mesh_partition = (5,5,5)
  el_size = (100,100,100)
  hilb_solver_options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
    -ksp_converged_reason -ksp_rtol 1.0e-12 -mat_block_size 3
    -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_coarse_sub_pc_type cholesky"
  
  GridapPETSc.with(args=split(hilb_solver_options)) do
    main(mesh_partition,distribute,el_size)
  end
end