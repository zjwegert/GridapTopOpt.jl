using Gridap, Gridap.MultiField, GridapDistributed, GridapPETSc, GridapSolvers, 
  PartitionedArrays, LSTO_Distributed, SparseMatricesCSR

using GridapSolvers: NewtonSolver

"""
  (MPI) Minimum hyperelastic compliance with Lagrangian method in 3D.

  Optimisation problem:
    ...
""" 
function main(mesh_partition,distribute,el_size)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  ## Parameters
  order = 1
  xmax,ymax,zmax=(2.0,1.0,1.0)
  prop_Γ_N = 0.4
  dom = (0,xmax,0,ymax,0,zmax)
  γ = 0.05
  γ_reinit = 0.5
  max_steps = floor(Int,minimum(el_size)/3)
  tol = 1/(2order^2)*prod(inv,minimum(el_size))
  η_coeff = 2
  α_coeff = 4
  vf = 0.4
  path = dirname(dirname(@__DIR__))*"/results/3d_hyperelastic_compliance_ALM"

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

  ## Material properties and loading
  _E = 1000
  ν = 0.3
  μ, λ = _E/(2*(1 + ν)), _E*ν/((1 + ν)*(1 - 2*ν))
  g = VectorValue(0,0,-25)

  ## Saint Venant–Kirchhoff law
  F(∇u) = one(∇u) + ∇u'
  E(F) = 0.5*( F' ⋅ F - one(F) )
  Σ(∇u) = λ*tr(E(F(∇u)))*one(∇u)+2*μ*E(F(∇u))
  T(∇u) = F(∇u) ⋅ Σ(∇u)
  res(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*((T ∘ ∇(u)) ⊙ ∇(v)))*dΩ - ∫(g⋅v)dΓ_N

  ## Optimisation functionals
  J(u,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*((T ∘ ∇(u)) ⊙ ∇(u)))dΩ
  Vol(u,φ,dΩ,dΓ_N) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
  dVol(q,u,φ,dΩ,dΓ_N) = ∫(1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  stencil = AdvectionStencil(FirstOrderStencil(3,Float64),model,V_φ,tol,max_steps)

  ## Setup solver and FE operators
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
  pcfs = PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dC=[dVol])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(Δ)
  a_hilb(p,q) = ∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(
    a_hilb,U_reg,V_reg;
    assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls = PETScLinearSolver()
  )
  
  ## Optimiser
  make_dir(path,ranks=ranks)
  optimiser = AugmentedLagrangian(pcfs,stencil,vel_ext,φh;
    γ,γ_reinit,verbose=i_am_main(ranks),constraint_names=[:Vol])
  for (it, uh, φh) in optimiser
    write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh"=>uh])
    write_history(path*"/history.txt",optimiser.history;ranks=ranks)
  end
  it = optimiser.history.niter; uh = get_state(optimiser.problem)
  write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh"=>uh];iter_mod=1)
end

with_mpi() do distribute
  mesh_partition = (5,5,5)
  el_size = (160,80,80)
  hilb_solver_options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
    -ksp_converged_reason -ksp_rtol 1.0e-12 -mat_block_size 3
    -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_coarse_sub_pc_type cholesky"
  
  GridapPETSc.with(args=split(hilb_solver_options)) do
    main(mesh_partition,distribute,el_size)
  end
end