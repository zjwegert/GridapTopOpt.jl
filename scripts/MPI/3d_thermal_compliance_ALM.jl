using Gridap, Gridap.MultiField, GridapDistributed, GridapPETSc, GridapSolvers, 
  PartitionedArrays, LevelSetTopOpt, SparseMatricesCSR

global elx = parse(Int,ARGS[1])
global ely = parse(Int,ARGS[2])
global elz = parse(Int,ARGS[3])
global Px = parse(Int,ARGS[4])
global Py = parse(Int,ARGS[5])
global Pz = parse(Int,ARGS[6])

"""
  (MPI) Minimum thermal compliance with augmented Lagrangian method in 3D.

  Optimisation problem:
      Min J(Ω) = ∫ κ*∇(u)⋅∇(u) dΩ
      Ω
    s.t., Vol(Ω) = vf,
          ⎡u∈V=H¹(Ω;u(Γ_D)=0),
          ⎣∫ κ*∇(u)⋅∇(v) dΩ = ∫ v dΓ_N, ∀v∈V.
"""
function main(mesh_partition,distribute,el_size)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  ## Parameters
  order = 1
  xmax=ymax=zmax=1.0
  prop_Γ_N = 0.2
  prop_Γ_D = 0.2
  dom = (0,xmax,0,ymax,0,zmax)
  γ = 0.1
  γ_reinit = 0.5
  max_steps = floor(Int,order*minimum(el_size)/10)
  tol = 1/(coef*order^2)/minimum(el_size)
  κ = 1
  η_coeff = 2
  α_coeff = 4max_steps*γ
  vf = 0.4
  path = dirname(dirname(@__DIR__))*"/results/3d_thermal_compliance_ALM_Nx$(el_size[1])/"
  iter_mod = 10
  i_am_main(ranks) && mkdir(path)

  ## FE Setup
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
  el_Δ = get_el_Δ(model)
  f_Γ_D(x) = (x[1] ≈ 0.0) && (x[2] <= ymax*prop_Γ_D + eps() || x[2] >= ymax-ymax*prop_Γ_D - eps()) &&
    (x[3] <= zmax*prop_Γ_D + eps() || x[3] >= zmax-zmax*prop_Γ_D - eps())
  f_Γ_N(x) = (x[1] ≈ xmax) && (ymax/2-ymax*prop_Γ_N/2 - eps() <= x[2] <= ymax/2+ymax*prop_Γ_N/2 + eps()) &&
    (zmax/2-zmax*prop_Γ_N/2 - eps() <= x[3] <= zmax/2+zmax*prop_Γ_N/2 + eps())
  update_labels!(1,model,f_Γ_D,"Gamma_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")

  ## Triangulations and measures
  Ω = Triangulation(model)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΩ = Measure(Ω,2order)
  dΓ_N = Measure(Γ_N,2order)
  vol_D = sum(∫(1)dΩ)

  ## Spaces
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,0.0)
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)

  ## Create FE functions
  φh = interpolate(initial_lsf(4,0.2),V_φ);

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  a(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*κ*∇(u)⋅∇(v))dΩ
  l(v,φ,dΩ,dΓ_N) = ∫(10v)dΓ_N

  ## Optimisation functionals
  J(u,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*κ*∇(u)⋅∇(u))dΩ
  dJ(q,u,φ,dΩ,dΓ_N) = ∫((κ*∇(u)⋅∇(u))*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  Vol(u,φ,dΩ,dΓ_N) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
  dVol(q,u,φ,dΩ,dΓ_N) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  ls_evo = HamiltonJacobiEvolution(FirstOrderStencil(3,Float64),model,V_φ,tol,max_steps)

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
  pcfs = PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dJ=dJ,analytic_dC=[dVol])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_Δ)
  a_hilb(p,q) = ∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(
    a_hilb, U_reg, V_reg;
    assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls = solver
  )

  ## Optimiser
  optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
    γ,γ_reinit,verbose=i_am_main(ranks),constraint_names=[:Vol])
  for (it, uh, φh) in optimiser
    data = ["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh]
    iszero(it % iter_mod) && writevtk(Ω,path*"out$it",cellfields=data)
    write_history(path*"/history.txt",optimiser.history;ranks=ranks)
  end
  it = get_history(optimiser).niter; uh = get_state(pcfs)
  writevtk(Ω,path*"out$it",cellfields=["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
end

with_mpi() do distribute
  mesh_partition = (Px,Py,Pz)
  el_size = (elx,ely,elz)
  all_solver_options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
    -ksp_converged_reason -ksp_rtol 1.0e-12"
  
  GridapPETSc.with(args=split(all_solver_options)) do
    main(mesh_partition,distribute,el_size)
  end
end;