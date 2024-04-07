using Gridap, Gridap.MultiField, GridapDistributed, GridapPETSc, GridapSolvers, 
  PartitionedArrays, LevelSetTopOpt, SparseMatricesCSR

using GridapSolvers: NewtonSolver

global elx = parse(Int,ARGS[1])
global ely = parse(Int,ARGS[2])
global elz = parse(Int,ARGS[3])
global Px = parse(Int,ARGS[4])
global Py = parse(Int,ARGS[5])
global Pz = parse(Int,ARGS[6])

"""
  (MPI) Minimum hyperelastic compliance with augmented Lagrangian method in 3D.
""" 
function main(mesh_partition,distribute,el_size)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  ## Parameters
  order = 1
  xmax,ymax,zmax=(2.0,1.0,1.0)
  prop_Γ_N = 0.2
  dom = (0,xmax,0,ymax,0,zmax)
  γ = 0.1
  γ_reinit = 0.5
  max_steps = floor(Int,order*minimum(el_size)/10)
  tol = 1/(5order^2)/minimum(el_size)
  η_coeff = 2
  α_coeff = 4max_steps*γ
  vf = 0.4
  path = dirname(dirname(@__DIR__))*"/results/3d_hyperelastic_compliance_neohook_ALM/"
  iter_mod = 10
  i_am_main(ranks) && mkdir(path)

  ## FE Setup
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size)
  el_Δ = get_el_Δ(model)
  f_Γ_D(x) = (x[1] ≈ 0.0)
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
  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,VectorValue(0.0,0.0,0.0))
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)

  ## Create FE functions
  φh = interpolate(initial_lsf(4,0.2),V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  ## Material properties
  E = 1000
  ν = 0.3
  μ, λ = E/(2*(1 + ν)), E*ν/((1 + ν)*(1 - 2*ν))
  g = VectorValue(0,0,-100)

  ## Neohookean hyperelastic material
  # Deformation gradient
  F(∇u) = one(∇u) + ∇u'
  J(F) = sqrt(det(C(F)))

  # Derivative of green Strain
  dE(∇du,∇u) = 0.5*( ∇du⋅F(∇u) + (∇du⋅F(∇u))' )

  # Right Caughy-green deformation tensor
  C(F) = (F')⋅F

  # Constitutive law (Neo hookean)
  function S(∇u)
    Cinv = inv(C(F(∇u)))
    μ*(one(∇u)-Cinv) + λ*log(J(F(∇u)))*Cinv
  end

  # Cauchy stress tensor and residual
  σ(∇u) = (1.0/J(F(∇u)))*F(∇u)⋅S(∇u)⋅(F(∇u))'
  res(u,v,φ,dΩ,dΓ_N) = ∫( (I ∘ φ)*((dE∘(∇(v),∇(u))) ⊙ (S∘∇(u))) )*dΩ - ∫(g⋅v)dΓ_N

  ## Optimisation functionals
  Obj(u,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*((dE∘(∇(u),∇(u))) ⊙ (S∘∇(u))))dΩ
  Vol(u,φ,dΩ,dΓ_N) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
  dVol(q,u,φ,dΩ,dΓ_N) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

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
  pcfs = PDEConstrainedFunctionals(Obj,[Vol],state_map,analytic_dC=[dVol])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_Δ)
  a_hilb(p,q) = ∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(
    a_hilb,U_reg,V_reg;
    assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls = PETScLinearSolver()
  )
  
  ## Optimiser
  optimiser = AugmentedLagrangian(pcfs,stencil,vel_ext,φh;
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
  hilb_solver_options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
    -ksp_converged_reason -ksp_rtol 1.0e-12 -mat_block_size 3
    -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_coarse_sub_pc_type cholesky"
  
  GridapPETSc.with(args=split(hilb_solver_options)) do
    main(mesh_partition,distribute,el_size)
  end
end