using Gridap, Gridap.MultiField, GridapDistributed, GridapPETSc, GridapSolvers, 
  PartitionedArrays, LevelSetTopOpt, SparseMatricesCSR
using GridapSolvers: NewtonSolver

function main(mesh_partition,distribute,el_size,δₓ)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  ## Parameters
  order = 1
  xmax,ymax,zmax=(1.0,1.0,1.0)
  dom = (0,xmax,0,ymax,0,zmax)
  η_coeff = 2
  prop_Γ_N = 0.2;
  path = dirname(dirname(@__DIR__))*"/results/testing_hyper_elast"
  i_am_main(ranks) && ~isdir(path) && mkdir(path)

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
  dΩ = Measure(Ω,2order)

  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D","Gamma_N"])
  U = TrialFESpace(V,[VectorValue(0.0,0.0,0.0),VectorValue(δₓ,0.0,0.0)])
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)

  ## Create FE functions
  # φh = interpolate(initial_lsf(4,0.2),V_φ)
  φh = interpolate(x->-1,V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  ## Material properties and loading
  _E = 1000
  ν = 0.3
  μ, λ = _E/(2*(1 + ν)), _E*ν/((1 + ν)*(1 - 2*ν))

  ## Saint Venant–Kirchhoff law
  F(∇u) = one(∇u) + ∇u'
  E(F) = 0.5*( F' ⋅ F - one(F) )
  Σ(∇u) = λ*tr(E(F(∇u)))*one(∇u)+2*μ*E(F(∇u))
  T(∇u) = F(∇u) ⋅ Σ(∇u)
  res(u,v,φ,dΩ) = ∫((I ∘ φ)*((T ∘ ∇(u)) ⊙ ∇(v)))dΩ

  ## Setup solver and FE operators
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  lin_solver = MUMPSSolver()
  nl_solver = NewtonSolver(lin_solver;maxiter=50,rtol=10^-8,verbose=i_am_main(ranks))

  state_map = NonlinearFEStateMap(
    res,U,V,V_φ,U_reg,φh,dΩ;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
    nls = nl_solver, adjoint_ls = lin_solver
  )

  ## Optimiser
  u = LevelSetTopOpt.forward_solve!(state_map,φh)
  uh = FEFunction(U,u)
  writevtk(Ω,path*"/struc_$δₓ",cellfields=["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
end

function main_alt(mesh_partition,distribute,el_size,gz)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  ## Parameters
  order = 1
  xmax,ymax,zmax=(2.0,1.0,1.0)
  dom = (0,xmax,0,ymax,0,zmax)
  η_coeff = 2
  prop_Γ_N = 0.8;
  γ_reinit = 0.5
  max_steps = floor(Int,minimum(el_size)/3)
  tol = 1/(5order^2)/minimum(el_size)
  path = dirname(dirname(@__DIR__))*"/results/testing_hyper_elast"
  i_am_main(ranks) && ~isdir(path) && mkdir(path)

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

  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,VectorValue(0.0,0.0,0.0))
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)

  ## Create FE functions
  # φh = interpolate(initial_lsf(4,0.2),V_φ)
  φh = interpolate(x->-1,V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  ## Material properties and loading
  _E = 1000
  ν = 0.3
  μ, λ = _E/(2*(1 + ν)), _E*ν/((1 + ν)*(1 - 2*ν))
  g = VectorValue(-gz,0,0)

  # Deformation gradient
  # F(∇u) = one(∇u) + ∇u'
  # J(F) = sqrt(det(C(F)))
  # # Derivative of green Strain
  # dE(∇du,∇u) = 0.5*( ∇du⋅F(∇u) + (∇du⋅F(∇u))' )
  # # Right Caughy-green deformation tensor
  # C(F) = (F')⋅F
  # # Constitutive law (Neo hookean)
  # function S(∇u)
  #   Cinv = inv(C(F(∇u)))
  #   μ*(one(∇u)-Cinv) + λ*log(J(F(∇u)))*Cinv
  # end
  # # Cauchy stress tensor
  # σ(∇u) = (1.0/J(F(∇u)))*F(∇u)⋅S(∇u)⋅(F(∇u))'
  # res(u,v,φ,dΩ,dΓ_N) = ∫( (I ∘ φ)*((dE∘(∇(v),∇(u))) ⊙ (S∘∇(u))) )*dΩ - ∫(g⋅v)dΓ_N

  # ALT
  F(∇u) = one(∇u) + ∇u'
  E(F) = 0.5*( F' ⋅ F - one(F) )
  Σ(∇u) = λ*tr(E(F(∇u)))*one(∇u)+2*μ*E(F(∇u))
  T(∇u) = F(∇u) ⋅ Σ(∇u)
  res(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*((T ∘ ∇(u)) ⊙ ∇(v)))dΩ - ∫(g⋅v)dΓ_N

  ## Finite difference solver and level set function
  # stencil = AdvectionStencil(FirstOrderStencil(3,Float64),model,V_φ,tol,max_steps)
  # reinit!(stencil,φh,γ_reinit)

  ## Setup solver and FE operators
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  lin_solver = MUMPSSolver()
  nl_solver = NewtonSolver(lin_solver;maxiter=50,rtol=10^-8,verbose=i_am_main(ranks))

  state_map = NonlinearFEStateMap(
    res,U,V,V_φ,U_reg,φh,dΩ,dΓ_N;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
    nls = nl_solver, adjoint_ls = lin_solver
  )

  ## Optimiser
  u = LevelSetTopOpt.forward_solve!(state_map,φh)
  uh = FEFunction(U,u)
  writevtk(Ω,path*"/struc_neohook_$gz",cellfields=["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
end

with_mpi() do distribute
  mesh_partition = (2,2,2)
  el_size = (43,43,43)
  # solver = "-pc_type jacobi -ksp_type cg -ksp_monitor_short 
    # -ksp_converged_reason -ksp_rtol 1.0e-3 -mat_block_size 3"
  # solver = "-pc_type mg -mg_levels_ksp_max_it 2 -snes_monitor_short -ksp_monitor_short -npc_snes_type fas 
    # -npc_fas_levels_snes_type ncg -npc_fas_levels_snes_max_it 3 -npc_snes_monitor_short -snes_max_it 2"
  # solver = "-snes_type aspin -snes_monitor_short -ksp_monitor_short -npc_sub_snes_rtol 1e-2 -ksp_rtol 1e-2 -ksp_max_it 14 
  #   -snes_converged_reason -snes_max_linear_solve_fail 100 -snes_max_it 4 -npc_sub_ksp_type preonly -npc_sub_pc_type lu"
  # solver = "-snes_rtol 1e-05 -snes_monitor_short -snes_converged_reason
  #   -ksp_type fgmres -ksp_rtol 1e-10 -ksp_monitor_short -ksp_converged_reason
  #   -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type upper
  #   -fieldsplit_deformation_ksp_type preonly -fieldsplit_deformation_pc_type lu
  #   -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi"
  
  GridapPETSc.with() do #args=split(solver)) do
    # main(mesh_partition,distribute,el_size,0.02)
    # main(mesh_partition,distribute,el_size,0.05)
    main(mesh_partition,distribute,el_size,0.1)
    # main_alt(mesh_partition,distribute,el_size,-20)
    # main_alt(mesh_partition,distribute,el_size,-50)
    # main_alt(mesh_partition,distribute,el_size,-100)
    # main_alt(mesh_partition,distribute,el_size,-150)
  end
end