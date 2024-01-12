using Gridap, GridapDistributed, GridapPETSc, PartitionedArrays, LSTO_Distributed

"""
(MPI) Minimum thermal compliance with Lagrangian method in 2D.

Optimisation problem:
    Min J(Ω) = ∫ D*∇(u)⋅∇(u) + ξ dΩ
    Ω
  s.t., ⎡u∈V=H¹(Ω;u(Γ_D)=0),
        ⎣∫ D*∇(u)⋅∇(v) dΩ = ∫ v dΓ_N, ∀v∈V.
"""
function main(mesh_partition,el_size,distribute)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  ## Parameters
  order = 1
  xmax=ymax=1.0
  prop_Γ_N = 0.4
  prop_Γ_D = 0.2
  dom = (0,xmax,0,ymax)
  γ = 0.1
  γ_reinit = 0.5
  max_steps = floor(Int,minimum(el_size)/10)
  tol = 1/(order^2*10)*prod(inv,minimum(el_size))
  D = 1
  η_coeff = 2
  α_coeff = 4

  ## FE Setup
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
  Δ = get_Δ(model)
  f_Γ_D(x) = (x[1] ≈ 0.0) && ((x[2] <= ymax*prop_Γ_D + eps()) || (x[2] >= ymax-ymax*prop_Γ_D - eps()))
  f_Γ_N(x) = (x[1] ≈ xmax) && (ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= ymax/2+ymax*prop_Γ_N/4 + eps())
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
  φh = interpolate(gen_lsf(4,0.2),V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  a(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*D*∇(u)⋅∇(v))dΩ
  l(v,φ,dΩ,dΓ_N) = ∫(v)dΓ_N

  ## Optimisation functionals
  ξ = 0.3
  J(u,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*D*∇(u)⋅∇(u) + ξ*(ρ ∘ φ))dΩ
  dJ(q,u,φ,dΩ,dΓ_N) = ∫((ξ-D*∇(u)⋅∇(u))*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;

  ## Finite difference solver and level set function
  stencil = AdvectionStencil(FirstOrderStencil(2,Float64),model,V_φ,tol,max_steps)
  reinit!(stencil,φh,γ_reinit)

  ## Setup solver and FE operators
  state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh,dΩ,dΓ_N)
  pcfs = PDEConstrainedFunctionals(J,state_map,analytic_dJ=dJ)

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

  ## Optimiser
  return AugmentedLagrangian(pcfs,stencil,vel_ext,φh;γ,γ_reinit)
end

with_mpi() do distribute
  opt = main((2,3),(400,400),distribute)
  ## Benchmark optimiser
  # bopt = benchmark_optimizer(opt, 1, nothing)
  ## Benchmark forward problem
  bfwd = benchmark_forward_problem(opt.problem.state_map, opt.φ0, nothing)
  ## Benchmark advection
  v = get_free_dof_values(interpolate(FEFunction(LSTO_Distributed.get_deriv_space(opt.problem.state_map),
    opt.problem.dJ),LSTO_Distributed.get_aux_space(opt.problem.state_map)))
  badv = benchmark_advection(opt.stencil, get_free_dof_values(opt.φ0), v, 0.1, nothing)
  ## Benchmark reinitialisation
  brinit = benchmark_reinitialisation(opt.stencil, get_free_dof_values(opt.φ0), 0.1, nothing)
  ## Benchmark velocity extension
  bvelext = benchmark_velocity_extension(opt.vel_ext, opt.problem.dJ, nothing)
  ## Printing
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  if i_am_main(ranks)
    # display("bopt => $bopt")
    display("bfwd => $bfwd")
    display("badv => $badv")
    display("brinit => $brinit")
    display("bvelext => $bvelext")
  end
  nothing
end;

