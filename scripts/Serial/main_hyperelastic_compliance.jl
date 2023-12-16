using Gridap, GridapDistributed, GridapPETSc, PartitionedArrays, LSTO_Distributed

using Gridap.Algebra: NewtonRaphsonSolver

"""
  (Serial) Minimum hyperelastic compliance with Lagrangian method in 2D.

  Optimisation problem:
    ...
""" 
function main()
  ## Parameters
  order = 1;
  xmax,ymax=2.0,1.0
  prop_Γ_N = 0.4;
  dom = (0,xmax,0,ymax);
  el_size = (200,200);
  γ = 0.03;
  γ_reinit = 0.5;
  max_steps = floor(Int,minimum(el_size)/10)
  tol = 1/(order^2*10)*prod(inv,minimum(el_size)) # <- We can do better than this I think
  η_coeff = 2;
  α_coeff = 4;

  ## FE Setup
  model = CartesianDiscreteModel(dom,el_size);
  Δ = get_Δ(model)
  f_Γ_D(x) = (x[1] ≈ 0.0) ? true : false;
  f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= 
    ymax/2+ymax*prop_Γ_N/4 + eps()) ? true : false;
  update_labels!(1,model,f_Γ_D,"Gamma_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")

  ## Triangulations and measures
  Ω = Triangulation(model)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΩ = Measure(Ω,2order)
  dΓ_N = Measure(Γ_N,2order)
  vol_D = sum(∫(1)dΩ)

  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,VectorValue(0.0,0.0))
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)

  ## Create FE functions
  φh = interpolate(gen_lsf(4,0.2),V_φ);
  φ = get_free_dof_values(φh)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  _E = 1000;
  ν = 0.3;
  μ, λ = _E/(2*(1 + ν)), _E*ν/((1 + ν)*(1 - 2*ν))
  g = VectorValue(0,-20)
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
  # Cauchy stress tensor
  σ(∇u) = (1.0/J(F(∇u)))*F(∇u)⋅S(∇u)⋅(F(∇u))'
  res(u,v,φ,dΩ,dΓ_N) = ∫( (I ∘ φ)*((dE∘(∇(v),∇(u))) ⊙ (S∘∇(u))) )*dΩ - ∫(g⋅v)dΓ_N

  ## Saint Venant–Kirchhoff law
  # F(∇u) = one(∇u) + ∇u'
  # E(F) = 0.5*( F' ⋅ F - one(F) )
  # Σ(∇u) = λ*tr(E(F(∇u)))*one(∇u)+2*μ*E(F(∇u))
  # T(∇u) = F(∇u) ⋅ Σ(∇u)
  # res(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*((T ∘ ∇(u)) ⊙ ∇(v)))*dΩ - ∫(g⋅v)dΓ_N
  ## Optimisation functionals
  # ξ = 0.5;
  # Obj = (u,φ,dΩ,dΓ_N) -> ∫((I ∘ φ)*((T ∘ ∇(u)) ⊙ ∇(u)) + ξ*(ρ ∘ φ))dΩ

  ## Optimisation functionals
  ξ = 0.5;
  Obj = (u,φ,dΩ,dΓ_N) -> ∫((I ∘ φ)*((dE∘(∇(u),∇(u))) ⊙ (S∘∇(u))) + ξ*(ρ ∘ φ))dΩ
  # Obj = (u,φ,dΩ,dΓ_N) -> ∫((I ∘ φ)*((T ∘ ∇(u)) ⊙ ∇(u)) + ξ*(ρ ∘ φ))dΩ

  ## Finite difference solver and level set function
  stencil = AdvectionStencil(FirstOrderStencil(2,Float64),model,V_φ,Δ./order,max_steps,tol)
  reinit!(stencil,φ,γ_reinit)

  ## Setup solver and FE operators
  nls = NewtonRaphsonSolver(BackslashSolver(),10^-10,50,true)  
  state_map = NonlinearFEStateMap(res,U,V,V_φ,U_reg,φh,dΩ,dΓ_N;nls=FESolver(nls))
  pcfs = PDEConstrainedFunctionals(Obj,state_map)

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)
  
  ## Optimiser
  path = "./results/main_hyperelastic_compliance_neohook_NonSymmetric_xi=$ξ"
  make_dir(path)
  optimiser = AugmentedLagrangian(φ,pcfs,stencil,vel_ext,interp,el_size,γ,γ_reinit);
  for history in optimiser
    it,Ji,_,_ = last(history)
    print_history(it,["J"=>Ji])
    write_history(history,path*"/history.csv")
    uhi = get_state(pcfs)
    write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh"=>uhi])
  end
  it,Ji,_,_ = last(optimiser.history)
  print_history(it,["J"=>Ji])
  write_history(optimiser.history,path*"/history.csv")
  uhi = get_state(pcfs)
  write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh"=>uhi];iter_mod=1)
end

main();