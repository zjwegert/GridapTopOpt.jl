using Gridap, GridapDistributed, GridapPETSc, PartitionedArrays, LSTO_Distributed

"""
  (Serial) Minimum thermal compliance with Lagrangian method in 2D with nonlinear diffusivity.

  Optimisation problem:
      Min J(Ω) = ∫ D(u)*∇(u)⋅∇(u) + ξ dΩ
        Ω
    s.t., ⎡u∈V=H¹(Ω;u(Γ_D)=0),
          ⎣∫ D(u)*∇(u)⋅∇(v) dΩ = ∫ v dΓ_N, ∀v∈V.
  
  In this example D(u) = D0*(exp(ξ*u))
""" 
function main()
  ## Parameters
  order = 1;
  xmax=ymax=1.0
  prop_Γ_N = 0.4;
  prop_Γ_D = 0.2
  dom = (0,xmax,0,ymax);
  el_size = (200,200);
  γ = 0.1;
  γ_reinit = 0.5;
  max_steps = floor(Int,minimum(el_size)/10)
  tol = 1/(order^2*10)*prod(inv,minimum(el_size)) # <- We can do better than this I think
  η_coeff = 2;
  α_coeff = 4;
  path = "./Results/main_nonlinear"

  ## FE Setup
  model = CartesianDiscreteModel(dom,el_size);
  Δ = get_Δ(model)
  f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= ymax*prop_Γ_D + eps() || 
    x[2] >= ymax-ymax*prop_Γ_D - eps())) ? true : false;
  f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= 
    ymax/2+ymax*prop_Γ_N/4 + eps()) ? true : false;
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
  φh = interpolate(gen_lsf(4,0.2),V_φ);
  φ = get_free_dof_values(φh)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  D0 = 1;
  ξ = -1;
  D(u) = D0*(exp(ξ*u));

  res(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*(D ∘ u)*∇(u)⋅∇(v))dΩ - ∫(v)dΓ_N

  ## Optimisation functionals
  ξ = 0.2;
  J = (u,φ,dΩ,dΓ_N) -> ∫((I ∘ φ)*(D ∘ u)*∇(u)⋅∇(u) + ξ*(ρ ∘ φ))dΩ

  ## Finite difference solver and level set function
  stencil = AdvectionStencil(FirstOrderStencil(2,Float64),model,V_φ,Δ./order,max_steps,tol)
  reinit!(stencil,φ,γ_reinit)

  ## Setup solver and FE operators
  state_map = NonlinearFEStateMap(res,U,V,V_φ,U_reg,φh,dΩ,dΓ_N)
  pcfs = PDEConstrainedFunctionals(J,state_map)

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)
  
  ## Optimiser
  make_dir(path)
  _conv_cond = t->LSTO_Distributed.conv_cond(t;coef=1/50);
  optimiser = AugmentedLagrangian(φ,pcfs,stencil,vel_ext,interp,el_size,γ,γ_reinit;conv_criterion=_conv_cond);
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