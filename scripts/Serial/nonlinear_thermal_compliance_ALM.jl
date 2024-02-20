using Gridap, LevelSetTopOpt

"""
  (Serial) Minimum thermal compliance with augmented Lagrangian method in 2D with nonlinear diffusivity.

  Optimisation problem:
      Min J(Ω) = ∫ κ(u)*∇(u)⋅∇(u) dΩ
        Ω
    s.t., Vol(Ω) = vf,
          ⎡u∈V=H¹(Ω;u(Γ_D)=0),
          ⎣∫ κ(u)*∇(u)⋅∇(v) dΩ = ∫ v dΓ_N, ∀v∈V.
  
  In this example κ(u) = κ0*(exp(ξ*u))
""" 
function main()
  ## Parameters
  order = 1
  xmax=ymax=1.0
  prop_Γ_N = 0.4
  prop_Γ_D = 0.2
  dom = (0,xmax,0,ymax)
  el_size = (200,200)
  γ = 0.1
  γ_reinit = 0.5
  max_steps = floor(Int,minimum(el_size)/10)
  tol = 1/(10order^2)*prod(inv,minimum(el_size))
  η_coeff = 2
  α_coeff = 4
  vf = 0.4
  path = dirname(dirname(@__DIR__))*"/results/nonlinear_thermal_compliance_ALM"
  mkdir(path)

  ## FE Setup
  model = CartesianDiscreteModel(dom,el_size);
  el_Δ = get_el_Δ(model)
  f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= ymax*prop_Γ_D + eps() || 
      x[2] >= ymax-ymax*prop_Γ_D - eps()));
  f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= 
      ymax/2+ymax*prop_Γ_N/4 + eps());
  update_labels!(1,model,f_Γ_D,"Gamma_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")

  ## Triangulations and measures
  Ω = Triangulation(model)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΩ = Measure(Ω,2*order)
  dΓ_N = Measure(Γ_N,2*order)
  vol_D = sum(∫(1)dΩ)

  ## Spaces
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,0.0)
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)

  ## Create FE functions
  φh = interpolate(initial_lsf(4,0.2),V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  κ0 = 1
  ξ = -1
  κ(u) = κ0*(exp(ξ*u))
  res(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*(κ ∘ u)*∇(u)⋅∇(v))dΩ - ∫(v)dΓ_N

  ## Optimisation functionals
  J(u,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*(κ ∘ u)*∇(u)⋅∇(u))dΩ
  Vol(u,φ,dΩ,dΓ_N) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ;
  dVol(q,u,φ,dΩ,dΓ_N) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  stencil = AdvectionStencil(FirstOrderStencil(2,Float64),model,V_φ,tol,max_steps)

  ## Setup solver and FE operators
  state_map = NonlinearFEStateMap(res,U,V,V_φ,U_reg,φh,dΩ,dΓ_N)
  pcfs = PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dC=[dVol])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)
  
  ## Optimiser
  optimiser = AugmentedLagrangian(pcfs,stencil,vel_ext,φh;
    γ,γ_reinit,verbose=true,constraint_names=[:Vol])
  for (it, uh, φh) in optimiser
    write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
    write_history(path*"/history.txt",optimiser.history)
  end
  it = get_history(optimiser).niter; uh = get_state(pcfs)
  write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi)|"=>(norm ∘ ∇(φh)),"uh"=>uh];iter_mod=1)
end

main()