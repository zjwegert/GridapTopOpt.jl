using Gridap, LSTO_Distributed

"""
  (Serial) Inverter mechanism with Hilbertian projection method in 2D.

  Optimisation problem:
      Min J(Ω) = ηᵢₙ*∫ u⋅e₁ dΓᵢₙ/Vol(Γᵢₙ)
        Ω
    s.t., Vol(Ω) = vf,
            C(Ω) = 0, 
          ⎡u∈V=H¹(Ω;u(Γ_D)=0)ᵈ, 
          ⎣∫ C ⊙ ε(u) ⊙ ε(v) dΩ + ∫ kₛv⋅u dΓₒᵤₜ = ∫ v⋅g dΓᵢₙ , ∀v∈V.
        
    where C(Ω) = ∫ -u⋅e₁-δₓ dΓₒᵤₜ/Vol(Γₒᵤₜ). We assume symmetry in the problem to aid
     convergence.
""" 
function main()
  ## Parameters
  order = 1
  dom = (0,1,0,0.5)
  el_size = (200,100)
  γ = 0.05
  γ_reinit = 0.5
  max_steps = floor(Int,minimum(el_size)/10)
  tol = 1/(10order^2)*prod(inv,minimum(el_size))
  C = isotropic_2d(1.0,0.3)
  η_coeff = 2
  α_coeff = 4
  vf = 0.4
  δₓ = 0.75
  ks = 0.01
  g = VectorValue(1,0)
  path = dirname(dirname(@__DIR__))*"/results/inverter_HPM"
  
  ## FE Setup
  model = CartesianDiscreteModel(dom,el_size)
  Δ = get_Δ(model)
  f_Γ_in(x) = (x[1] ≈ 0.0) && (x[2] <= 0.03 + eps())
  f_Γ_out(x) = (x[1] ≈ 1.0) && (x[2] <= 0.07 + eps())
  f_Γ_D(x) = (x[1] ≈ 0.0) && (x[2] >= 0.4)
  f_Γ_D2(x) = (x[2] ≈ 0.0)
  update_labels!(1,model,f_Γ_in,"Gamma_in")
  update_labels!(2,model,f_Γ_out,"Gamma_out")
  update_labels!(3,model,f_Γ_D,"Gamma_D")
  update_labels!(4,model,f_Γ_D2,"SymLine")

  ## Triangulations and measures
  Ω = Triangulation(model)
  Γ_in = BoundaryTriangulation(model,tags="Gamma_in")
  Γ_out = BoundaryTriangulation(model,tags="Gamma_out")
  dΩ = Measure(Ω,2order)
  dΓ_in = Measure(Γ_in,2order)
  dΓ_out = Measure(Γ_out,2order)
  vol_D = sum(∫(1)dΩ)
  vol_Γ_in = sum(∫(1)dΓ_in)
  vol_Γ_out = sum(∫(1)dΓ_out)

  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D","SymLine"],
    dirichlet_masks=[(true,true),(false,true)])
  U = TrialFESpace(V,[VectorValue(0.0,0.0),VectorValue(0.0,0.0)])
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_in","Gamma_out"])
  U_reg = TrialFESpace(V_reg,[0,0])

  ## Create FE functions
  lsf_fn(x) = max(gen_lsf(6,0.2)(x),-sqrt((x[1]-1)^2+(x[2]-0.5)^2)+0.2)
  φh = interpolate(lsf_fn,V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  a(u,v,φ,dΩ,dΓ_in,dΓ_out) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(v)))dΩ + ∫(ks*(u⋅v))dΓ_out
  l(v,φ,dΩ,dΓ_in,dΓ_out) = ∫(v⋅g)dΓ_in

  ## Optimisation functionals
  e₁ = VectorValue(1,0)
  J(u,φ,dΩ,dΓ_in,dΓ_out) = ∫((u⋅e₁)/vol_Γ_in)dΓ_in
  Vol(u,φ,dΩ,dΓ_in,dΓ_out) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
  dVol(q,u,φ,dΩ,dΓ_in,dΓ_out) = ∫(1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  UΓ_out(u,φ,dΩ,dΓ_in,dΓ_out) = ∫((u⋅-e₁-δₓ)/vol_Γ_out)dΓ_out

  ## Finite difference solver and level set function
  stencil = AdvectionStencil(FirstOrderStencil(2,Float64),model,V_φ,tol,max_steps)
  reinit!(stencil,φh,γ_reinit)

  ## Setup solver and FE operators
  state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh,dΩ,dΓ_in,dΓ_out)
  pcfs = PDEConstrainedFunctionals(J,[Vol,UΓ_out],state_map,analytic_dC=[dVol,nothing])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(Δ)
  a_hilb(p,q) = ∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)
  
  ## Optimiser
  make_dir(path)
  optimiser = HilbertianProjection(pcfs,stencil,vel_ext,φh;
    γ,γ_reinit,α_min=0.5,verbose=true,constraint_names=[:Vol,:UΓ_out])
  for (it,uh,φh) in optimiser
    write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh"=>uh])
    write_history(path*"/history.txt",optimiser.history)
  end
  it = optimiser.history.niter; uh = get_state(optimiser.problem)
  write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh"=>uh];iter_mod=1)
end

main()