using Gridap, GridapDistributed, GridapPETSc, PartitionedArrays, LevelSetTopOpt

"""
  (Serial) Inverter mechanism with augmented Lagrangian method in 2D.
  
  Ref: http://doi.org/10.1007/s00158-018-1950-2

  Optimisation problem:
      Min J(Ω) = ∫ ηᵢₙ * uₓ dΓᵢₙ + ∫ ηₒᵤₜ * uₓ dΓₒᵤₜ
        Ω
    s.t., Vol(Ω) = Vf,
          ⎡u∈V=H¹(Ω;u(Γ_D)=0)ᵈ, 
          ⎣∫ C ⊙ ε(u) ⊙ ε(v) dΩ + ∫ kₛv⋅u dΓₒᵤₜ = ∫ v⋅g dΓᵢₙ , ∀v∈V.
""" 
function main()
  ## Parameters
  order = 1;
  dom = (0,1,0,1);
  el_size = (200,200);
  γ = 0.1;
  γ_reinit = 0.5;
  max_steps = floor(Int,minimum(el_size)/10)
  tol = 1/(10order^2)*prod(inv,minimum(el_size))
  C = isotropic_2d(1.0,0.3);
  η_coeff = 2;
  α_coeff = 4;
  path = dirname(dirname(@__DIR__))*"/results/testing_inverter_AD"

  η_in = 2;
  η_out = 1;
  ks = 0.01;
  g = VectorValue(1,0);

  ## FE Setup
  model = CartesianDiscreteModel(dom,el_size);
  Δ = get_Δ(model)
  f_Γ_in(x) = (x[1] ≈ 0.0) && 0.47 - eps() <= x[2] <= 0.53 + eps() ? true : false;
  f_Γ_out(x) = (x[1] ≈ 1.0) && 0.43 - eps() <= x[2] <= 0.57 + eps() ? true : false;
  f_Γ_D(x) = x[1] ≈ 0.0 && (x[2] <= 0.1 || x[2] >= 0.9)  ? true : false;
  update_labels!(1,model,f_Γ_in,"Gamma_in")
  update_labels!(2,model,f_Γ_out,"Gamma_out")
  update_labels!(3,model,f_Γ_D,"Gamma_D")

  ## Triangulations and measures
  Ω = Triangulation(model)
  Γ_in = BoundaryTriangulation(model,tags="Gamma_in")
  Γ_out = BoundaryTriangulation(model,tags="Gamma_out")
  dΩ = Measure(Ω,2order)
  dΓ_in = Measure(Γ_in,2order)
  dΓ_out = Measure(Γ_out,2order)
  vol_D = sum(∫(1)dΩ)

  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,VectorValue(0.0,0.0))
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_in","Gamma_out"])
  U_reg = TrialFESpace(V_reg,[0,0])

  ## Create FE functions
  φh = interpolate(gen_lsf(4,0.2),V_φ);
  φ = get_free_dof_values(φh)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ),ϵₘ=10^-8)
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  a(u,v,φ,dΩ,dΓ_in,dΓ_out) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(v)))dΩ + ∫(ks*(u⋅v))dΓ_out
  l(v,φ,dΩ,dΓ_in,dΓ_out) = ∫(v⋅g)dΓ_in
  res(u,v,φ,dΩ,dΓ_in,dΓ_out) = a(u,v,φ,dΩ,dΓ_in,dΓ_out) - l(v,φ,dΩ,dΓ_in,dΓ_out)

  ## Optimisation functionals
  J = (u,φ,dΩ,dΓ_in,dΓ_out) -> ∫(η_in*u⋅VectorValue(1,0))dΓ_in + 
                               ∫(η_out*u⋅VectorValue(1,0))dΓ_out

  function dJ(q,u,φ,dΩ,dΓ_in,dΓ_out)
    a_adjoint(λ,v) = ∫((I ∘ φ)*(C ⊙ ε(λ) ⊙ ε(v)))dΩ + ∫(ks*(λ⋅v))dΓ_out
    l_adjoint(v) = ∫(-η_in*v⋅VectorValue(1,0))dΓ_in + ∫(-η_out*v⋅VectorValue(1,0))dΓ_out

    op = AffineFEOperator(a_adjoint,l_adjoint,U,V)
    λh = solve(op)

    ∫(q*(C ⊙ ε(u) ⊙ ε(λh))*(DH ∘ φh))dΩ
  end

  Vol = (u,φ,dΩ,dΓ_in,dΓ_out) -> ∫(((ρ ∘ φ) - 0.5)/vol_D)dΩ;
  dVol = (q,u,φ,dΩ,dΓ_in,dΓ_out) -> ∫(1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  stencil = AdvectionStencil(FirstOrderStencil(2,Float64),model,V_φ,Δ./order,max_steps,tol)
  reinit!(stencil,φ,γ_reinit)

  ## Setup solver and FE operators
  state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh,dΩ,dΓ_in,dΓ_out)
  pcfs = PDEConstrainedFunctionals(J,[Vol,J],state_map,analytic_dC=[dVol,dJ])

  J_val,C_val,dJ,dC = Gridap.evaluate!(pcfs,φ)

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)
  
  ## Optimiser
  make_dir(path)

  writevtk(Ω,"$path/TestVariational",cellfields=["φh"=>φh,"dJ"=>FEFunction(U_reg,dJ),"dJ_analytic"=>FEFunction(U_reg,dC[2])])

  maximum(abs,dJ-dC[2]),maximum(abs,dJ-dC[2])/maximum(abs,dC[2])
end

main();