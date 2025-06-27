module NeohookAnalyticJacALMTests
using Test

using Gridap, GridapTopOpt

"""
  (Serial) Minimum hyperelastic (neohookean) compliance with augmented Lagrangian method in 2D.
"""
function main()
  ## Parameters
  order = 1
  xmax=ymax=1.0
  prop_Γ_N = 0.2
  dom = (0,xmax,0,ymax)
  el_size = (10,10)
  γ = 0.1
  γ_reinit = 0.5
  max_steps = floor(Int,order*minimum(el_size)/10)
  tol = 1/(5order^2)/minimum(el_size)
  η_coeff = 2
  α_coeff = 4max_steps*γ
  vf = 0.6

  ## FE Setup
  model = CartesianDiscreteModel(dom,el_size);
  el_Δ = get_el_Δ(model)
  f_Γ_D(x) = (x[1] ≈ 0.0)
  f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/2 - eps() <= x[2] <= ymax/2+ymax*prop_Γ_N/2 + eps())
  update_labels!(1,model,f_Γ_D,"Gamma_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")

  ## Triangulations and measures
  Ω = Triangulation(model)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΩ = Measure(Ω,2*order)
  dΓ_N = Measure(Γ_N,2*order)
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
  φh = interpolate(initial_lsf(4,0.2),V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  ## Material properties
  _E = 1000
  ν = 0.3
  μ, λ = _E/(2*(1 + ν)), _E*ν/((1 + ν)*(1 - ν))
  g = VectorValue(0,-20)

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

  function dS(∇du,∇u)
    Cinv = inv(C(F(∇u)))
    _dE = dE(∇du,∇u)
    λ*(Cinv⊙_dE)*Cinv + 2*(μ-λ*log(J(F(∇u))))*Cinv⋅_dE⋅(Cinv')
  end

  # Cauchy stress tensor and residual
  σ(∇u) = (1.0/J(F(∇u)))*F(∇u)⋅S(∇u)⋅(F(∇u))'
  res(u,v,φ) = ∫( (I ∘ φ)*((dE∘(∇(v),∇(u))) ⊙ (S∘∇(u))) )*dΩ - ∫(g⋅v)dΓ_N
  jac_mat(u,du,v,φ) =  ∫( (I ∘ φ)*(dE∘(∇(v),∇(u))) ⊙ (dS∘(∇(du),∇(u))) )*dΩ
  jac_geo(u,du,v,φ) = ∫( (I ∘ φ)*∇(v) ⊙ ( (S∘∇(u))⋅∇(du) ) )*dΩ
  jac(u,du,v,φ) = jac_mat(u,du,v,φ) + jac_geo(u,du,v,φ)

  ## Optimisation functionals
  J(u,φ) = ∫((I ∘ φ)*((dE∘(∇(u),∇(u))) ⊙ (S∘∇(u))))dΩ
  Vol(u,φ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ;
  dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  ls_evo = HamiltonJacobiEvolution(FirstOrderStencil(2,Float64),model,V_φ,tol,max_steps)

  ## Setup solver and FE operators
  state_map = NonlinearFEStateMap(res,U,V,V_φ,φh;jac)
  pcfs = PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dC=[dVol])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

  ## Optimiser
  optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
    γ,γ_reinit,verbose=true,constraint_names=[:Vol])

  # Do a few iterations
  vars, state = iterate(optimiser)
  vars, state = iterate(optimiser,state)
  true
end

# Test that these run successfully
@test main()

end # module