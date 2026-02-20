module ThermalComplianceALMTests
using Test

using Gridap, GridapTopOpt

"""
  (Serial) Minimum thermal compliance with augmented Lagrangian method in 2D.

  Optimisation problem:
      Min J(Ω) = ∫ κ*∇(u)⋅∇(u) dΩ
        Ω
    s.t., Vol(Ω) = vf,
          ⎡u∈V=H¹(Ω;u(Γ_D)=0),
          ⎣∫ κ*∇(u)⋅∇(v) dΩ = ∫ v dΓ_N, ∀v∈V.
"""

order = 1 
#function main(;order,AD)
  ## Parameters
  xmax = ymax = 1.0
  prop_Γ_N = 0.2
  prop_Γ_D = 0.2
  dom = (0,xmax,0,ymax)
  el_size = (10,10)
  γ = 0.1
  γ_reinit = 0.5
  max_steps = floor(Int,order*minimum(el_size)/10)
  tol = 1/(5*order^2)/minimum(el_size)
  κ = 1
  vf = 0.4
  η_coeff = 2
  α_coeff = 4max_steps*γ

  ## FE Setup
  model = CartesianDiscreteModel(dom,el_size);
  el_Δ = get_el_Δ(model)
  f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= ymax*prop_Γ_D + eps() ||
      x[2] >= ymax-ymax*prop_Γ_D - eps()))
  f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/2 - eps() <= x[2] <=
      ymax/2+ymax*prop_Γ_N/2 + eps())
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

  a(u,v,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(v))dΩ
  l(v,φ) = ∫(v)dΓ_N

  ## Optimisation functionals
  J(u,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(u))dΩ
  dJ(q,u,φ) = ∫(κ*∇(u)⋅∇(u)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;
  Vol(u,φ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ;
  dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  evo = FiniteDifferenceEvolver(FirstOrderStencil(2,Float64),model,V_φ;max_steps)
  reinit = FiniteDifferenceReinitialiser(FirstOrderStencil(2,Float64),model,V_φ;tol,γ_reinit)
  ls_evo = LevelSetEvolution(evo,reinit)

  ## Setup solver and FE operators
  state_map = AffineFEStateMap(a,l,U,V,V_φ)
  pcfs =
    PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dJ=dJ,analytic_dC=[dVol])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

  ## Optimiser
  optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
    γ,verbose=true,constraint_names=[:Vol])

  # Do a few iterations
  vars, state = iterate(optimiser)
  #vars, state = iterate(optimiser,state)
  #true


  using ChainRulesCore
  import GridapTopOpt: StateParamMap 

function ChainRulesCore.frule((Δself, du, dφ),
  u_to_j::StateParamMap,
  u, φ)

  F = u_to_j.F
  @show u_to_j
  U,V_φ = u_to_j.spaces
  assem_U,assem_deriv = u_to_j.assems

@show u_to_j
  ∂j∂u_vec,∂j∂φ_vec,∂F∂u = u_to_j.caches# primal
  
  
  uh = FEFunction(U,u)
  φh = FEFunction(V_φ,φ)
  j = sum(F(uh, φh))  # same as u_to_j(uh, φh)

  # directional derivative
  dj = ∂j∂φ_vec⋅dφ + ∂j∂u_vec⋅du

  return j, dj
end

nofields = ZeroTangent()

u_to_j = StateParamMap(J, state_map)
u = state.uh.free_values
φ = φh.free_values
du = u 
dφ = φ
frule((nofields, du,dφ), u_to_j, u,φ )

function φ_to_j(φ)
  u_to_j(u,φ)
end
using Zygote
∇f(φ) = Zygote.gradient(φ_to_j,φ)[1]
v = φ 
x = φ 
using ForwardDiff
ForwardDiff.derivative(α->∇f(x+α*v),0)



end