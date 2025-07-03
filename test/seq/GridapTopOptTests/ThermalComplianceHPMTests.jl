module ThermalComplianceHPMTests
using Test

using Gridap, GridapTopOpt
using GridapTopOpt: WithAutoDiff, NoAutoDiff
"""
  (Serial) Minimum thermal compliance with augmented Lagrangian method in 2D.

  Optimisation problem:
      Min J(Ω) = ∫ κ*∇(u)⋅∇(u) dΩ
        Ω
    s.t., Vol(Ω) = vf,
          ⎡u∈V=H¹(Ω;u(Γ_D)=0),
          ⎣∫ κ*∇(u)⋅∇(v) dΩ = ∫ v dΓ_N, ∀v∈V.
"""
function main(;order,AD_case)
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
  ls_evo = HamiltonJacobiEvolution(FirstOrderStencil(2,Float64),model,V_φ,tol,max_steps)

  ## Setup solver and FE operators
  state_map = AffineFEStateMap(a,l,U,V,V_φ,φh)
  pcfs = if AD_case == :no_ad
    PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dJ=dJ,analytic_dC=[dVol])
  elseif AD_case == :with_ad
    PDEConstrainedFunctionals(J,[Vol],state_map)
  elseif AD_case == :partial_ad1
    PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dJ=dJ)
  elseif AD_case == :partial_ad2
    PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dC=[dVol])
  elseif AD_case == :custom_pcf
    objective = GridapTopOpt.StateParamMap(J,state_map)
    constraints = map(Ci -> GridapTopOpt.StateParamIntegrandWithMeasure(Ci,state_map),[Vol])
    function φ_to_jc(φ)
      u = state_map(φ)
      j = objective(u,φ)
      c = map(constraint -> constraint(u,φ),constraints)
      [j,c...]
    end
    CustomPDEConstrainedFunctionals(φ_to_jc,length(constraints);state_map)
  elseif AD_case == :custom_pcf_analyticVol
    objective = GridapTopOpt.StateParamMap(J,state_map)
    constraints = map(Ci -> GridapTopOpt.StateParamIntegrandWithMeasure(Ci,state_map),[Vol])
    function φ_to_jc2(φ)
      u = state_map(φ)
      j = objective(u,φ)
      c = map(constraint -> constraint(u,φ),constraints)
      [j,c...]
    end
    function analytic_dC1!(dC,φ)
      println("!!         analytic_dC1! called")
      φh = FEFunction(V_φ,φ)
      uh = get_state(state_map)
      _dC(q) = dVol(q,uh,φh)
      Gridap.FESpaces.assemble_vector!(_dC,dC,V_φ)
    end
    CustomPDEConstrainedFunctionals(φ_to_jc2,length(constraints);state_map,
      analytic_dC=[analytic_dC1!])
  else
    @error "AD case not defined"
  end

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

  ## Optimiser
  optimiser = HilbertianProjection(pcfs,ls_evo,vel_ext,φh;
    γ,γ_reinit,verbose=true,constraint_names=[:Vol])

  AD_case ∈ (:with_ad,:partial_ad1,:partial_ad2) && @test typeof(optimiser) <: HilbertianProjection{WithAutoDiff}
  AD_case ∈ (:no_ad,) && @test typeof(optimiser) <: HilbertianProjection{NoAutoDiff}

  # Do a few iterations
  vars, state = iterate(optimiser)
  vars, state = iterate(optimiser,state)
  true
end

# Test that these run successfully
@test main(;order=1,AD_case=:with_ad)
@test main(;order=1,AD_case=:partial_ad1)
@test main(;order=1,AD_case=:partial_ad2)
@test main(;order=1,AD_case=:no_ad)
@test main(;order=1,AD_case=:custom_pcf)
@test main(;order=1,AD_case=:custom_pcf_analyticVol)

end # module