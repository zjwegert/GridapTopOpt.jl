module ThermalComplianceHPMTests
using Test
using Gridap, GridapTopOpt
using GridapTopOpt: WithAutoDiff, NoAutoDiff

order = 1 
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

V_φ = TestFESpace(model,reffe_scalar)#;dirichlet_tags=["Gamma_N"])
# U_φ = TrialFESpace(V_φ,-0.1)

V_φ_ = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
U_φ_ = TrialFESpace(V_φ_,-0.01)

  a(u,v,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(v))dΩ
  l(v,φ) = ∫(v)dΓ_N

  ## Optimisation functionals
  J(u,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(u))dΩ
  dJ(q,u,φ) = ∫(κ*∇(u)⋅∇(u)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;
  Vol(u,φ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ;
  dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

## Interpolation and weak form
interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  ## Setup solver and FE operators
  state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh)
  pcfs = if AD_case == :no_ad
    PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dJ=dJ,analytic_dC=[dVol])
  elseif AD_case == :with_ad
    PDEConstrainedFunctionals(J,[Vol],state_map)
  elseif AD_case == :partial_ad1
    PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dJ=dJ)
  elseif AD_case == :partial_ad2
    PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dC=[dVol])
  else
    @error "AD case not defined"
  end

## Optimisation functionals
J(u,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*κ*∇(u)⋅∇(u))dΩ
dJ(q,u,φ,dΩ,dΓ_N) = ∫(κ*∇(u)⋅∇(u)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;
Vol(u,φ,dΩ,dΓ_N) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ;
dVol(q,u,φ,dΩ,dΓ_N) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

## Finite difference solver and level set function
ls_evo = HamiltonJacobiEvolution(FirstOrderStencil(2,Float64),model,V_φ,tol,max_steps)

## Setup solver and FE operators
state_map = AffineFEStateMap(a,l,U,V,U_φ_,U_reg,φh,dΩ,dΓ_N)

import GridapTopOpt: StateParamIntegrandWithMeasure

C=[Vol]

objective = StateParamIntegrandWithMeasure(J,state_map)
constraints = map(Ci -> StateParamIntegrandWithMeasure(Ci,state_map),C)

using Zygote

φ = φh.free_values

function φ_to_jc(φ)
  u = state_map(φ)
  j = objective(u,φ)
  c = map(constrainti -> constrainti(u,φ),constraints)
  [j,c...]
end

φ_to_jc(φ)


φh.fe_space

Zygote.jacobian(φ_to_jc,φ)

φh_bg =  interpolate(φh,V_φ)


φh_bg.free_values
φh.free_values

pcf = CustomPDEConstrainedFunctionals(φ_to_jc,state_map,φh_bg)

## Hilbertian extension-regularisation problems
α = α_coeff*maximum(el_Δ)
a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

## Optimiser
optimiser = HilbertianProjection(pcf,ls_evo,vel_ext,φh_bg;
  γ,γ_reinit,verbose=true,constraint_names=[:Vol])

# Do a few iterations
vars, state = iterate(optimiser)
vars, state = iterate(optimiser,state)

end # module