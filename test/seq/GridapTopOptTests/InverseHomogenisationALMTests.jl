module InverseHomogenisationALMTests
using Test

using Gridap, GridapTopOpt

"""
  (Serial) Maximum bulk modulus inverse homogenisation with augmented Lagrangian method in 2D.

  Optimisation problem:
      Min J(Ω) = -κ(Ω)
        Ω
    s.t., Vol(Ω) = vf,
          ⎡For unique εᴹᵢ, find uᵢ∈V=H¹ₚₑᵣ(Ω)ᵈ,
          ⎣∫ ∑ᵢ C ⊙ ε(uᵢ) ⊙ ε(vᵢ) dΩ = ∫ -∑ᵢ C ⊙ ε⁰ᵢ ⊙ ε(vᵢ) dΩ, ∀v∈V.
"""
function main(;AD)
  ## Parameters
  order = 1
  xmax,ymax=(1.0,1.0)
  dom = (0,xmax,0,ymax)
  el_size = (10,10)
  γ = 0.05
  γ_reinit = 0.5
  max_steps = floor(Int,order*minimum(el_size)/10)
  tol = 1/(5order^2)/minimum(el_size)
  C = isotropic_elast_tensor(2,1.,0.3)
  η_coeff = 2
  α_coeff = 4max_steps*γ
  vf = 0.5

  ## FE Setup
  model = CartesianDiscreteModel(dom,el_size,isperiodic=(true,true))
  el_Δ = get_el_Δ(model)
  f_Γ_D(x) = iszero(x)
  update_labels!(1,model,f_Γ_D,"origin")

  ## Triangulations and measures
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*order)
  vol_D = sum(∫(1)dΩ)

  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["origin"])
  U = TrialFESpace(V,VectorValue(0.0,0.0))
  V_reg = V_φ = TestFESpace(model,reffe_scalar)
  U_reg = TrialFESpace(V_reg)

  ## Create FE functions
  lsf_fn = x->max(initial_lsf(2,0.4)(x),initial_lsf(2,0.4;b=VectorValue(0,0.5))(x))
  φh = interpolate(lsf_fn,V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  εᴹ = (TensorValue(1.,0.,0.,0.),     # ϵᵢⱼ⁽¹¹⁾≡ϵᵢⱼ⁽¹⁾
        TensorValue(0.,0.,0.,1.),     # ϵᵢⱼ⁽²²⁾≡ϵᵢⱼ⁽²⁾
        TensorValue(0.,1/2,1/2,0.))   # ϵᵢⱼ⁽¹²⁾≡ϵᵢⱼ⁽³⁾

  a(u,v,φ) = ∫((I ∘ φ) * C ⊙ ε(u) ⊙ ε(v) )dΩ
  l = [(v,φ) -> ∫(-(I ∘ φ)* C ⊙ εᴹ[i] ⊙ ε(v))dΩ for i in 1:3]

  ## Optimisation functionals
  Cᴴ(r,s,u,φ) = ∫((I ∘ φ)*(C ⊙ (ε(u[r])+εᴹ[r]) ⊙ εᴹ[s]))dΩ
  dCᴴ(r,s,q,u,φ) = ∫(-q*(C ⊙ (ε(u[r])+εᴹ[r]) ⊙ (ε(u[s])+εᴹ[s]))*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  κ(u,φ) = -1/4*(Cᴴ(1,1,u,φ)+Cᴴ(2,2,u,φ)+2*Cᴴ(1,2,u,φ))
  dκ(q,u,φ) = -1/4*(dCᴴ(1,1,q,u,φ)+dCᴴ(2,2,q,u,φ)+2*dCᴴ(1,2,q,u,φ))
  Vol(u,φ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
  dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  ls_evo = HamiltonJacobiEvolution(FirstOrderStencil(2,Float64),model,V_φ,tol,max_steps)

  ## Setup solver and FE operators
  state_map = RepeatingAffineFEStateMap(3,a,l,U,V,V_φ,φh)
  pcfs = if AD
    PDEConstrainedFunctionals(κ,[Vol],state_map;analytic_dJ=dκ,analytic_dC=[dVol])
  else
    PDEConstrainedFunctionals(κ,[Vol],state_map)
  end

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
@test main(;AD=true)
@test main(;AD=false)

end # module