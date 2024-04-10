using Gridap, LevelSetTopOpt

"""
  (Serial) Maximum bulk modulus inverse homogenisation with augmented Lagrangian method in 2D.

  Optimisation problem:
      Min J(Ω) = -κ(Ω)
        Ω
    s.t., Vol(Ω) = vf,
          ⎡For unique εᴹᵢ, find uᵢ∈V=H¹ₚₑᵣ(Ω)ᵈ, 
          ⎣∫ ∑ᵢ C ⊙ ε(uᵢ) ⊙ ε(vᵢ) dΩ = ∫ -∑ᵢ C ⊙ ε⁰ᵢ ⊙ ε(vᵢ) dΩ, ∀v∈V.
""" 
function main()
  ## Parameters
  order = 1
  xmax,ymax=(1.0,1.0)
  dom = (0,xmax,0,ymax)
  el_size = (200,200)
  γ = 0.05
  γ_reinit = 0.5
  max_steps = floor(Int,order*minimum(el_size)/10)
  tol = 1/(5order^2)/minimum(el_size)
  C = isotropic_elast_tensor(2,1.,0.3)
  η_coeff = 2
  α_coeff = 4max_steps*γ
  vf = 0.5
  path = dirname(dirname(@__DIR__))*"/results/inverse_homenisation_ALM/"
  iter_mod = 10
  mkdir(path)

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

  a(u,v,φ,dΩ) = ∫((I ∘ φ) * C ⊙ ε(u) ⊙ ε(v) )dΩ
  l = [(v,φ,dΩ) -> ∫(-(I ∘ φ)* C ⊙ εᴹ[i] ⊙ ε(v))dΩ for i in 1:3]

  ## Optimisation functionals
  Cᴴ(r,s,u,φ,dΩ) = ∫((I ∘ φ)*(C ⊙ (ε(u[r])+εᴹ[r]) ⊙ εᴹ[s]))dΩ
  dCᴴ(r,s,q,u,φ,dΩ) = ∫(-q*(C ⊙ (ε(u[r])+εᴹ[r]) ⊙ (ε(u[s])+εᴹ[s]))*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  κ(u,φ,dΩ) = -1/4*(Cᴴ(1,1,u,φ,dΩ)+Cᴴ(2,2,u,φ,dΩ)+2*Cᴴ(1,2,u,φ,dΩ))
  dκ(q,u,φ,dΩ) = -1/4*(dCᴴ(1,1,q,u,φ,dΩ)+dCᴴ(2,2,q,u,φ,dΩ)+2*dCᴴ(1,2,q,u,φ,dΩ))
  Vol(u,φ,dΩ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
  dVol(q,u,φ,dΩ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  stencil = HamiltonJacobiEvolution(FirstOrderStencil(2,Float64),model,V_φ,tol,max_steps)

  ## Setup solver and FE operators
  state_map = RepeatingAffineFEStateMap(3,a,l,U,V,V_φ,U_reg,φh,dΩ)
  pcfs = PDEConstrainedFunctionals(κ,[Vol],state_map;analytic_dJ=dκ,analytic_dC=[dVol])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)
  
  ## Optimiser
  optimiser = AugmentedLagrangian(pcfs,stencil,vel_ext,φh;
    γ,γ_reinit,verbose=true,constraint_names=[:Vol])
  for (it,uh,φh) in optimiser
    data = ["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh))]
    iszero(it % iter_mod) && writevtk(Ω,path*"out$it",cellfields=data)
    write_history(path*"/history.txt",optimiser.history)
  end
  it = get_history(optimiser).niter; uh = get_state(pcfs)
  writevtk(Ω,path*"out$it",cellfields=["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh))])
end

main()