using Pkg; Pkg.activate()

using Gridap,GridapTopOpt
include("../embedded_measures_AD_DISABLED.jl")

function main(path="./results/UnfittedFEM_inverse_homenisation_ALM/")
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
  α_coeff = 4max_steps*γ
  vf = 0.4
  iter_mod = 1
  rm(path,force=true,recursive=true)
  mkpath(path)

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
  embedded_meas = EmbeddedMeasureCache(φh,V_φ)
  update_meas(φ) = update_embedded_measures!(φ,embedded_meas)
  get_meas(φ) = get_embedded_measures(φ,embedded_meas)

  ## Weak form
  εᴹ = (TensorValue(1.,0.,0.,0.),     # ϵᵢⱼ⁽¹¹⁾≡ϵᵢⱼ⁽¹⁾
        TensorValue(0.,0.,0.,1.),     # ϵᵢⱼ⁽²²⁾≡ϵᵢⱼ⁽²⁾
        TensorValue(0.,1/2,1/2,0.))   # ϵᵢⱼ⁽¹²⁾≡ϵᵢⱼ⁽³⁾

  a(u,v,φ,dΩ,dΩ1,dΩ2,dΓ) = ∫(C ⊙ ε(u) ⊙ ε(v) )dΩ1 + ∫((10^-6*C) ⊙ ε(u) ⊙ ε(v) )dΩ2
  l = [(v,φ,dΩ,dΩ1,dΩ2,dΓ) -> ∫(-C ⊙ εᴹ[i] ⊙ ε(v))dΩ1 + ∫(-(10^-6*C) ⊙ εᴹ[i] ⊙ ε(v))dΩ2 for i in 1:3]

  ## Optimisation functionals
  Cᴴ(r,s,u,dΩ1,dΩ2,dΓ) = ∫(C ⊙ (ε(u[r])+εᴹ[r]) ⊙ εᴹ[s])dΩ1
  dCᴴ(r,s,q,u,dΩ1,dΩ2,dΓ) = ∫(-q*(C ⊙ (ε(u[r])+εᴹ[r]) ⊙ (ε(u[s])+εᴹ[s])))dΓ
  κ(u,φ,dΩ,dΩ1,dΩ2,dΓ) = -1/4*(Cᴴ(1,1,u,dΩ1,dΩ2,dΓ)+Cᴴ(2,2,u,dΩ1,dΩ2,dΓ)+2*Cᴴ(1,2,u,dΩ1,dΩ2,dΓ))
  dκ(q,u,φ,dΩ,dΩ1,dΩ2,dΓ) = -1/4*(dCᴴ(1,1,q,u,dΩ1,dΩ2,dΓ)+dCᴴ(2,2,q,u,dΩ1,dΩ2,dΓ)+2*dCᴴ(1,2,q,u,dΩ1,dΩ2,dΓ))
  Vol(u,φ,dΩ,dΩ1,dΩ2,dΓ) = ∫(1/vol_D)dΩ1 - ∫(vf/vol_D)dΩ;
  dVol(q,u,φ,dΩ,dΩ1,dΩ2,dΓ) = ∫(-1/vol_D*q)dΓ

  a_iem = IntegrandWithEmbeddedMeasure(a,(dΩ,),update_meas)
  l_iem = map(li->IntegrandWithEmbeddedMeasure(li,(dΩ,),get_meas),l)
  J_iem = IntegrandWithEmbeddedMeasure(κ,(dΩ,),get_meas)
  dJ_iem = IntegrandWithEmbeddedMeasure(dκ,(dΩ,),get_meas)
  Vol_iem = IntegrandWithEmbeddedMeasure(Vol,(dΩ,),get_meas)
  dVol_iem = IntegrandWithEmbeddedMeasure(dVol,(dΩ,),get_meas)

  ## Finite difference solver and level set function
  ls_evo = HamiltonJacobiEvolution(FirstOrderStencil(2,Float64),model,V_φ,tol,max_steps)

  ## Setup solver and FE operators
  state_map = RepeatingAffineFEStateMap(3,a_iem,l_iem,U,V,V_φ,U_reg,φh,(dΩ,))
  pcfs = PDEConstrainedFunctionals(J_iem,[Vol_iem],state_map,analytic_dJ=dJ_iem,analytic_dC=[dVol_iem])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

  ## Optimiser
  optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;reinit_mod=5,
    γ,γ_reinit,verbose=true,constraint_names=[:Vol])
  for (it,uh,φh) in optimiser
    _Ω1,_,_Γ = get_embedded_triangulations(embedded_meas)
    if iszero(it % iter_mod)
      writevtk(_Ω1,path*"Omega_out$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh1"=>uh[1],"uh2"=>uh[2],"uh3"=>uh[3]])
      writevtk(_Γ,path*"Gamma_out$it",cellfields=["normal"=>get_normal_vector(_Γ)])
    end
    write_history(path*"/history.txt",optimiser.history)
  end
  it = get_history(optimiser).niter; uh = get_state(pcfs)
  _Ω1,_,_Γ = get_embedded_triangulations(embedded_meas)
  writevtk(_Ω1,path*"Omega_out$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh1"=>uh[1],"uh2"=>uh[2],"uh3"=>uh[3]])
  writevtk(_Γ,path*"Gamma_out$it",cellfields=["normal"=>get_normal_vector(_Γ)])
end

main()