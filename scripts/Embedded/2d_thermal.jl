using Pkg; Pkg.activate()

using Gridap,GridapTopOpt
include("embedded_measures.jl")

function main()
  path="./results/UnfittedFEM_thermal_compliance_ALM/"
  n = 200
  order = 1
  γ = 0.1
  γ_reinit = 0.5
  max_steps = floor(Int,order*minimum(n)/10)
  tol = 1/(1*order^2)/minimum(n)
  vf = 0.4
  α_coeff = 4max_steps*γ
  iter_mod = 1

  model = CartesianDiscreteModel((0,1,0,1),(n,n));
  el_Δ = get_el_Δ(model)
  f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= 0.2 + eps() || x[2] >= 0.8 - eps()))
  f_Γ_N(x) = (x[1] ≈ 1 && 0.4 - eps() <= x[2] <= 0.6 + eps())
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

  φh = interpolate(x->-cos(4π*x[1])*cos(4*pi*x[2])/4-0.2/4,V_φ)
  embedded_meas = EmbeddedMeasureCache(φh,V_φ)
  update_meas(φ) = update_embedded_measures!(φ,embedded_meas)
  get_meas(φ) = get_embedded_measures(φ,embedded_meas)

  ## Weak form
  a(u,v,φ,dΩ,dΓ_N,dΩ1,dΩ2,dΓ) = ∫(∇(u)⋅∇(v))dΩ1 + ∫(10^-6*∇(u)⋅∇(v))dΩ2
  l(v,φ,dΩ,dΓ_N,dΩ1,dΩ2,dΓ) = ∫(v)dΓ_N

  ## Optimisation functionals
  J(u,φ,dΩ,dΓ_N,dΩ1,dΩ2,dΓ) = ∫(∇(u)⋅∇(u))dΩ1
  dJ(q,u,φ,dΩ,dΓ_N,dΩ1,dΩ2,dΓ) = ∫(∇(u)⋅∇(u)*q)dΓ;
  Vol(u,φ,dΩ,dΓ_N,dΩ1,dΩ2,dΓ) = ∫(1/vol_D)dΩ1 - ∫(vf)dΩ;
  dVol(q,u,φ,dΩ,dΓ_N,dΩ1,dΩ2,dΓ) = ∫(-1/vol_D*q)dΓ

  ## IntegrandWithEmbeddedMeasure
  a_iem = IntegrandWithEmbeddedMeasure(a,(dΩ,dΓ_N),update_meas)
  l_iem = IntegrandWithEmbeddedMeasure(l,(dΩ,dΓ_N),update_meas)
  J_iem = IntegrandWithEmbeddedMeasure(J,(dΩ,dΓ_N),update_meas)
  dJ_iem = IntegrandWithEmbeddedMeasure(dJ,(dΩ,dΓ_N),update_meas)
  Vol_iem = IntegrandWithEmbeddedMeasure(Vol,(dΩ,dΓ_N),update_meas)
  dVol_iem = IntegrandWithEmbeddedMeasure(dVol,(dΩ,dΓ_N),update_meas)

  ## Evolution Method
  ls_evo = HamiltonJacobiEvolution(FirstOrderStencil(2,Float64),model,V_φ,tol,max_steps)

  ## Setup solver and FE operators
  state_map = AffineFEStateMap(a_iem,l_iem,U,V,V_φ,U_reg,φh,(dΩ,dΓ_N))
  pcfs = PDEConstrainedFunctionals(J_iem,[Vol_iem],state_map,analytic_dJ=dJ_iem,analytic_dC=[dVol_iem])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

  ## Optimiser
  rm(path,force=true,recursive=true)
  mkpath(path)
  optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
    γ,γ_reinit,verbose=true,constraint_names=[:Vol])
  for (it,uh,φh) in optimiser
    dΩ1,_,dΓ = get_meas(φh)
    if iszero(it % iter_mod)
      writevtk(dΩ1.quad.trian,path*"Omega_out$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
      writevtk(dΓ.quad.trian,path*"Gamma_out$it",cellfields=["normal"=>get_normal_vector(dΓ.quad.trian)])
    end
    write_history(path*"/history.txt",optimiser.history)
  end
  it = get_history(optimiser).niter; uh = get_state(pcfs)
  dΩ1,_,dΓ = get_meas(φh)
  writevtk(dΩ1.quad.trian,path*"Omega_out$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
  writevtk(dΓ.quad.trian,path*"Gamma_out$it",cellfields=["normal"=>get_normal_vector(dΓ.quad.trian)])
end

main()