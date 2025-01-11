using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers
using GridapGmsh
using GridapTopOpt

import GridapSolvers.SolverInterfaces: SOLVER_DIVERGED_MAXITER, SOLVER_DIVERGED_BREAKDOWN

Base.abs(x::VectorValue) = VectorValue(abs.(x.data))
Base.sign(x::VectorValue) = VectorValue(sign.(x.data))

function GridapTopOpt.forward_solve!(φ_to_u::NonlinearFEStateMap,φh)
  U, V, _, _ = φ_to_u.spaces
  nls, nls_cache, x, assem_U = φ_to_u.fwd_caches

  xprev = copy(x)

  res(u,v) = φ_to_u.res(u,v,φh)
  jac(u,du,v) = φ_to_u.jac(u,du,v,φh)
  op = Gridap.FESpaces.get_algebraic_operator(FEOperator(res,jac,U,V,assem_U))
  solve!(x,nls,op,nls_cache)
  if GridapTopOpt._get_solver_flag(nls.log) == SOLVER_DIVERGED_MAXITER
    g0 = VectorValue(0,-20)
    copy!(x,xprev)
    _ramp_solve!(0,x,xprev,2/3*g0,g0,0g0,φ_to_u,φh,U,V,assem_U,nls,nls_cache,0,0,false)
  end
  return x
end

function _ramp_solve!(ramp_call,x,xprev,g,g0,gprev,φ_to_u,φh,U,V,assem_U,nls,nls_cache,ramp_div_it,conv_it,has_prev_converged)
  if ramp_call > 50 && GridapTopOpt._get_solver_flag(nls.log) == SOLVER_DIVERGED_MAXITER
    error("Ramp solver did not converge")
  end

  println("------- RAMP (Called $ramp_call times): solving with g_ramp = ",g,". Last converged with ", gprev)
  res(u,v) = φ_to_u.res(u,v,φh,g)
  jac(u,du,v) = φ_to_u.jac(u,du,v,φh)
  op = Gridap.FESpaces.get_algebraic_operator(FEOperator(res,jac,U,V,assem_U))
  solve!(x,nls,op,nls_cache)

  if GridapTopOpt._get_solver_flag(nls.log) == SOLVER_DIVERGED_MAXITER
    if ~has_prev_converged
      _ramp_solve!(ramp_call+1,x,xprev,1/2^(ramp_div_it+1)*g0,g0,gprev,φ_to_u,φh,U,V,assem_U,nls,nls_cache,ramp_div_it+1,0,false)
    else
      copy!(x,xprev)
      # _gnew = min((1 + 1/2^(ramp_div_it+2))*abs(gprev),gprev+abs(gprev-g0)/2^(ramp_div_it+1))
      _gnew = gprev + abs(gprev-g)/2^(ramp_div_it+1).*sign(gprev)
      _ramp_solve!(ramp_call+1,x,xprev,_gnew,g0,gprev,φ_to_u,φh,U,V,assem_U,nls,nls_cache,ramp_div_it+1,0,true)
    end
  elseif g ≈ g0
    return x
  else
    g_new = min((1+0.005*2^(conv_it+1))*abs(g),abs(g0)).*sign(g0)
    copy!(xprev,x)
    _ramp_solve!(ramp_call+1,x,xprev,g_new,g0,g,φ_to_u,φh,U,V,assem_U,nls,nls_cache,0,conv_it+1,true)
  end
end

function main_lin_elast(n,path="./results/CutFEM_hyperelastic_compliance_neohook_$(n)_withALParams/linear_elastic/")
  ## Parameters
  order = 1
  xmax,ymax=2.0,1.0
  prop_Γ_N = 0.2
  el_size = (2n,n)
  γ = 0.1
  max_steps = floor(Int,order*n/10)
  α_coeff = max_steps*γ
  vf = 0.4
  iter_mod = 1
  mkpath(path)

  ## FE Setup
  _model = CartesianDiscreteModel((0,xmax,0,ymax),el_size)
  base_model = UnstructuredDiscreteModel(_model)
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = ref_model.model
  el_Δ = get_el_Δ(_model)
  h = maximum(el_Δ)
  h_refine = maximum(el_Δ)/2
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
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)

  reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)

  function build_spaces(Ω_act)
    V = TestFESpace(Ω_act,reffe;dirichlet_tags=["Gamma_D"])
    U = TrialFESpace(V,VectorValue(0.0,0.0))
    return U,V
  end

  ## Create FE functions
  φh = interpolate(initial_lsf(4,0.1),V_φ)
  Ωs = EmbeddedCollection(model,φh) do cutgeo,_
    Ωin = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_IN),V_φ)
    Ωout = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
    Γ = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
    Γg = GhostSkeleton(cutgeo)
    Ω_act = Triangulation(cutgeo,ACTIVE)
    (;
      :Ωin   => Ωin,
      :dΩin  => Measure(Ωin,2*order),
      :Ωout  => Ωout,
      :dΩout => Measure(Ωout,2*order),
      :Γg    => Γg,
      :dΓg   => Measure(Γg,2*order),
      :n_Γg  => get_normal_vector(Γg),
      :Γ     => Γ,
      :dΓ    => Measure(Γ,2*order),
      :Ω_act => Ω_act,
      :χ   => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_D"])
    )
  end

  ## Material properties
  _E = 1000
  _v = 0.3
  μ, λ = _E/(2*(1 + _v)), _E*_v/((1 + _v)*(1 - _v))
  g = VectorValue(0,-20)

  # Stabilization
  α_Gd = 1e-7
  k_d = 1.0
  γ_Γg(h) = α_Gd*(λ + μ)*h^3

  # Residual
  σ(ε) = λ*tr(ε)*one(ε) + 2*μ*ε
  a_Ω(u,v) = ε(u) ⊙ (σ ∘ ε(v)) # Elasticity
  j_Γg(u,v,h::Function) = mean(γ_Γg ∘ h)*jump(Ωs.n_Γg ⋅ ∇(u)) ⋅ jump(Ωs.n_Γg ⋅ ∇(v)) # Ghost penalty
  j_Γg(u,v,h::Number) = γ_Γg(h)*jump(Ωs.n_Γg ⋅ ∇(u)) ⋅ jump(Ωs.n_Γg ⋅ ∇(v))
  v_χ(u,v) = k_d*Ωs.χ*u⋅v # Isolated volume term

  a(u,v,φ) = ∫(a_Ω(u,v))Ωs.dΩin +
    ∫(j_Γg(u,v,h))Ωs.dΓg + ∫(v_χ(u,v))Ωs.dΩin
  l(v,φ) = ∫(g⋅v)dΓ_N

  ## Optimisation functionals
  Obj(u,φ) = ∫(a_Ω(u,u))Ωs.dΩin
  Vol(u,φ) = ∫(1/vol_D)Ωs.dΩin - ∫(vf/vol_D)dΩ
  dVol(q,u,φ) = ∫(-1/vol_D*q/(norm ∘ (∇(φ))))Ωs.dΓ

  ## Finite difference solver and level set function
  evo = CutFEMEvolve(V_φ,Ωs,dΩ,h;max_steps,γg=0.01)
  reinit1 = StabilisedReinit(V_φ,Ωs,dΩ,h;stabilisation_method=ArtificialViscosity(1.0))
  reinit2 = StabilisedReinit(V_φ,Ωs,dΩ,h;stabilisation_method=GridapTopOpt.InteriorPenalty(V_φ,γg=1.0))
  reinit = GridapTopOpt.MultiStageStabilisedReinit([reinit1,reinit2])
  ls_evo = UnfittedFEEvolution(evo,reinit)

  ## Setup solver and FE operators
  state_collection = GridapTopOpt.EmbeddedCollection_in_φh(model,φh) do _φh
    update_collection!(Ωs,_φh)
    U,V = build_spaces(Ωs.Ω_act)
    state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh)
    (;
      :state_map => state_map,
      :J => GridapTopOpt.StateParamMap(Obj,state_map),
      :C => map(Ci -> GridapTopOpt.StateParamMap(Ci,state_map),[Vol,])
    )
  end
  pcfs = EmbeddedPDEConstrainedFunctionals(state_collection;analytic_dC=(dVol,))

  ## Hilbertian extension-regularisation problems
  α = (α_coeff*h_refine/order)^2
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

  ## Optimiser
  converged(m) = GridapTopOpt.default_al_converged(
    m;
    L_tol = 0.01*h_refine,
    C_tol = 0.01
  )
  optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
    γ,verbose=true,constraint_names=[:Vol],converged,maxiter=100)
  for (it,uh,φh) in optimiser
    if iszero(it % iter_mod)
      data = ["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh]
      writevtk(Ω,path*"Omega_$it",cellfields=data)
      writevtk(Ωs.Ωin,path*"Omega_in_$it",cellfields=data)
    end
    write_history(path*"/history.txt",optimiser.history)
  end
  it = get_history(optimiser).niter; uh = get_state(pcfs)
  data = ["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh]
  writevtk(Ω,path*"Omega_$it",cellfields=data)
  writevtk(Ωs.Ωin,path*"Omega_in_$it",cellfields=data)

  return get_free_dof_values(φh),get_history(optimiser)
end

function main_neo(n,φ=nothing,λelast=nothing,Λelast=nothing;path="./results/CutFEM_hyperelastic_compliance_neohook_$(n)_withALParams/g=20/")
  ## Parameters
  order = 1
  xmax,ymax=2.0,1.0
  prop_Γ_N = 0.2
  el_size = (2n,n)
  γ = 0.1
  max_steps = floor(Int,order*n/10)
  α_coeff = max_steps*γ
  vf = 0.4
  iter_mod = 1
  mkpath(path)

  ## FE Setup
  _model = CartesianDiscreteModel((0,xmax,0,ymax),el_size)
  base_model = UnstructuredDiscreteModel(_model)
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = ref_model.model
  el_Δ = get_el_Δ(_model)
  h = maximum(el_Δ)
  h_refine = maximum(el_Δ)/2
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
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)

  reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)

  function build_spaces(Ω_act)
    V = TestFESpace(Ω_act,reffe;dirichlet_tags=["Gamma_D"])
    U = TrialFESpace(V,VectorValue(0.0,0.0))
    return U,V
  end

  ## Create FE functions
  if φ isa Nothing
    φh = interpolate(initial_lsf(4,0.1),V_φ)
  else
    φh = FEFunction(V_φ,φ)
  end
  Ωs = EmbeddedCollection(model,φh) do cutgeo,_
    Ωin = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_IN),V_φ)
    Ωout = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
    Γ = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
    Γg = GhostSkeleton(cutgeo)
    Ω_act = Triangulation(cutgeo,ACTIVE)
    (;
      :Ωin   => Ωin,
      :dΩin  => Measure(Ωin,2*order),
      :Ωout  => Ωout,
      :dΩout => Measure(Ωout,2*order),
      :Γg    => Γg,
      :dΓg   => Measure(Γg,2*order),
      :n_Γg  => get_normal_vector(Γg),
      :Γ     => Γ,
      :dΓ    => Measure(Γ,2*order),
      :Ω_act => Ω_act,
      :χ   => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_D"])
    )
  end

  ## Material properties
  _E = 1000
  _v = 0.3
  μ, λ = _E/(2*(1 + _v)), _E*_v/((1 + _v)*(1 - _v))
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

  # Stabilization
  α_Gd = 1e-7
  k_d = 1.0
  γ_Γg(h) = α_Gd*(λ + μ)*h^3

  # Cauchy stress tensor and residual
  σ(∇u) = (1.0/J(F(∇u)))*F(∇u)⋅S(∇u)⋅(F(∇u))'
  a_Ω(u,v) = (dE∘(∇(v),∇(u))) ⊙ (S∘∇(u)) # Elasticity
  j_Γg(u,v,h::Function) = mean(γ_Γg ∘ h)*jump(Ωs.n_Γg ⋅ ∇(u)) ⋅ jump(Ωs.n_Γg ⋅ ∇(v)) # Ghost penalty
  j_Γg(u,v,h::Number) = γ_Γg(h)*jump(Ωs.n_Γg ⋅ ∇(u)) ⋅ jump(Ωs.n_Γg ⋅ ∇(v))
  v_χ(u,v) = k_d*Ωs.χ*u⋅v # Isolated volume term

  res(u,v,φ) = ∫(a_Ω(u,v))Ωs.dΩin - ∫(g⋅v)dΓ_N +
    ∫(j_Γg(u,v,h))Ωs.dΓg + ∫(v_χ(u,v))Ωs.dΩin
  res(u,v,φ,g) = ∫(a_Ω(u,v))Ωs.dΩin - ∫(g⋅v)dΓ_N +
    ∫(j_Γg(u,v,h))Ωs.dΓg + ∫(v_χ(u,v))Ωs.dΩin

  ## Optimisation functionals
  Obj(u,φ) = ∫((dE∘(∇(u),∇(u))) ⊙ (S∘∇(u)))Ωs.dΩin
  Vol(u,φ) = ∫(1/vol_D)Ωs.dΩin - ∫(vf/vol_D)dΩ
  dVol(q,u,φ) = ∫(-1/vol_D*q/(norm ∘ (∇(φ))))Ωs.dΓ

  ## Finite difference solver and level set function
  evo = CutFEMEvolve(V_φ,Ωs,dΩ,h;max_steps,γg=0.01)
  reinit1 = StabilisedReinit(V_φ,Ωs,dΩ,h;stabilisation_method=ArtificialViscosity(1.0))
  reinit2 = StabilisedReinit(V_φ,Ωs,dΩ,h;stabilisation_method=GridapTopOpt.InteriorPenalty(V_φ,γg=1.0))
  reinit = GridapTopOpt.MultiStageStabilisedReinit([reinit1,reinit2])
  ls_evo = UnfittedFEEvolution(evo,reinit)

  ## Setup solver and FE operators
  state_collection = GridapTopOpt.EmbeddedCollection_in_φh(model,φh) do _φh
    update_collection!(Ωs,_φh)
    U,V = build_spaces(Ωs.Ω_act)
    state_map = NonlinearFEStateMap(res,U,V,V_φ,U_reg,φh;
      nls = GridapSolvers.NewtonSolver(LUSolver();maxiter=20,rtol=1e-8,atol=1e-11,verbose=true))
    (;
      :state_map => state_map,
      :J => GridapTopOpt.StateParamMap(Obj,state_map),
      :C => map(Ci -> GridapTopOpt.StateParamMap(Ci,state_map),[Vol,])
    )
  end
  pcfs = EmbeddedPDEConstrainedFunctionals(state_collection;analytic_dC=(dVol,))

  ## Hilbertian extension-regularisation problems
  α = (α_coeff*h_refine/order)^2
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

  ## Optimiser
  function initial_parameters(J,C)
    if isnothing(λelast) || isnothing(Λelast)
      return GridapTopOpt.default_al_init_params(J,C)
    else
      return [λelast],[Λelast]
    end
  end
  converged(m) = GridapTopOpt.default_al_converged(
    m;
    L_tol = 0.01*h_refine,
    C_tol = 0.01
  )
  optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
    γ,verbose=true,constraint_names=[:Vol],converged,initial_parameters)
  for (it,uh,φh) in optimiser
    if iszero(it % iter_mod)
      data = ["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh]
      writevtk(Ω,path*"Omega_$it",cellfields=data)
      writevtk(Ωs.Ωin,path*"Omega_in_$it",cellfields=data)
    end
    write_history(path*"/history.txt",optimiser.history)
  end
  it = get_history(optimiser).niter; uh = get_state(pcfs)
  data = ["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh]
  writevtk(Ω,path*"Omega_$it",cellfields=data)
  writevtk(Ωs.Ωin,path*"Omega_in_$it",cellfields=data)
end

_φ,alm_params = main_lin_elast(75)
λelast = alm_params[:λ1][end]
Λelast = alm_params[:Λ1][end]
neo_φ = copy(_φ)
main_neo(75,neo_φ,λelast,Λelast)