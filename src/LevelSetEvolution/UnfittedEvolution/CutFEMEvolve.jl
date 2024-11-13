"""
    mutable struct CutFEMEvolve{V,M} <: Evolver

CutFEM method for level-set evolution based on method developed by 
Burman et al. (2017). DOI: `10.1016/j.cma.2017.09.005`.

# Parameters
- `ode_solver::ODESolver`: ODE solver 
- `Ωs::B`: `EmbeddedCollection` holding updatable triangulation and measures from GridapEmbedded
- `dΩ_bg::C`: Measure for integration
- `space::B`: Level-set FE space
- `assembler::Assembler`: FE assembler
- `params::D`: Tuple of stabilisation parameter `Γg`, mesh size `h`, and
  max steps `max_steps`
- `cache`: Cache for evolver, initially `nothing`.

# Note
- The stepsize `dt = 0.1` in `RungeKutta` is a place-holder and is updated using
  the `γ` passed to `solve!`.
- We expect the EmbeddedCollection `Ωs` to contain `:F_act` and `:dF_act`. If 
  this is not available we add it to the recipe list in `Ωs` and a warning will appear. 
"""
mutable struct CutFEMEvolve{A,B,C} <: Evolver
  ode_solver::ODESolver
  Ωs::EmbeddedCollection
  dΩ_bg::A
  space::B
  assembler::Assembler
  params::C
  cache
  function CutFEMEvolve(V_φ::B,Ωs::EmbeddedCollection,dΩ_bg::A,h::Real;
      max_steps=10,
      Γg = 0.1,
      ode_ls = LUSolver(),
      ode_nl = NLSolver(ode_ls, show_trace=false, method=:newton, iterations=10),
      ode_solver = MutableRungeKutta(ode_nl, ode_ls, 0.1, :DIRK_CrankNicolson_2_2),
      assembler=SparseMatrixAssembler(V_φ,V_φ)) where {A,B}
    if !(:F_act ∈ keys(Ωs.objects))
      @warn "Expected triangulation ':F_act' not found in the 
      EmbeddedCollection. This and the corresponding measure ':dF_act' have been
      added to the recipe list. 
      
      Ensure that you are not using ':F_act' under a different
      name to avoid additional computation for cutting."
      function F_act_recipe(cutgeo)
        Ω_act = Triangulation(cutgeo,ACTIVE)
        F_act = SkeletonTriangulation(Ω_act)
        (; 
          :F_act => F_act,
          :dF_act => Measure(F_act,2get_order(V_φ))
        )
      end
      add_recipe!(Ωs,F_act_recipe)
    end
    params = (;Γg,h,max_steps)
    new{A,B,typeof(params)}(ode_solver,Ωs,dΩ_bg,V_φ,assembler,params,nothing)
  end
end

get_ode_solver(s::CutFEMEvolve) = s.ode_solver
get_assembler(s::CutFEMEvolve) = s.assembler
get_space(s::CutFEMEvolve) = s.space
get_embedded_collection(s::CutFEMEvolve) = s.Ωs
get_measure(s::CutFEMEvolve) = s.dΩ_bg
get_params(s::CutFEMEvolve) = s.params
get_cache(s::CutFEMEvolve) = s.cache

function get_transient_operator(φh,velh,s::CutFEMEvolve)
  Ωs, V_φ, dΩ_bg, assembler, params = s.Ωs, s.space, s.dΩ_bg, s.assembler, s.params
  Γg, h = params.Γg, params.h
  ϵ = 1e-20

  update_collection!(Ωs,φh)
  F_act = Ωs.F_act
  dF_act = Ωs.dF_act
  n = get_normal_vector(F_act)

  β = velh*∇(φh)/(ϵ + norm ∘ ∇(φh))
  stiffness(t,u,v) = ∫((β ⋅ ∇(u)) * v)dΩ_bg + ∫(Γg*h^2*jump(∇(u) ⋅ n)*jump(∇(v) ⋅ n))dF_act
  mass(t, ∂ₜu, v) = ∫(∂ₜu * v)dΩ_bg
  forcing(t,v) = ∫(0v)dΩ_bg
  Ut_φ = TransientTrialFESpace(V_φ)
  ode_op = TransientLinearFEOperator((stiffness,mass),forcing,Ut_φ,V_φ;
    constant_forms=(false,true),assembler)
  return ode_op
end

function update_reuse!(state,reuse_new;zero_tF=false)
  U, (tF, stateF, state0, uF, odecache) = state
  odeslvrcache, odeopcache = odecache
  _, ui_pre, slopes, J, r, sysslvrcaches = odeslvrcache

  odeslvrcache_new = (reuse_new, ui_pre, slopes, J, r, sysslvrcaches)
  odecache_new = odeslvrcache_new, odeopcache
  _tF = zero_tF ? 0.0 : tF
  return U, (_tF, stateF, state0, uF, odecache_new)
end

function solve!(s::CutFEMEvolve,φh,velh,γ,cache::Nothing)
  ode_solver = get_ode_solver(s)
  params = get_params(s)
  h, max_steps = params.h, params.max_steps

  # Setup FE operator and solver
  ode_op = get_transient_operator(φh,velh,s)
  dt = _compute_Δt(h,γ,get_free_dof_values(velh))
  ode_solver.dt = dt
  ode_sol = solve(ode_solver,ode_op,0.0,dt*max_steps,φh)
  
  # March
  march = Base.iterate(ode_sol)
  data, state = march
  state_new = update_reuse!(state,true)
  march_new = data, state_new
  while march_new !== nothing
    data, state_new = march_new
    march_new = Base.iterate(ode_sol,state_new)
  end

  # Update φh and cache
  _, φhF = data
  copy!(get_free_dof_values(φh),get_free_dof_values(φhF))
  s.cache = state_new
  
  return φh
end

function solve!(s::CutFEMEvolve,φh,velh,γ,cache)
  ode_solver = get_ode_solver(s)
  params = get_params(s)
  h, max_steps = params.h, params.max_steps

  ## Update state
  # `get_transient_operator` re-creates the entire TransientLinearFEOperator wrapper.
  #   We do this so that the first iterate of ODESolution always recomputes the 
  #   stiffness matrix and associated the Jacboian, numerical setups, etc via
  #   `constant_forms = (false,true)`.
  ode_op = get_transient_operator(φh,velh,s)
  # Between the first iterate and subsequent iterates we use the function 
  #   `update_reuse!` to update the iterator state so that we re-use
  #   the stiffness matrix, etc. The Optional argument `zero_tF` indicates 
  #   whether we are solving a new ODE with the same functional form but
  #   updated coefficients in the weak form. If so, we want to re-use the cache.
  state_inter = update_reuse!(cache,false;zero_tF=true)
  dt = _compute_Δt(h,γ,get_free_dof_values(velh))
  ode_solver.dt = dt

  ## March
  ode_sol = solve(ode_solver,ode_op,0.0,dt*max_steps,φh)
  march = Base.iterate(ode_sol,state_inter) # First step includes stiffness matrix update
  data, state = march
  state_updated = update_reuse!(state,true) # Fix the stiffness matrix for remaining march
  march_updated = data, state_updated
  while march_updated !== nothing
    data, state_updated = march_updated
    march_updated = Base.iterate(ode_sol,state_updated)
  end

  ## Update φh and cache
  _, φhF = data
  copy!(get_free_dof_values(φh),get_free_dof_values(φhF))
  s.cache = state_updated
  
  return φh
end

function _compute_Δt(h,γ,vel)
  T = eltype(γ)
  v_norm = maximum(abs,vel)
  return γ * h / (eps(T)^2 + v_norm)
end