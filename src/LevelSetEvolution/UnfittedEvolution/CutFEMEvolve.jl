"""
    mutable struct CutFEMEvolve{V,M} <: Evolver

CutFEM method for level-set evolution based on method developed by 
Burman et al. (2017). DOI: `10.1016/j.cma.2017.09.005`.

# Parameters
- `ode_solver::ODESolver`: ODE solver 
- `model::A`: FE model
- `space::B`: Level-set FE space
- `dΩ::C`: Measure for integration
- `assembler::Assembler`: FE assembler
- `params::D`: Tuple of stabilisation parameter `Γg`, mesh size `h`, 
  max steps `max_steps`, and FE space `order`
- `cache`: Cache for evolver, initially `nothing`.

# Note
The stepsize `dt = 0.1` in `RungeKutta` is a place-holder and is updated using
the `γ` passed to `solve!`.
"""
mutable struct CutFEMEvolve{A,B,C,D} <: Evolver
  ode_solver::ODESolver
  model::A
  space::B
  dΩ::C
  assembler::Assembler
  params::D
  cache
  function CutFEMEvolve(model::A,V_φ::B,dΩ::C,h::Real;
      max_steps=10,
      Γg = 0.1,
      ode_ls = LUSolver(),
      ode_nl = NLSolver(ode_ls, show_trace=false, method=:newton, iterations=10),
      ode_solver = MutableRungeKutta(ode_nl, ode_ls, 0.1, :DIRK_CrankNicolson_2_2),
      assembler=SparseMatrixAssembler(V_φ,V_φ)) where {A,B,C}
    params = (;Γg,h,max_steps,order=get_order(V_φ))
    new{A,B,C,typeof(params)}(ode_solver,model,V_φ,dΩ,assembler,params,nothing)
  end
end

get_ode_solver(s::CutFEMEvolve) = s.ode_solver
get_assembler(s::CutFEMEvolve) = s.assembler
get_space(s::CutFEMEvolve) = s.space
get_model(s::CutFEMEvolve) = s.model
get_measure(s::CutFEMEvolve) = s.dΩ
get_params(s::CutFEMEvolve) = s.params
get_cache(s::CutFEMEvolve) = s.cache

function get_stiffness_matrix_map(φh,velh,dΩ,model,order,Γg,h)
  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)
  Ω_act = Triangulation(cutgeo,ACTIVE)
  F_act = SkeletonTriangulation(Ω_act)
  dF_act = Measure(F_act,2*order)
  n = get_normal_vector(F_act)
  ϵ = 1e-20

  β = velh*∇(φh)/(ϵ + norm ∘ ∇(φh))
  return (t,u,v) -> ∫((β ⋅ ∇(u)) * v)dΩ + ∫(Γg*h^2*jump(∇(u) ⋅ n)*jump(∇(v) ⋅ n))dF_act
end

function solve!(s::CutFEMEvolve,φh,velh,γ,cache::Nothing)
  ode_solver, model, V_φ, dΩ, assembler = s.ode_solver, s.model, s.space, s.dΩ, s.assembler
  Γg, h, max_steps, order = s.params

  # Weak form
  stiffness = get_stiffness_matrix_map(φh,velh,dΩ,model,order,Γg,h)
  mass(t, ∂ₜu, v) = ∫(∂ₜu * v)dΩ
  forcing(t,v) = ∫(0v)dΩ

  # Setup FE operator and solver
  Ut_φ = TransientTrialFESpace(V_φ)
  ode_op = TransientLinearFEOperator((stiffness,mass),forcing,Ut_φ,V_φ;
    constant_forms=(true,true),assembler)
  dt = _compute_Δt(h,γ,get_free_dof_values(velh))
  ode_solver.dt = dt
  ode_sol = solve(ode_solver,ode_op,0.0,dt*max_steps,φh)
  
  # March
  march = Base.iterate(ode_sol)
  data, state = march
  while march !== nothing
    data, state = march
    march = Base.iterate(ode_sol,state)
  end

  # Update φh and cache
  _, φhF = data
  copy!(get_free_dof_values(φh),get_free_dof_values(φhF))
  s.cache = (ode_op, state)
  
  return φh
end

function solve!(s::CutFEMEvolve,φh,velh,γ,cache)
  ode_solver, model, dΩ, V_φ, assembler, = s.ode_solver, s.model, s.dΩ, s.space, s.assembler
  Γg, h, max_steps, order = s.params
  ode_op, state0 = cache

  U, (tF, stateF, state0, uF, odecache) = state0
  odeslvrcache, odeopcache = odecache
  K, M = odeopcache.const_forms

  # Update odeopcache and solver
  stiffness = get_stiffness_matrix_map(φh,velh,dΩ,model,order,Γg,h)
  assemble_matrix!((u,v)->stiffness(0.0,u,v),K,assembler,V_φ,V_φ)
  state = U, (0.0, stateF, state0, uF, odecache)
  dt = _compute_Δt(h,γ,get_free_dof_values(velh))
  ode_solver.dt = dt

  # March
  ode_sol = solve(ode_solver,ode_op,0.0,dt*max_steps,φh)
  march = Base.iterate(ode_sol,state)
  data, state = march
  while march !== nothing
    data, state = march
    march = Base.iterate(ode_sol,state)
  end

  # Update φh and cache
  _, φhF = data
  copy!(get_free_dof_values(φh),get_free_dof_values(φhF))
  s.cache = (ode_op, state)
  
  return φh
end

function _compute_Δt(h,γ,vel)
  T = eltype(γ)
  v_norm = maximum(abs,vel)
  return γ * h / (eps(T)^2 + v_norm)
end