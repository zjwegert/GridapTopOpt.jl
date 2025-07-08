"""
    mutable struct CutFEMEvolver{V,M} <: Evolver

CutFEM method for level-set evolution based on method developed by
- Burman et al. (2018). DOI: `10.1016/j.cma.2017.09.005`.
- Burman et al. (2017). DOI: `10.1016/j.cma.2016.12.021`.
- Burman and Fernández (2009). DOI: `10.1016/j.cma.2009.02.011`
This solves the tranport equation
``
\\frac{\\partial\\phi(t,\boldsymbol{x})}{\\partial t}+\\boldsymbol{\\beta}\\cdot\\boldsymbol{\\nabla}\\phi(t,\\boldsymbol{x})=0,
``
with ``boldsymbol{\\beta}=\\boldsymbol{n}v_h``, ``\\phi(0,\\boldsymbol{x})=\\phi_0(\\boldsymbol{x}),`` and ``\\quad\\boldsymbol{x}\\in D,~t\\in(0,T)``.

# Parameters
- `ode_solver::ODESolver`: ODE solver
- `Ωs::B`: `EmbeddedCollection` holding updatable triangulation and measures from GridapEmbedded
- `dΩ_bg::C`: Measure for integration
- `space::B`: Level-set FE space
- `assembler::Assembler`: FE assembler
- `params::D`: Tuple of stabilisation parameter `γg`, mesh sizes `h`, and
  max steps `max_steps`, and background mesh skeleton parameters

!!! warning
    Caching for the `CutFEMEvolver` method is currently disabled. This will be
    re-enabled in the future."
"""
struct CutFEMEvolver{A,B,C} <: Evolver
  ode_solver::ODESolver
  Ωs::EmbeddedCollection
  dΩ_bg::A
  space::B
  assembler::Assembler
  params::C

  @doc """
      CutFEMEvolver(V_φ::B,Ωs::EmbeddedCollection,dΩ_bg::A,h;
        max_steps=10,
        γg = 0.1,
        ode_ls = LUSolver(),
        ode_nl = ode_ls,
        ode_solver = MutableRungeKutta(ode_nl, ode_ls, 0.1, :DIRK_CrankNicolson_2_2),
        assembler=SparseMatrixAssembler(V_φ,V_φ)
      ) where {A,B}

  Create an instance of `CutFEMEvolver` with the space for the level-set `V_φ`,
  the `EmbeddedCollection` `Ωs` for the triangulation and measures, the measure
  `dΩ_bg` for the background mesh, and the mesh size `h`. The mesh size `h` can
  either be a scalar or a `CellField` object.

  The optional arguments are:
  - `correct_ls`: Boolean for whether or not to ensure LS DOFs aren't zero
    this MUST be true for differentiation in unfitted methods.
  - `max_steps`: Maximum number of steps for the ODE solver.
  - `γg`: Stabilisation parameter for the continuous interior penalty term.
  - `ode_ls`: Linear solver for the ODE solver.
  - `ode_nl`: Non-linear solver for the ODE solver.
  - `ode_solver`: ODE solver, default is `MutableRungeKutta(ode_nl, ode_ls, 0.1, :DIRK_CrankNicolson_2_2)`.
  - `assembler`: Assembler for the finite element space, default is `SparseMatrixAssembler(V_φ,V_φ)`.

  # Note
  - The stepsize `dt = 0.1` in `MutableRungeKutta` is a place-holder and is updated using
    the `γ` passed to `evolve!`.
  """
  function CutFEMEvolver(V_φ::B,Ωs::EmbeddedCollection,dΩ_bg::A,h;
      correct_ls = true,
      max_steps=10,
      γg = 0.1,
      ode_ls = LUSolver(),
      ode_nl = ode_ls,
      ode_solver = MutableRungeKutta(ode_nl, ode_ls, 0.1, :DIRK_CrankNicolson_2_2),
      assembler=SparseMatrixAssembler(V_φ,V_φ)) where {A,B}
    model = get_background_model(get_triangulation(V_φ))
    Γg = SkeletonTriangulation(model)
    dΓg = Measure(Γg,2get_order(V_φ))
    n_Γg = get_normal_vector(Γg)
    hmin = minimum(get_element_diameters(model))
    params = (;γg,h,hmin,max_steps,dΓg,n_Γg,correct_ls)
    new{A,B,typeof(params)}(ode_solver,Ωs,dΩ_bg,V_φ,assembler,params)
  end
end

function get_min_dof_spacing(s::CutFEMEvolver)
  V_φ = get_ls_space(s)
  hmin = s.params.hmin
  return hmin/get_order(V_φ)
end

function get_ls_space(s::CutFEMEvolver)
  s.space
end

function evolve!(s::CutFEMEvolver,φ::AbstractVector,vel::AbstractVector,args...)
  φh = FEFunction(get_ls_space(s),φ)
  velh = FEFunction(get_ls_space(s),vel)
  evolve!(s,φh,velh,args...)
end

function evolve!(s::CutFEMEvolver,φh,velh,γ)
  ode_solver = s.ode_solver
  params = s.params
  hmin, max_steps, correct_ls = params.hmin, params.max_steps, params.correct_ls

  # Setup FE operator and solver
  ode_solver.dt = γ*hmin
  ode_op = get_transient_operator(φh,velh,s)
  ode_sol = solve(ode_solver,ode_op,0.0,ode_solver.dt*max_steps,φh)

  # March
  march = Base.iterate(ode_sol)
  data, state = march
  # state_new = update_reuse!(state,true) # TODO: This has been disabled due to bug. See below discussion.
  state_new = state

  march_new = data, state_new
  while march_new !== nothing
    data, state_new = march_new
    march_new = Base.iterate(ode_sol,state_new)
  end

  # Update φh and cache
  _, φhF = data
  copy!(get_free_dof_values(φh),get_free_dof_values(φhF))
  # TODO: This has been disabled for the time being. Originally when this code
  #   was written, we expected that changing reuse to false and iterating once
  #   would update the stiffness matrix. However, this does not appear to be the case.
  # cache = state_new
  cache = nothing
  correct_ls && correct_ls!(φh)
  update_collection!(s.Ωs,φh)
  return get_free_dof_values(φh), cache
end

function evolve!(s::CutFEMEvolver,φh,velh,γ,::Nothing)
  evolve!(s,φh,velh,γ)
end

## Disabled due to above
# function update_reuse!(state,reuse_new;zero_tF=false)
#   U, (tF, stateF, state0, uF, odecache) = state
#   odeslvrcache, odeopcache = odecache
#   _, ui_pre, slopes, J, r, sysslvrcaches = odeslvrcache

#   odeslvrcache_new = (reuse_new, ui_pre, slopes, J, r, sysslvrcaches)
#   odecache_new = odeslvrcache_new, odeopcache
#   _tF = zero_tF ? 0.0 : tF
#   return U, (_tF, stateF, state0, uF, odecache_new)
# end

# function evolve!(s::CutFEMEvolver,φh,velh,γ,cache)
#   ode_solver = s.ode_solver
#   params = s.params(s)
#   hmin, max_steps, correct_ls = params.hmin, params.max_steps, params.correct_ls

#   ## Update state
#   # `get_transient_operator` re-creates the entire TransientLinearFEOperator wrapper.
#   #   We do this so that the first iterate of ODESolution always recomputes the
#   #   stiffness matrix and associated the Jacboian, numerical setups, etc via
#   #   `constant_forms = (false,true)`.
#   ode_solver.dt = γ*hmin
#   ode_op = get_transient_operator(φh,velh,s)
#   # Between the first iterate and subsequent iterates we use the function
#   #   `update_reuse!` to update the iterator state so that we re-use
#   #   the stiffness matrix, etc. The Optional argument `zero_tF` indicates
#   #   whether we are solving a new ODE with the same functional form but
#   #   updated coefficients in the weak form. If so, we want to re-use the cache.
#   state_inter = update_reuse!(cache,false;zero_tF=true)

#   ## March
#   ode_sol = solve(ode_solver,ode_op,0.0,ode_solver.dt*max_steps,φh)
#   march = Base.iterate(ode_sol,state_inter) # First step includes stiffness matrix update
#   data, state = march
#   state_updated = update_reuse!(state,true) # Fix the stiffness matrix for remaining march
#   march_updated = data, state_updated
#   while march_updated !== nothing
#     data, state_updated = march_updated
#     march_updated = Base.iterate(ode_sol,state_updated)
#   end

#   ## Update φh and cache
#   _, φhF = data
#   copy!(get_free_dof_values(φh),get_free_dof_values(φhF))
#   correct_ls && correct_ls!(φh)
#   update_collection!(s.Ωs,φh) # TODO: remove?
#   return φh,cache
# end

function get_transient_operator(φh,velh,s::CutFEMEvolver)
  V_φ, dΩ_bg, assembler, params = s.space, s.dΩ_bg, s.assembler, s.params
  γg, h, dΓg, n_Γg = params.γg, params.h, params.dΓg, params.n_Γg
  ϵ = 1e-20

  v_norm = maximum(abs,get_free_dof_values(velh))
  β(vh,∇φ) = vh/(ϵ + v_norm) * ∇φ/(ϵ + norm(∇φ))
  γ(h) = γg*h^2
  βh = β ∘ (velh,∇(φh))
  βh_n_Γg = abs ∘ (βh.plus ⋅ n_Γg.plus)

  aₛ(u,v,h::CellField) = ∫(mean(γ ∘ h)*βh_n_Γg*jump(∇(u) ⋅ n_Γg)*jump(∇(v) ⋅ n_Γg))dΓg
  aₛ(u,v,h::Real) = ∫(γ(h)*βh_n_Γg*jump(∇(u) ⋅ n_Γg)*jump(∇(v) ⋅ n_Γg))dΓg

  stiffness(t,u,v) = ∫((βh ⋅ ∇(u)) * v)dΩ_bg + aₛ(u,v,h)
  mass(t, ∂ₜu, v) = ∫(∂ₜu * v)dΩ_bg
  forcing(t,v) = ∫(0v)dΩ_bg + ∫(0*jump(∇(v) ⋅ n_Γg))dΓg
  # Second term is added to address the following issue:
  #  - ODEs is allocating separately the residual and jacobian
  #  - This is fine in serial, but in parallel there are some instances where the the following happens:
  #     - The residual is touched by less ghost entries than the columns of the matrix
  #     - If we assemble both jac and res together, we communicate the extra ghost ids to
  #       the residual, so everything is consistent.
  #     - However, if we assemble the residual and jacobian separately,
  #       the residual is not aware of the extra ghost ids
  # This happens when there are touched ghost entries that do not belong to the local domain.
  # In particular, this happens when we have jumps, where some contributions come from two
  # cells away. Boundary cells then get contributions from cells which are not in the local domain.
  Ut_φ = TransientTrialFESpace(V_φ)

  # TODO: This has been disabled due to bug. See below discussion.
  # ode_op = TransientLinearFEOperator((stiffness,mass),forcing,Ut_φ,V_φ;
  #   constant_forms=(false,true),assembler)
  ode_op = TransientLinearFEOperator((stiffness,mass),forcing,Ut_φ,V_φ;
    constant_forms=(true,true),assembler)
  return ode_op
end