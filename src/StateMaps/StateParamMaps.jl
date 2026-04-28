# Abstract type is only needed for compat with staggered state maps. This
#  type will be deprecated in a future release.
abstract type AbstractStateParamMap{N} end

get_diff_order(::AbstractStateParamMap{N}) where N = N

"""
    struct StateParamMap{A,B,C,D,N} <: AbstractStateParamMap

A wrapper to handle partial differentation of a function F
of a specific form (see below) in a `ChainRules.jl` compatible way with caching.

# Assumptions

We assume that we have a function F of the following form:

`F(u,φ) = ∫(f(u,φ))dΩ₁ + ∫(g(u,φ))dΩ₂ + ...,`.

where `u` and `φ` are each expected to inherit from `Union{FEFunction,MultiFieldFEFunction}`
or the GridapDistributed equivalent.
"""
struct StateParamMap{A,B,C,D,N} <: AbstractStateParamMap{N}
  F       :: A
  spaces  :: B
  assems  :: C
  cache  :: D

  """
      StateParamMap(F,U::FESpace,V_φ::FESpace,
      assem_U::Assembler,assem_deriv::Assembler)

  Create an instance of `StateParamMap`.

  Use the optional argument `∂F∂u` and/or `∂F∂φ`  to specify the directional derivative of
  F(u,φ) with respect to the field u in the direction q as ∂F∂u(q,u,φ) and/or with respect
  to the field φ in the direction q as ∂F∂φ(q,u,φ).

  Optional arguments `∂u_ad_type` and `∂φ_ad_type` specify the approach for AD for multifield
  problems (either :split or :monolithic). For SingleField FE problems, this does nothing. Description of options
  can be found in Gridap.MultiField.
  """
  function StateParamMap(
    F,U::FESpace,V_φ::FESpace,
    assem_U::Assembler,assem_deriv::Assembler;
    ∂u_ad_type::Symbol=:split,
    ∂φ_ad_type::Symbol=:monolithic,
    ∂F∂u::Function = (q,u,φ) -> __gradient(x->F(x,φ),u;ad_type=∂u_ad_type),
    ∂F∂φ::Function = (q,u,φ) -> __gradient(x->F(u,x),φ;ad_type=∂φ_ad_type),
    diff_order::Int = 1
  )
    ## Dev note (commit fd65d0a):
    # In the past we used the following code to allocate vectors for the derivatives.
    # This was required because we needed these to be RHS vectors for VelocityExtension
    # problem. As of v0.3.0 (commmit fd65d0a), this is no longer required because VelocityExtension
    # expects dF to be a vector of DOFs. This is then mapped onto an appropriate RHS vector
    # using `_interpolate_onto_rhs!`.
    #
    # In `u_to_j_pullback` below, we do in-place assembly via `assemble_vector!` on these
    # allocated vectors. This is a bit naughty but works!
    #######
    # φ₀, u₀ = interpolate(x->-sqrt((x[1]-1/2)^2+(x[2]-1/2)^2)+0.2,V_φ), zero(U)
    # ∂j∂u_vecdata = collect_cell_vector(U,_∂F∂u(get_fe_basis(U),u₀,φ₀))
    # ∂j∂φ_vecdata = collect_cell_vector(V_φ,∇(F,[u₀,φ₀],2))
    # ∂j∂u_vec = allocate_vector(assem_U,∂j∂u_vecdata)
    # ∂j∂φ_vec = allocate_vector(assem_deriv,∂j∂φ_vecdata)
    #######

    assems = (assem_U,assem_deriv)
    spaces = (U,V_φ)
    cache = StateParamMapCache(∂F∂u,∂F∂φ)
    A, B, C, D = typeof(F), typeof(spaces), typeof(assems), typeof(cache)
    !(diff_order ∈ (1,2)) && error("Unsupported diff_order = $diff_order. Expected 1 or 2.")
    return new{A,B,C,D,diff_order}(F,spaces,assems,cache)
  end
end

get_spaces(m::StateParamMap) = m.spaces
get_state(m::StateParamMap) = FEFunction(m.spaces[1], m.caches[5])
get_parameter(m::StateParamMap) = FEFunction(m.spaces[2], m.caches[6])

function StateParamMap(F::Function,φ_to_u::AbstractFEStateMap;kwargs...)
  U = get_trial_space(φ_to_u)
  V_φ = get_aux_space(φ_to_u)
  assem_deriv = get_deriv_assembler(φ_to_u)
  assem_U = get_pde_assembler(φ_to_u)
  StateParamMap(F,U,V_φ,assem_U,assem_deriv;kwargs...)
end

"""
    (u_to_j::AbstractStateParamMap)(uh,φh)

Evaluate the `u_to_j` at parameters `uh` and `φh`.
"""
function (u_to_j::AbstractStateParamMap)(uh,φh)
  j = sum(u_to_j.F(uh,φh))
  check_and_build_cache!(u_to_j,uh,φh,j)
  return j
end

function (u_to_j::AbstractStateParamMap)(u::AbstractVector,φ::AbstractVector)
  U,V_φ = get_spaces(u_to_j)
  uh = FEFunction(U,u)
  φh = FEFunction(V_φ,φ)
  return u_to_j(uh,φh)
end

function check_and_build_cache!(u_to_j::StateParamMap,uh,φh,j)
  if !is_cache_built(u_to_j.cache)
    build_cache!(u_to_j,uh,φh)
  end
  update_diff_cache!(u_to_j,uh,φh,j)
  nothing
end

update_diff_cache!(::StateParamMap{A,B,C,D,1},uh,φh,j) where {A,B,C,D} = nothing

"""
    ChainRulesCore.rrule(u_to_j::StateParamMap,u,φ)

Return the evaluation of a `StateParamMap` and a
a function for evaluating the pullback of `u_to_j`. This enables
compatiblity with `ChainRules.jl`
"""
function ChainRulesCore.rrule(u_to_j::StateParamMap,u,φ)
  return u_to_j(u,φ), dj -> pullback(u_to_j,u,φ,dj)
end

function pullback(u_to_j::StateParamMap,uh,φh,dj)
  U,V_φ = get_spaces(u_to_j)
  assem_U,assem_deriv = u_to_j.assems
  ∂j∂u_vec,∂j∂φ_vec,∂F∂u,∂F∂φ = u_to_j.cache.plb_cache

  ## Compute ∂F/∂uh(uh,φh) and ∂F/∂φh(uh,φh)
  ∂j∂u = ∂F∂u(get_fe_basis(U),uh,φh)
  ∂j∂u_vecdata = collect_cell_vector(U,∂j∂u)
  assemble_vector!(∂j∂u_vec,assem_U,∂j∂u_vecdata)
  ∂j∂φ = ∂F∂φ(get_fe_basis(V_φ),uh,φh)
  ∂j∂φ_vecdata = collect_cell_vector(V_φ,∂j∂φ)
  assemble_vector!(∂j∂φ_vec,assem_deriv,∂j∂φ_vecdata)
  ∂j∂u_vec .*= dj
  ∂j∂φ_vec .*= dj
  update_inc_obj_cache!(u_to_j,uh,φh)
  (  NoTangent(), ∂j∂u_vec, ∂j∂φ_vec )
end

update_inc_obj_cache!(::StateParamMap{A,B,C,D,1},uh,φh) where {A,B,C,D} = nothing

function pullback(u_to_j::StateParamMap,u::AbstractVector,φ::AbstractVector,dj)
  U,V_φ = get_spaces(u_to_j)
  uh = FEFunction(U,u)
  φh = FEFunction(V_φ,φ)
  return pullback(u_to_j,uh,φh,dj)
end

# Cache
mutable struct StateParamMapCache
  fwd_cache::Tuple
  plb_cache::Tuple
  inc_obj_cache::Tuple
  cache_built::Bool
  fwd_ran:: Bool
  bwd_ran:: Bool
end

is_cache_built(c::StateParamMapCache) = c.cache_built

function StateParamMapCache(∂F∂u,∂F∂φ)
  plb_cache = (nothing, nothing, ∂F∂u, ∂F∂φ)
  StateParamMapCache((),plb_cache,(),false,false,false)
end

function build_cache!(u_to_j::StateParamMap{A,B,C,D,1},uh,φh) where {A,B,C,D}
  U, V_φ = get_spaces(u_to_j)
  ∂j∂u_vec = get_free_dof_values(zero(U))
  ∂j∂φ_vec = get_free_dof_values(zero(V_φ))
  ∂F∂u,∂F∂φ = u_to_j.cache.plb_cache[3:4]
  u_to_j.cache.plb_cache = (∂j∂u_vec,∂j∂φ_vec,∂F∂u,∂F∂φ)
  u_to_j.cache.cache_built = true
  return nothing
end

# Second order

function update_diff_cache!(u_to_j::StateParamMap{A,B,C,D,2},uh,φh,j) where {A,B,C,D}
  u, φ, j_cache = u_to_j.cache.fwd_cache
  copyto!(u, get_free_dof_values(uh))
  copyto!(φ, get_free_dof_values(φh))
  j_cache[] = j
  u_to_j.cache.fwd_ran = true
  u_to_j.cache.bwd_ran = false
  nothing
end

function build_cache!(u_to_j::StateParamMap{A,B,C,D,2},uh,φh) where {A,B,C,D}
  _,_,∂F∂u,∂F∂φ = u_to_j.cache.plb_cache
  U,V_φ = u_to_j.spaces
  u_to_j.cache.fwd_cache = (get_free_dof_values(zero(U)), get_free_dof_values(zero(V_φ)), Ref(0.0))
  u_to_j.cache.plb_cache = (get_free_dof_values(zero(U)), get_free_dof_values(zero(V_φ)), ∂F∂u, ∂F∂φ)
  u_to_j.cache.inc_obj_cache = build_inc_obj_cache(u_to_j.F,uh,φh,u_to_j.spaces)
  u_to_j.cache.cache_built = true
  u_to_j.cache.fwd_ran = false
  u_to_j.cache.bwd_ran = false
  return nothing
end

function build_inc_obj_cache(F,uh,φh,spaces)
  U,V_φ = spaces

  # ∂²J / ∂u² * u̇
  ∂2J∂u2 = Gridap.hessian(uh->F(uh,φh),uh)
  assem_∂2J∂u2 = SparseMatrixAssembler(U,U)
  ∂2J∂u2_mat = assemble_matrix(∂2J∂u2,assem_∂2J∂u2,U,U)

  # ∂/∂φ (∂J/∂u ) * ̇φ
  ∂J∂u(uh,φh) = Gridap.gradient(uh->F(uh,φh),uh)
  ∂2J∂u∂φ = Gridap.jacobian(φ->∂J∂u(uh,φ),φh)
  assem_∂2J∂u∂φ = SparseMatrixAssembler(V_φ,U)
  ∂2J∂u∂φ_mat = assemble_matrix(∂2J∂u∂φ,assem_∂2J∂u∂φ,V_φ,U)

  # ∂²J / ∂φ² * ̇φ
  ∂2J∂φ2 = Gridap.hessian(φ->F(uh,φ),φh)
  assem_∂2J∂φ2 = SparseMatrixAssembler(V_φ,V_φ)
  ∂2J∂φ2_mat = assemble_matrix(∂2J∂φ2,assem_∂2J∂φ2,V_φ,V_φ)

  # ∂/∂u (∂J / ∂φ) * u̇
  ∂J∂φ(uh,φh) = Gridap.gradient(φ->F(uh,φ),φh)
  ∂2J∂φ∂u = Gridap.jacobian(uh->∂J∂φ(uh,φh),uh)
  assem_∂2J∂φ∂u = SparseMatrixAssembler(U,V_φ)
  ∂2J∂φ∂u_mat = assemble_matrix(∂2J∂φ∂u,assem_∂2J∂φ∂u,U,V_φ)

  dṗ_from_j = get_free_dof_values(zero(V_φ))
  du̇_from_j = get_free_dof_values(zero(U))

  dṗ_from_j, du̇_from_j, assem_∂2J∂u2, ∂2J∂u2_mat,   assem_∂2J∂u∂φ, ∂2J∂u∂φ_mat,   assem_∂2J∂φ2, ∂2J∂φ2_mat,   assem_∂2J∂φ∂u, ∂2J∂φ∂u_mat
end

function update_inc_obj_cache!(u_to_j::StateParamMap{A,B,C,D,2},uh,φh) where {A,B,C,D}
  U,V_φ = get_spaces(u_to_j)
  F = u_to_j.F
  inc_obj_cache = u_to_j.cache.inc_obj_cache
  dṗ_from_j, du̇_from_j, assem_∂2J∂u2, ∂2J∂u2_mat, assem_∂2J∂u∂φ,
    ∂2J∂u∂φ_mat, assem_∂2J∂φ2, ∂2J∂φ2_mat, assem_∂2J∂φ∂u, ∂2J∂φ∂u_mat = inc_obj_cache

  ∂2J∂u2 = Gridap.hessian(uh->F(uh,φh),uh)
  assemble_matrix!(∂2J∂u2,∂2J∂u2_mat,assem_∂2J∂u2,U,U)

  ∂J∂u(uh,φh) = Gridap.gradient(uh->F(uh,φh),uh)
  ∂2J∂u∂φ = Gridap.jacobian(φ->∂J∂u(uh,φ),φh)
  assemble_matrix!(∂2J∂u∂φ,∂2J∂u∂φ_mat,assem_∂2J∂u∂φ,V_φ,U)

  ∂2J∂φ2 = Gridap.hessian(φ->F(uh,φ),φh)
  assemble_matrix!(∂2J∂φ2,∂2J∂φ2_mat,assem_∂2J∂φ2,V_φ,V_φ)

  ∂J∂φ(uh,φh) = Gridap.gradient(φ->F(uh,φ),φh)
  ∂2J∂φ∂u = Gridap.jacobian(uh->∂J∂φ(uh,φh),uh)
  assemble_matrix!(∂2J∂φ∂u,∂2J∂φ∂u_mat,assem_∂2J∂φ∂u,U,V_φ)

  u_to_j.cache.bwd_ran = true

  return nothing
end

# IO
function Base.show(io::IO,object::AbstractStateParamMap)
  print(io,"$(nameof(typeof(object)))")
end

# Backwards compat
function StateParamIntegrandWithMeasure(args...)
  error(
    """
    As of v0.4.0, StateParamIntegrandWithMeasure was deprecated in favour of StateParamMap.
    """
  )
end
function StateParamMap(
    F,U::FESpace,V_φ::FESpace,U_reg,assem_U::Assembler,assem_deriv::Assembler;kwargs...)
  error(_msg_v0_3_0(StateParamMap))
end
