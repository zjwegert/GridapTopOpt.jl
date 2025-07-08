# Abstract type is only needed for compat with staggered state maps. This
#  type will be deprecated in a future release.
abstract type AbstractStateParamMap end

"""
    struct StateParamMap{A,B,C,D} <: AbstractStateParamMap

A wrapper to handle partial differentation of a function F
of a specific form (see below) in a `ChainRules.jl` compatible way with caching.

# Assumptions

We assume that we have a function F of the following form:

`F(u,φ) = ∫(f(u,φ))dΩ₁ + ∫(g(u,φ))dΩ₂ + ...,`.

where `u` and `φ` are each expected to inherit from `Union{FEFunction,MultiFieldFEFunction}`
or the GridapDistributed equivalent.
"""
struct StateParamMap{A,B,C,D} <: AbstractStateParamMap
  F       :: A
  spaces  :: B
  assems  :: C
  caches  :: D
end

"""
    StateParamMap(F,U::FESpace,V_φ::FESpace,
    assem_U::Assembler,assem_deriv::Assembler)

Create an instance of `StateParamMap`.

Use the optional argument `∂F∂u` and/or `∂F∂φ`  to specify the directional derivative of
F(u,φ) with respect to the field u in the direction q as ∂F∂u(q,u,φ) and/or with respect
to the field φ in the direction q as ∂F∂φ(q,u,φ).
"""
function StateParamMap(
  F,U::FESpace,V_φ::FESpace,
  assem_U::Assembler,assem_deriv::Assembler;
  ∂F∂u::T = nothing,
  ∂F∂φ::V = nothing
) where {T<:Union{Function,Nothing},V<:Union{Function,Nothing}}

  # Use analytic derivatives?
  _∂F∂u(q,u,φ) = (T <: Nothing) ? Gridap.gradient(x->F(x,φ),u) : ∂F∂u(q,u,φ)
  _∂F∂φ(q,u,φ) = (T <: Nothing) ? Gridap.gradient(x->F(u,x),u) : ∂F∂φ(q,u,φ)

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

  ∂j∂u_vec = get_free_dof_values(zero(U))
  ∂j∂φ_vec = get_free_dof_values(zero(V_φ))
  assems = (assem_U,assem_deriv)
  spaces = (U,V_φ)
  caches = (∂j∂u_vec,∂j∂φ_vec,_∂F∂u)
  return StateParamMap(F,spaces,assems,caches)
end

function get_∂F∂φ_vec(u_to_j::StateParamMap)
  u_to_j.caches[2]
end

function StateParamMap(F::Function,φ_to_u::AbstractFEStateMap;kwargs...)
  U = get_trial_space(φ_to_u)
  V_φ = get_aux_space(φ_to_u)
  assem_deriv = get_deriv_assembler(φ_to_u)
  assem_U = get_pde_assembler(φ_to_u)
  StateParamMap(F,U,V_φ,assem_U,assem_deriv;kwargs...)
end

"""
    (u_to_j::StateParamMap)(uh,φh)

Evaluate the `StateParamMap` at parameters `uh` and `φh`.
"""
(u_to_j::AbstractStateParamMap)(uh,φh) = sum(u_to_j.F(uh,φh))

function (u_to_j::StateParamMap)(u::AbstractVector,φ::AbstractVector)
  U,V_φ = u_to_j.spaces
  uh = FEFunction(U,u)
  φh = FEFunction(V_φ,φ)
  return u_to_j(uh,φh)
end

"""
    ChainRulesCore.rrule(u_to_j::StateParamMap,uh,φh)

Return the evaluation of a `StateParamMap` and a
a function for evaluating the pullback of `u_to_j`. This enables
compatiblity with `ChainRules.jl`
"""
function ChainRulesCore.rrule(u_to_j::StateParamMap,uh,φh)
  F = u_to_j.F
  U,V_φ = u_to_j.spaces
  assem_U,assem_deriv = u_to_j.assems
  ∂j∂u_vec,∂j∂φ_vec,∂F∂u = u_to_j.caches

  function u_to_j_pullback(dj)
    ## Compute ∂F/∂uh(uh,φh) and ∂F/∂φh(uh,φh)
    ∂j∂u = ∂F∂u(get_fe_basis(U),uh,φh)
    ∂j∂u_vecdata = collect_cell_vector(U,∂j∂u)
    assemble_vector!(∂j∂u_vec,assem_U,∂j∂u_vecdata)
    ∂j∂φ = ∇(F,[uh,φh],2)
    ∂j∂φ_vecdata = collect_cell_vector(V_φ,∂j∂φ)
    assemble_vector!(∂j∂φ_vec,assem_deriv,∂j∂φ_vecdata)
    ∂j∂u_vec .*= dj
    ∂j∂φ_vec .*= dj
    (  NoTangent(), ∂j∂u_vec, ∂j∂φ_vec )
  end
  return u_to_j(uh,φh), u_to_j_pullback
end

function ChainRulesCore.rrule(u_to_j::StateParamMap,u::AbstractVector,φ::AbstractVector)
  U,V_φ = u_to_j.spaces
  uh = FEFunction(U,u)
  φh = FEFunction(V_φ,φ)
  return ChainRulesCore.rrule(u_to_j,uh,φh)
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
  @error _msg_v0_3_0(StateParamMap)
end