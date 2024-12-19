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
    U_reg::FESpace,assem_U::Assembler,assem_deriv::Assembler)

Create an instance of `StateParamMap`.
"""
function StateParamMap(
  F,U::FESpace,V_φ::FESpace,U_reg::FESpace,
  assem_U::Assembler,assem_deriv::Assembler
)
  φ₀, u₀ = interpolate(x->-sqrt((x[1]-1/2)^2+(x[2]-1/2)^2)+0.2,V_φ), zero(U)
  # TODO: Can we make F a dummy functional?
  ∂j∂u_vecdata = collect_cell_vector(U,∇(F,[u₀,φ₀],1))
  ∂j∂φ_vecdata = collect_cell_vector(U_reg,∇(F,[u₀,φ₀],2))
  ∂j∂u_vec = allocate_vector(assem_U,∂j∂u_vecdata)
  ∂j∂φ_vec = allocate_vector(assem_deriv,∂j∂φ_vecdata)
  assems = (assem_U,assem_deriv)
  spaces = (U,V_φ,U_reg)
  caches = (∂j∂u_vec,∂j∂φ_vec)
  return StateParamMap(F,spaces,assems,caches)
end

function StateParamMap(F::Function,φ_to_u::AbstractFEStateMap)
  U = get_trial_space(φ_to_u)
  V_φ = get_aux_space(φ_to_u)
  U_reg = get_deriv_space(φ_to_u)
  assem_deriv = get_deriv_assembler(φ_to_u)
  assem_U = get_pde_assembler(φ_to_u)
  StateParamMap(F,U,V_φ,U_reg,assem_U,assem_deriv)
end

"""
    (u_to_j::StateParamMap)(uh,φh)

Evaluate the `StateParamMap` at parameters `uh` and `φh`.
"""
(u_to_j::AbstractStateParamMap)(uh,φh) = sum(u_to_j.F(uh,φh))

function (u_to_j::StateParamMap)(u::AbstractVector,φ::AbstractVector)
  U,V_φ,_ = u_to_j.spaces
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
  U,V_φ,U_reg = u_to_j.spaces
  assem_U,assem_deriv = u_to_j.assems
  ∂j∂u_vec,∂j∂φ_vec = u_to_j.caches

  function u_to_j_pullback(dj)
    ## Compute ∂F/∂uh(uh,φh) and ∂F/∂φh(uh,φh)
    ∂j∂u = ∇(F,[uh,φh],1)
    ∂j∂u_vecdata = collect_cell_vector(U,∂j∂u)
    assemble_vector!(∂j∂u_vec,assem_U,∂j∂u_vecdata)
    ∂j∂φ = ∇(F,[uh,φh],2)
    ∂j∂φ_vecdata = collect_cell_vector(U_reg,∂j∂φ)
    assemble_vector!(∂j∂φ_vec,assem_deriv,∂j∂φ_vecdata)
    ∂j∂u_vec .*= dj
    ∂j∂φ_vec .*= dj
    (  NoTangent(), ∂j∂u_vec, ∂j∂φ_vec )
  end
  return u_to_j(uh,φh), u_to_j_pullback
end

function ChainRulesCore.rrule(u_to_j::StateParamMap,u::AbstractVector,φ::AbstractVector)
  U,V_φ,U_reg = u_to_j.spaces
  uh = FEFunction(U,u)
  φh = FEFunction(V_φ,φ)
  return ChainRulesCore.rrule(u_to_j,uh,φh)
end

# Backwards compat
const StateParamIntegrandWithMeasure = StateParamMap

# IO
function Base.show(io::IO,object::AbstractStateParamMap)
  print(io,"$(nameof(typeof(object)))")
end