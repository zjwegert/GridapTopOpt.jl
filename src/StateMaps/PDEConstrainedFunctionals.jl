abstract type AbstractPDEConstrainedFunctionals{N} end

"""
    struct PDEConstrainedFunctionals{N,A} <: AbstractPDEConstrainedFunctionals{N}

An object that computes the objective, constraints, and their derivatives.

# Implementation

This implementation computes derivatives of a integral quantity

``F(u(\\varphi),\\varphi,\\mathrm{d}\\Omega_1,\\mathrm{d}\\Omega_2,...) =
\\Sigma_{i}\\int_{\\Omega_i} f_i(\\varphi)~\\mathrm{d}\\Omega``

with respect to an auxiliary parameter ``\\varphi`` where ``u``
is the solution to a PDE and implicitly depends on ``\\varphi``.
This requires two pieces of information:

 1) Computation of ``\\frac{\\partial F}{\\partial u}`` and
    ``\\frac{\\partial F}{\\partial \\varphi}`` (handled by [`StateParamMap `](@ref)).
 2) Computation of ``\\frac{\\partial F}{\\partial u}
    \\frac{\\partial u}{\\partial \\varphi}`` at ``\\varphi`` and ``u``
    using the adjoint method (handled by [`AbstractFEStateMap`](@ref)). I.e., let

    ``\\frac{\\partial F}{\\partial u}
    \\frac{\\partial u}{\\partial \\varphi} = -\\lambda^\\intercal
    \\frac{\\partial \\mathcal{R}}{\\partial \\varphi}``

    where ``\\mathcal{R}`` is the residual and solve the (linear) adjoint
    problem:

    ``\\frac{\\partial \\mathcal{R}}{\\partial u}^\\intercal\\lambda =
    \\frac{\\partial F}{\\partial u}^\\intercal.``

The gradient is then ``\\frac{\\partial F}{\\partial \\varphi} =
\\frac{\\partial F}{\\partial \\varphi} -
\\frac{\\partial F}{\\partial u}\\frac{\\partial u}{\\partial \\varphi}``.

# Parameters

- `J`: A `StateParamMap` corresponding to the objective.
- `C`: A vector of `StateParamMap` corresponding to the constraints.
- `dJ`: The DoFs for the objective sensitivity.
- `dC`: The DoFs for each constraint sensitivity.
- `analytic_dJ`: a `Function` for computing the analytic objective sensitivity.
- `analytic_dC`: A vector of `Function` for computing the analytic objective sensitivities.
- `state_map::A`: The state map for the problem.

# Note

- If `analytic_dJ = nothing` automatic differentiation will be used.
- If `analytic_dC[i] = nothing` automatic differentiation will be used for `C[i]`.
"""
struct PDEConstrainedFunctionals{N,A} <: AbstractPDEConstrainedFunctionals{N}
  J
  C
  dJ
  dC
  analytic_dJ
  analytic_dC
  state_map :: A

  function PDEConstrainedFunctionals(
      objective   :: AbstractStateParamMap,
      constraints :: Vector{<:AbstractStateParamMap},
      state_map   :: AbstractFEStateMap;
      analytic_dJ = nothing,
      analytic_dC = fill(nothing,length(constraints)))

    # Preallocate
    dJ = similar(get_∂F∂φ_vec(objective))
    dC = map(Ci->similar(get_∂F∂φ_vec(Ci)),constraints)

    N = length(constraints)
    T = typeof(state_map)
    return new{N,T}(objective,constraints,dJ,dC,analytic_dJ,analytic_dC,state_map)
  end
end

"""
PDEConstrainedFunctionals(objective::Function,constraints::Vector{<:Function},
  state_map::AbstractFEStateMap;analytic_dJ;analytic_dC)

Create an instance of `PDEConstrainedFunctionals`. The arguments for the objective
and constraints must follow the specification in [`StateParamMap`](@ref).
By default we use automatic differentation for the objective and all constraints. This
can be disabled by passing the shape derivative as a type `Function` to `analytic_dJ`
and/or entires in `analytic_dC`.
"""
function PDEConstrainedFunctionals(
    objective   :: Function,
    constraints :: Vector{<:Function},
    state_map   :: Union{AffineFEStateMap,NonlinearFEStateMap,RepeatingAffineFEStateMap};
    analytic_dJ = nothing,
    analytic_dC = fill(nothing,length(constraints)))

  # Create StateParamMaps
  J = StateParamMap(objective,state_map)
  C = map(Ci -> StateParamMap(Ci,state_map),constraints)
  if isempty(C)
    C = StateParamMap[]
  end

  return PDEConstrainedFunctionals(J,C,state_map;analytic_dJ,analytic_dC)
end

"""
    PDEConstrainedFunctionals(objective,state_map;analytic_dJ)

Create an instance of `PDEConstrainedFunctionals` when the problem has no constraints.
"""
PDEConstrainedFunctionals(J,state_map::AbstractFEStateMap;analytic_dJ=nothing) =
  PDEConstrainedFunctionals(J,typeof(J)[],state_map;analytic_dJ = analytic_dJ,analytic_dC = Nothing[])

## PDEConstrainedFunctionals for StaggeredStateParamMap
function PDEConstrainedFunctionals(
    objective   :: Function,
    ∂J∂xhi,
    constraints :: Vector{<:Function},
    ∂Cj∂xhi,
    state_map   :: StaggeredFEStateMapTypes;
    analytic_dJ = nothing,
    analytic_dC = fill(nothing,length(constraints)))

  # Create StateParamMaps
  J = StaggeredStateParamMap(objective,∂J∂xhi,state_map)
  C = map((Cj,∂Cj∂xhi) -> StaggeredStateParamMap(Cj,∂Cj∂xhi,state_map),constraints,∂Cj∂xhi)
  if isempty(C)
    C = StaggeredStateParamMap[]
  end

  return PDEConstrainedFunctionals(J,C,state_map;analytic_dJ,analytic_dC)
end

function PDEConstrainedFunctionals(
  objective   :: Function,
  constraints :: Vector{<:Function},
  state_map   :: StaggeredFEStateMapTypes;
  analytic_dJ = nothing,
  analytic_dC = fill(nothing,length(constraints))
)
  # Create StateParamMaps
  J = StaggeredStateParamMap(objective,state_map)
  C = map(Cj -> StaggeredStateParamMap(Cj,state_map),constraints)
  if isempty(C)
    C = StaggeredStateParamMap[]
  end

  return PDEConstrainedFunctionals(J,C,state_map;analytic_dJ,analytic_dC)
end

PDEConstrainedFunctionals(J,∂J∂xhi::Tuple{Vararg{Function}},state_map::StaggeredFEStateMapTypes;analytic_dJ=nothing) =
  PDEConstrainedFunctionals(J,∂J∂xhi,typeof(J)[],typeof(J)[],state_map;analytic_dJ = analytic_dJ,analytic_dC = Nothing[])

PDEConstrainedFunctionals(J,state_map::StaggeredFEStateMapTypes;analytic_dJ=nothing) =
  PDEConstrainedFunctionals(J,typeof(J)[],state_map;analytic_dJ = analytic_dJ,analytic_dC = Nothing[])

get_state_map(m::PDEConstrainedFunctionals) = m.state_map
get_state(m::PDEConstrainedFunctionals) = get_state(get_state_map(m))

"""
    evaluate_functionals!(pcf::PDEConstrainedFunctionals,φh)

Evaluate the objective and constraints at `φh`.
"""
function evaluate_functionals!(pcf::PDEConstrainedFunctionals,φh;kwargs...)
  u  = get_state_map(pcf)(φh)
  U  = get_trial_space(get_state_map(pcf))
  uh = FEFunction(U,u)
  return pcf.J(uh,φh), map(Ci->Ci(uh,φh),pcf.C)
end

function evaluate_functionals!(pcf::PDEConstrainedFunctionals,φ::AbstractVector;kwargs...)
  V_φ = get_aux_space(get_state_map(pcf))
  φh = FEFunction(V_φ,φ)
  return evaluate_functionals!(pcf,φh;kwargs...)
end

"""
    evaluate_derivatives!(pcf::PDEConstrainedFunctionals,φh)

Evaluate the derivatives of the objective and constraints at `φh`.
"""
function evaluate_derivatives!(pcf::PDEConstrainedFunctionals,φh;kwargs...)
  _,_,dJ,dC = evaluate!(pcf,φh)
  return dJ,dC
end

function evaluate_derivatives!(pcf::PDEConstrainedFunctionals,φ::AbstractVector;kwargs...)
  V_φ = get_aux_space(get_state_map(pcf))
  φh = FEFunction(V_φ,φ)
  return evaluate_derivatives!(pcf,φh;kwargs...)
end

"""
    Fields.evaluate!(pcf::PDEConstrainedFunctionals,φh)

Evaluate the objective and constraints, and their derivatives at
`φh`.
"""
function Fields.evaluate!(pcf::PDEConstrainedFunctionals,φh;kwargs...)
  J, C, dJ, dC = pcf.J,pcf.C,pcf.dJ,pcf.dC
  analytic_dJ  = pcf.analytic_dJ
  analytic_dC  = pcf.analytic_dC
  U = get_trial_space(get_state_map(pcf))

  V_φ = get_aux_space(get_state_map(pcf))
  deriv_assem = get_deriv_assembler(get_state_map(pcf))

  ## Foward problem
  u, u_pullback = rrule(get_state_map(pcf),φh)
  uh = FEFunction(U,u)

  function ∇!(F::AbstractStateParamMap,dF,::Nothing)
    # Automatic differentation
    j_val, j_pullback = rrule(F,uh,φh)   # Compute functional and pull back
    _, dFdu, dFdφ     = j_pullback(1)    # Compute dFdu, dFdφ
    _, dφ_adj         = u_pullback(dFdu) # Compute -dFdu*dudφ via adjoint
    copy!(dF,dφ_adj)
    dF .+= dFdφ
    return j_val
  end
  function ∇!(F::AbstractStateParamMap,dF,dF_analytic::Function)
    # Analytic shape derivative
    j_val = F(uh,φh)
    _dF(q) = dF_analytic(q,uh,φh)
    assemble_vector!(_dF,dF,deriv_assem,V_φ)
    return j_val
  end
  j = ∇!(J,dJ,analytic_dJ)
  c = map(∇!,C,dC,analytic_dC)

  return j,c,dJ,dC
end

function Fields.evaluate!(pcf::PDEConstrainedFunctionals,φ::AbstractVector;kwargs...)
  V_φ = get_aux_space(get_state_map(pcf))
  φh = FEFunction(V_φ,φ)
  return evaluate!(pcf,φh;kwargs...)
end

"""
    mutable struct EmbeddedPDEConstrainedFunctionals{N} <: AbstractPDEConstrainedFunctionals{N}

A mutable version of `PDEConstrainedFunctionals` that allows `state_map` to be
updated given new FE spaces for the forward problem. This is currently required
for body-fitted mesh methods and unfitted methods.
"""
struct EmbeddedPDEConstrainedFunctionals{N,T} <: AbstractPDEConstrainedFunctionals{N}
  dJ
  dC
  analytic_dJ
  analytic_dC
  embedded_collection

  @doc """
      EmbeddedPDEConstrainedFunctionals(objective::Function,constraints::Vector{<:Function},
        embedded_collection :: EmbeddedCollection;analytic_dJ;analytic_dC)

  Create an instance of `EmbeddedPDEConstrainedFunctionals`.

  The embedded_collection must be a `EmbeddedCollection` that contains the
  `:state_map`, `:J`, and `:C` objects.
  """
  function EmbeddedPDEConstrainedFunctionals(
      embedded_collection :: EmbeddedCollection;
      analytic_dJ = nothing,
      analytic_dC = nothing)

    @assert Set((:state_map,:J,:C)) == keys(embedded_collection.objects) """
    Expected EmbeddedCollection to have objects ':state_map,:J,:C'. Ensure that you
    have updated the collection after adding new recipes.

    You have $(keys(embedded_collection.objects))

    Note:
    - We require that this EmbeddedCollection is seperate to the one used for the
      UnfittedEvolution. This is because updating the FEStateMap is more expensive than
      cutting and there are instances where evolution and reinitialisation happen
      at before recomputing the forward solution. As such, we cut an extra time
      to avoid allocating the state map more often then required.
    - For problems with no constraints `:C` must at least point to an empty list
    """
    # Preallocate
    dJ = similar(get_∂F∂φ_vec(embedded_collection.J))
    dC = map(Ci->similar(get_∂F∂φ_vec(Ci)),embedded_collection.C)

    N = length(embedded_collection.C)
    if analytic_dC isa Nothing
      analytic_dC = fill(nothing,length(N))
    end

    T = typeof(embedded_collection.state_map)
    return new{N,T}(dJ,dC,analytic_dJ,analytic_dC,embedded_collection)
  end
end

get_state_map(m::EmbeddedPDEConstrainedFunctionals) = m.embedded_collection.state_map
get_state(m::EmbeddedPDEConstrainedFunctionals) = get_state(get_state_map(m))

"""
    evaluate_functionals!(pcf::EmbeddedPDEConstrainedFunctionals,φh;update_space::Bool=true)

Evaluate the objective and constraints at `φh`.

!!! warning
    Taking `update_space = false` will NOT update the underlying finite element
    spaces and assemblers that depend on `φh`. This should only be used
    when you are certain that `φh` has not been updated.
"""
function evaluate_functionals!(pcf::EmbeddedPDEConstrainedFunctionals,φh;update_space::Bool=true)
  update_space && update_collection_with_φh!(pcf.embedded_collection,φh)
  u  = get_state_map(pcf)(φh)
  U  = get_trial_space(get_state_map(pcf))
  uh = FEFunction(U,u)
  J = pcf.embedded_collection.J
  C = pcf.embedded_collection.C
  return J(uh,φh), map(Ci->Ci(uh,φh),C)
end

function evaluate_functionals!(pcf::EmbeddedPDEConstrainedFunctionals,φ::AbstractVector;kwargs...)
  V_φ = get_aux_space(get_state_map(pcf))
  φh  = FEFunction(V_φ,φ)
  return evaluate_functionals!(pcf,φh;kwargs...)
end

"""
    evaluate_derivatives!(pcf::EmbeddedPDEConstrainedFunctionals,φh;update_space::Bool=true)

Evaluate the derivatives of the objective and constraints at `φh`.

!!! warning
    Taking `update_space = false` will NOT update the underlying finite element
    spaces and assemblers that depend on `φh`. This should only be used
    when you are certain that `φh` has not been updated.
"""
function evaluate_derivatives!(pcf::EmbeddedPDEConstrainedFunctionals,φh;update_space::Bool=true)
  update_space && update_collection_with_φh!(pcf.embedded_collection,φh)
  _,_,dJ,dC = evaluate!(pcf,φh)
  return dJ,dC
end

function evaluate_derivatives!(pcf::EmbeddedPDEConstrainedFunctionals,φ::AbstractVector;kwargs...)
  V_φ = get_aux_space(get_state_map(pcf))
  φh = FEFunction(V_φ,φ)
  return evaluate_derivatives!(pcf,φh;kwargs...)
end

"""
    Fields.evaluate!(pcf::EmbeddedPDEConstrainedFunctionals,φh;update_space::Bool=true)

Evaluate the objective and constraints, and their derivatives at
`φh`.

!!! warning
    Taking `update_space = false` will NOT update the underlying finite element
    spaces and assemblers that depend on `φh`. This should only be used
    when you are certain that `φh` has not been updated.
"""
function Fields.evaluate!(pcf::EmbeddedPDEConstrainedFunctionals,φh;update_space::Bool=true)
  update_space && update_collection_with_φh!(pcf.embedded_collection,φh)

  J           = pcf.embedded_collection.J
  C           = pcf.embedded_collection.C
  dJ          = pcf.dJ
  dC          = pcf.dC
  analytic_dJ = pcf.analytic_dJ
  analytic_dC = pcf.analytic_dC
  state_map   = get_state_map(pcf)
  U           = get_trial_space(state_map)

  V_φ = get_aux_space(state_map)
  deriv_assem = get_deriv_assembler(state_map)

  ## Foward problem
  u, u_pullback = rrule(state_map,φh)
  uh = FEFunction(U,u)

  function ∇!(F::AbstractStateParamMap,dF,::Nothing)
    # Automatic differentation
    j_val, j_pullback = rrule(F,uh,φh)   # Compute functional and pull back
    _, dFdu, dFdφ     = j_pullback(1)    # Compute dFdu, dFdφ
    _, dφ_adj         = u_pullback(dFdu) # Compute -dFdu*dudφ via adjoint
    copy!(dF,dφ_adj)
    dF .+= dFdφ
    return j_val
  end
  function ∇!(F::AbstractStateParamMap,dF,dF_analytic::Function)
    # Analytic shape derivative
    j_val = F(uh,φh)
    _dF(q) = dF_analytic(q,uh,φh)
    assemble_vector!(_dF,dF,deriv_assem,V_φ)
    return j_val
  end
  j = ∇!(J,dJ,analytic_dJ)
  c = map(∇!,C,dC,analytic_dC)

  return j,c,dJ,dC
end

function Fields.evaluate!(pcf::EmbeddedPDEConstrainedFunctionals,φ::AbstractVector;kwargs...)
  V_φ = get_aux_space(get_state_map(pcf))
  φh = FEFunction(V_φ,φ)
  return evaluate!(pcf,φh;kwargs...)
end

"""
    EmbeddedCollection_in_φh(recipes::Union{<:Function,Vector{<:Function}},bgmodel,φ0)

Returns an `EmbeddedCollection` whoose recipes are only updated using the
parameter `φ0`. This is useful for problems where the recipes are not computed
using the cut geometry information.
"""
function EmbeddedCollection_in_φh(recipes::Union{<:Function,Vector{<:Function}},bgmodel,φ0)
  c = EmbeddedCollection(recipes,bgmodel)
  update_collection_with_φh!(c,φ0)
end

function update_collection_with_φh!(c::EmbeddedCollection,φh)
    for r in c.recipes
    merge!(c.objects,pairs(r(φh)))
  end
  return c
end

# IO
function Base.show(io::IO,::MIME"text/plain",f::AbstractPDEConstrainedFunctionals{N}) where N
  print(io,"$(nameof(typeof(f))):
    num_constraints: $N")
end





struct CustomPDEConstrainedFunctionals{N,A} <:  AbstractPDEConstrainedFunctionals{N}
  φ_to_jc :: Function
  dJ :: Vector{Float64}
  dC :: Vector{Vector{Float64}}
  analytic_dJ
  analytic_dC
  state_map :: A
  V_φ :: FESpace

    function CustomPDEConstrainedFunctionals(
      φ_to_jc :: Function,
      state_map :: AbstractFEStateMap,
      φh_bg;
    )

    # V_φ = φh_bg.fe_space
    # φh = interpolate(φh_bg,get_aux_space(state_map))
    # φ = φh.free_values

    φ = φh_bg.free_values
    V_φ = φh_bg.fe_space
    
    # Pre-allocaitng
    grad = Zygote.jacobian(φ_to_jc, φ)
    dJ = grad[1][1,:]
    dC = [collect(row) for row in eachrow(grad[1][2:end,:])]    

    N = length(dC)
    A = typeof(state_map)
    analytic_dJ = nothing
    analytic_dC = fill(nothing,N)

    return new{N,A}(φ_to_jc,dJ,dC,analytic_dJ,analytic_dC,state_map,V_φ)
  end
end

function Fields.evaluate!(pcf::CustomPDEConstrainedFunctionals,φh_bg)
  φ_to_jc,dJ,dC = pcf.φ_to_jc,pcf.dJ,pcf.dC

  φh = interpolate(φh_bg,get_aux_space(pcf.state_map))

  obj,grad = Zygote.withjacobian(φ_to_jc, φh.free_values)
  j = obj[1]
  c = obj[2:end]
  copy!(dJ,grad[1][1,:])
  copy!(dC,[collect(row) for row in eachrow(grad[1][2:end,:])])

  return j,c,dJ,dC
end

get_state(m::CustomPDEConstrainedFunctionals) = get_state(m.state_map)

function evaluate_functionals!(pcf::CustomPDEConstrainedFunctionals,φh::FEFunction)
  φ = φh.free_values
  return evaluate_functionals!(pcf,φ)
end

function evaluate_functionals!(pcf::CustomPDEConstrainedFunctionals,φ_bg::AbstractVector)
  φ_to_jc =  pcf.φ_to_jc
  φh_bg = FEFunction(pcf.V_φ,φ_bg)
  φh = interpolate(φh_bg,get_aux_space(pcf.state_map))
  φ = φh.free_values
  obj = φ_to_jc(φ)
  j = obj[1]
  c = obj[2:end]
  return j,c
end

struct CustomEmbeddedPDEConstrainedFunctionals{N,A} <:  AbstractPDEConstrainedFunctionals{N}
  φ_to_jc :: Function
  dJ :: Vector{Float64}
  dC :: Vector{Vector{Float64}}
  analytic_dJ
  analytic_dC
  embedded_collection :: EmbeddedCollection
  Ωs :: EmbeddedCollection
  V_φ :: FESpace

    function CustomEmbeddedPDEConstrainedFunctionals(
      φ_to_jc :: Function,
      embedded_collection :: EmbeddedCollection,
      Ωs,
      φh_bg;
    )
    update_collection!(Ωs,φh_bg)
    update_collection_with_φh!(embedded_collection,φh_bg)

    V_φ = φh_bg.fe_space
    φh = φh_bg #interpolate(φh_bg,get_aux_space(embedded_collection.state_map))
    φ = φh.free_values
    
    # Pre-allocaitng
    grad = Zygote.jacobian(φ_to_jc,φ)
    dJ = grad[1][1,:]
    dC = [collect(row) for row in eachrow(grad[1][2:end,:])]

    N = length(dC)
    A = typeof(embedded_collection.state_map)
    analytic_dJ = nothing
    analytic_dC = fill(nothing,N)

    return new{N,A}(φ_to_jc,dJ,dC,analytic_dJ,analytic_dC,embedded_collection,Ωs,V_φ)
  end
end

get_state_map(m::CustomEmbeddedPDEConstrainedFunctionals) = m.embedded_collection.state_map
get_state(m::CustomEmbeddedPDEConstrainedFunctionals) = get_state(get_state_map(m))

function Fields.evaluate!(pcf::CustomEmbeddedPDEConstrainedFunctionals,φh_bg)
  φ_to_jc,dJ,dC = pcf.φ_to_jc,pcf.dJ,pcf.dC
  state_collection = pcf.embedded_collection
  Ωs = pcf.Ωs
  update_collection!(Ωs,φh_bg)
  update_collection_with_φh!(state_collection,φh_bg)
  φh = φh_bg # interpolate(φh_bg,get_aux_space(state_collection.state_map))
  obj,grad = Zygote.withjacobian(φ_to_jc, φh.free_values)
  j = obj[1]
  c = obj[2:end]
  copy!(dJ,grad[1][1,:])
  copy!(dC,[collect(row) for row in eachrow(grad[1][2:end,:])])
  return j,c,dJ,dC
end

get_state(m::CustomEmbeddedPDEConstrainedFunctionals) = get_state(m.embedded_collection.state_map)

function evaluate_functionals!(pcf::CustomEmbeddedPDEConstrainedFunctionals,φh::FEFunction)
  φ = φh.free_values
  return evaluate_functionals!(pcf,φ)
end

function evaluate_functionals!(pcf::CustomEmbeddedPDEConstrainedFunctionals,φ_bg::AbstractVector)
  φ_to_jc =  pcf.φ_to_jc
  state_collection = pcf.embedded_collection
  Ωs = pcf.Ωs
  φh_bg = FEFunction(pcf.V_φ,φ_bg)
  update_collection!(Ωs,φh_bg)
  update_collection_with_φh!(state_collection,φh_bg)
  φh = φh_bg #interpolate(φh_bg,get_aux_space(state_collection.state_map))
  φ = φh.free_values
  obj = φ_to_jc(φ)
  j = obj[1]
  c = obj[2:end]
  return j,c
end