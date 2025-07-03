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
    struct EmbeddedPDEConstrainedFunctionals{N,T} <: AbstractPDEConstrainedFunctionals{N}

A version of `PDEConstrainedFunctionals` that has an `embedded_collection` to
allow the `state_map` to be updated given new FE spaces for the forward problem.
This is currently required for unfitted methods.
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

    @check Set((:state_map,:J,:C)) == keys(embedded_collection.objects) """
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

############## Zygote Compat ##############
"""
    CustomPDEConstrainedFunctionals{N,A} <:  AbstractPDEConstrainedFunctionals{N}

A version of `PDEConstrainedFunctionals` that allows for an arbitrary mapping
`φ_to_jc` that is used to compute the objective and constraints given the primal variable.

Under the hood, we use Zygote to compute the Jacobian of this mapping with the rrules defined
throughout GridapTopOpt.

# Parameters

- `φ_to_jc`: A function that defines the mapping from the primal variable `φ` to the
  objective and constraints. This should accept AbstractVector `φ` (the free values
  of `φh`) and output a scalar. In side this function, you should compute your states
  and evaluate your objectives and constraints that should be written as `GridapTopOpt.StateParamMap`.
  For example,
  ```julia
  using GridapTopOpt: StateParamMap
  ...
  J = StateParamMap(j,state_map)
  C = StateParamMap(c,state_map)
  function φ_to_jc(φ)
    u = state_map(φ)
    [J(u,φ),C(u,φ)^2]
  end
  pcfs = CustomPDEConstrainedFunctionals(φ_to_jc,state_map,φh)
  ```
- `analytic_dJ`: Either a `Function` (if using analytic derivative) or nothing if using AD.
- `analytic_dC`: A vector of `Nothing` or `Function` depending on whether using analytic derivatives or AD.
- `state_map::A`: The state map for the problem. NOTE: this is a place holder for the optimiser
  output and in theory you could use many different state maps inside `φ_to_jc`.


!!! warning
    The expected function for `analytic_dJ` and functions of `analytic_dC` are different to
    usual. Here, you should define a function that takes an `AbstractVector` input corresponding to
    the derivative and the primal variable dofs `φ` and assembles the derivative into the
    `AbstractVector` input. For example,
    ```julia
    function analytic_dJ!(dJ,φ)
      φh = FEFunction(V_φ,φ)
      uh = get_state(state_map)
      _dJ(q) = ∫(q*...)dΩ
      Gridap.FESpaces.assemble_vector!(_dJ,dJ,V_φ)
    end
    ```
    This functionality is subject to change.
"""
struct CustomPDEConstrainedFunctionals{N,A} <:  AbstractPDEConstrainedFunctionals{N}
  φ_to_jc :: Function
  analytic_dJ
  analytic_dC
  state_map :: A

    @doc"""
        CustomPDEConstrainedFunctionals(
          φ_to_jc :: Function,
          num_constraints;
          state_map :: Union{Nothing,AbstractFEStateMap,Vector{<:AbstractFEStateMap}},
          analytic_dJ = nothing,
          analytic_dC = fill(nothing,num_constraints)
        )

    Create an instance of `CustomPDEConstrainedFunctionals`. Here,
    `num_constraints` specifies the number of constraints.

    !!! note
        The `state_map` field is used to get the current state of the forward problems
        for the uh output in
        ```julia
        for (it, uh, φh) in optimiser
          ...
        end
        ```
        If you take state_map=nothing, then `get_state`, and the corresponding
        output of uh above will be `nothing`.

        This functionality is subject to change.
    """
    function CustomPDEConstrainedFunctionals(
      φ_to_jc :: Function,
      num_constraints; # <- can we get this from lowered φ_to_jc
      state_map :: Union{Nothing,AbstractFEStateMap,Vector{<:AbstractFEStateMap}} = nothing,
      analytic_dJ = nothing,
      analytic_dC = fill(nothing,num_constraints)
    )
    return new{num_constraints,typeof(state_map)}(φ_to_jc,analytic_dJ,analytic_dC,state_map)
  end
end

get_state(m::CustomPDEConstrainedFunctionals) = get_state(m.state_map)
get_state(::CustomPDEConstrainedFunctionals{N,Nothing}) where N = nothing

function Fields.evaluate!(pcf::CustomPDEConstrainedFunctionals{N},φh) where N
  φ_to_jc = pcf.φ_to_jc
  analytic_dJ!, analytic_dC! = pcf.analytic_dJ, pcf.analytic_dC

  # Compute derivatives
  ignore_pullback = findall(!isnothing,vcat(analytic_dJ!, analytic_dC!))
  val, grad = val_and_jacobian(φ_to_jc, get_free_dof_values(φh);ignore_pullback)

  # Unpack
  j = val[1]
  c = val[2:end]
  dJ = grad[1][1]
  dC = grad[1][2:end]

  # Analytic derivatives
  function _compute_dF!(dF,analytic_dF!::Function)
    analytic_dF!(dF,get_free_dof_values(φh))
    nothing
  end
  function _compute_dF!(dF,analytic_dF!::Nothing)
    nothing
  end
  _compute_dF!(dJ,analytic_dJ!)
  map(_compute_dF!,dC,analytic_dC!)

  return j,c,dJ,dC
end

function Fields.evaluate!(pcf::CustomPDEConstrainedFunctionals{0},φh)
  φ_to_jc = pcf.φ_to_jc
  analytic_dJ!, analytic_dC! = pcf.analytic_dJ, pcf.analytic_dC

  # Compute derivatives
  ignore_pullback = findall(!isnothing,vcat(analytic_dJ!, analytic_dC!))
  val, _grad = val_and_jacobian(φ_to_jc, get_free_dof_values(φh);ignore_pullback)

  # Unpack
  j = val[1]
  c = Vector{eltype(val)}()
  grad = first(_grad)
  dJ = grad[1]
  dC = Vector{eltype(grad)}();

  # Analytic derivative
  function _compute_dF!(dF,analytic_dF!::Function)
    analytic_dF!(dF,get_free_dof_values(φh))
    nothing
  end
  function _compute_dF!(dF,analytic_dF!::Nothing)
    nothing
  end
  _compute_dF!(dJ,pcf.analytic_dJ)

  return j,c,dJ,dC
end

function evaluate_functionals!(pcf::CustomPDEConstrainedFunctionals,φh)
  φ = get_free_dof_values(φh)
  return evaluate_functionals!(pcf,φ)
end

function evaluate_functionals!(pcf::CustomPDEConstrainedFunctionals{N},φ::AbstractVector) where N
  φ_to_jc =  pcf.φ_to_jc
  val = φ_to_jc(φ)
  j = val[1]
  c = val[2:end];
  return j,c
end

function evaluate_functionals!(pcf::CustomPDEConstrainedFunctionals{0},φ::AbstractVector)
  φ_to_jc =  pcf.φ_to_jc
  val = φ_to_jc(φ)
  j = val[1]
  c = Vector{eltype(val)}();
  return j,c
end

########## Zygote + Unfitted ##########
"""
    struct CustomEmbeddedPDEConstrainedFunctionals{N,A} <: AbstractPDEConstrainedFunctionals{N}

A version of `CustomPDEConstrainedFunctionals` that has an `embedded_collection` to
allow the `state_map` to be updated given new FE spaces for the forward problem.
This is currently required for unfitted methods.
"""
struct CustomEmbeddedPDEConstrainedFunctionals{N,A} <:  AbstractPDEConstrainedFunctionals{N}
  φ_to_jc :: Function
  analytic_dJ
  analytic_dC
  embedded_collection :: EmbeddedCollection
    @doc"""
        CustomEmbeddedPDEConstrainedFunctionals(
          φ_to_jc :: Function,
          num_constraints,
          embedded_collection :: EmbeddedCollection;
          analytic_dJ = nothing,
          analytic_dC = fill(nothing,num_constraints)
        )

    Create an instance of `CustomEmbeddedPDEConstrainedFunctionals`. Here,
    `num_constraints` specifies the number of constraints.

    !!! note
        If you have one or more `state_map` objects in `embedded_collection`, you should
        include these as a vector under the `:state_map` key of the `embedded_collection`.
        This is used in the optimiser to get the current state of the maps for the uh
        output in
        ```julia
        for (it, uh, φh) in optimiser
          ...
        end
        ```
        If you do not have a `:state_map` in the `embedded_collection`, then `get_state`,
        and the corresponding output of uh above will be `nothing`.

        This functionality is subject to change.
    """
    function CustomEmbeddedPDEConstrainedFunctionals(
      φ_to_jc :: Function,
      num_constraints,
      embedded_collection :: EmbeddedCollection;
      analytic_dJ = nothing,
      analytic_dC = fill(nothing,num_constraints)
    )

    state_map = :state_map ∈ keys(embedded_collection.objects) ? embedded_collection.state_map : nothing;
    A = typeof(state_map)
    return new{num_constraints,A}(φ_to_jc,analytic_dJ,analytic_dC,embedded_collection)
  end
end

get_state(m::CustomEmbeddedPDEConstrainedFunctionals) = get_state(m.state_map)
get_state(::CustomEmbeddedPDEConstrainedFunctionals{N,Nothing}) where N = nothing

function Fields.evaluate!(pcf::CustomEmbeddedPDEConstrainedFunctionals,φh;update_space::Bool=true)
  update_space && update_collection_with_φh!(pcf.embedded_collection,φh)
  φ_to_jc = pcf.φ_to_jc
  analytic_dJ!, analytic_dC! = pcf.analytic_dJ, pcf.analytic_dC

  # Compute derivatives
  ignore_pullback = findall(!isnothing,vcat(analytic_dJ!, analytic_dC!))
  val, grad = val_and_jacobian(φ_to_jc, get_free_dof_values(φh);ignore_pullback)

  # Unpack
  j = val[1]
  c = val[2:end]
  dJ = grad[1][1]
  dC = grad[1][2:end]

  # Analytic derivatives
  function _compute_dF!(dF,analytic_dF!::Function)
    analytic_dF!(dF,get_free_dof_values(φh))
    nothing
  end
  function _compute_dF!(dF,analytic_dF!::Nothing)
    nothing
  end
  _compute_dF!(dJ,analytic_dJ!)
  map(_compute_dF!,dC,analytic_dC!)

  return j,c,dJ,dC
end

function Fields.evaluate!(pcf::CustomEmbeddedPDEConstrainedFunctionals{0},φh;update_space::Bool=true)
  update_space && update_collection_with_φh!(pcf.embedded_collection,φh)
  φ_to_jc = pcf.φ_to_jc
  analytic_dJ!, analytic_dC! = pcf.analytic_dJ, pcf.analytic_dC

  # Compute derivatives
  ignore_pullback = findall(!isnothing,vcat(analytic_dJ!, analytic_dC!))
  val, _grad = val_and_jacobian(φ_to_jc, get_free_dof_values(φh);ignore_pullback)

  # Unpack
  j = val[1]
  c = Vector{eltype(val)}()
  grad = first(_grad)
  dJ = grad[1]
  dC = Vector{eltype(grad)}();

  # Analytic derivative
  function _compute_dF!(dF,analytic_dF!::Function)
    analytic_dF!(dF,get_free_dof_values(φh))
    nothing
  end
  function _compute_dF!(dF,analytic_dF!::Nothing)
    nothing
  end
  _compute_dF!(dJ,pcf.analytic_dJ)

  return j,c,dJ,dC
end

function evaluate_functionals!(pcf::CustomEmbeddedPDEConstrainedFunctionals,φh;update_space::Bool=true)
  update_space && update_collection_with_φh!(pcf.embedded_collection,φh)
  val = pcf.φ_to_jc(get_free_dof_values(φh))
  j = val[1]
  c = val[2:end];
  return j,c
end

function evaluate_functionals!(pcf::CustomEmbeddedPDEConstrainedFunctionals{0},φh;update_space::Bool=true)
  update_space && update_collection_with_φh!(pcf.embedded_collection,φh)
  val = pcf.φ_to_jc(get_free_dof_values(φh))
  j = val[1]
  c = Vector{eltype(val)}();
  return j,c
end