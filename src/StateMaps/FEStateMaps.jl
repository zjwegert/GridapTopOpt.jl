"""
    abstract type AbstractFEStateMap

Types inheriting from this abstract type should enable the evaluation and differentiation of
the solution to an FE problem `u` that implicitly depends on an auxiliary parameter `φ`.
"""
abstract type AbstractFEStateMap end

"""
    get_state(m::AbstractFEStateMap)

Return the solution/state `u` to the FE problem.
"""
get_state(::AbstractFEStateMap) = @abstractmethod

"""
    get_spaces(m::AbstractFEStateMap)

Return a collection of FE spaces. The first four entires should correspond to
[`get_trial_space`](@ref), [`get_test_space`](@ref), [`get_aux_space`](@ref), and
[`get_deriv_space`](@ref) unless these are overloaded for a particular implementation.
"""
get_spaces(::AbstractFEStateMap) = @abstractmethod

"""
    get_assemblers(m::AbstractFEStateMap)

Return a collection of assemblers. The first two entires should correspond to
[`get_pde_assembler`](@ref) and [`get_deriv_assembler`](@ref) unless these are
overloaded for a particular implementation.
"""
get_assemblers(::AbstractFEStateMap) = @abstractmethod

"""
    get_trial_space(m::AbstractFEStateMap)

Return trial space for FE problem.
"""
get_trial_space(m::AbstractFEStateMap) = get_spaces(m)[1]

"""
    get_test_space(m::AbstractFEStateMap)

Return test space for FE problem.
"""
get_test_space(m::AbstractFEStateMap) = get_spaces(m)[2]

"""
    get_aux_space(m::AbstractFEStateMap)

Return space for auxillary parameter.
"""
get_aux_space(m::AbstractFEStateMap) = get_spaces(m)[3]

"""
    get_deriv_space(m::AbstractFEStateMap)

Return space for derivatives.
"""
get_deriv_space(m::AbstractFEStateMap) = get_spaces(m)[4]

"""
    get_pde_assembler(m::AbstractFEStateMap)

Return assembler for FE problem.
"""
get_pde_assembler(m::AbstractFEStateMap) = get_assemblers(m)[1]

"""
    get_deriv_assembler(m::AbstractFEStateMap)

Return assembler for derivatives.
"""
get_deriv_assembler(m::AbstractFEStateMap) = get_assemblers(m)[2]

"""
    (φ_to_u::AbstractFEStateMap)(φh)

Evaluate the forward problem `u` given `φ`. This should compute the
FE problem.
"""
@inline (φ_to_u::AbstractFEStateMap)(φh) = forward_solve!(φ_to_u,φh)

"""
    forward_solve!(φ_to_u::AbstractFEStateMap,φh)

Evaluate the forward problem `u` given `φ`. This should compute the
FE problem.
"""
function forward_solve!(φ_to_u::AbstractFEStateMap,φh)
  @abstractmethod
end

"""
    update_adjoint_caches!(φ_to_u::AbstractFEStateMap,uh,φh)

Update the cache for the adjoint problem. This is usually a tuple
of objects.
"""
function update_adjoint_caches!(φ_to_u::AbstractFEStateMap,uh,φh)
  @abstractmethod
end

function update_adjoint_caches!(φ_to_u::AbstractFEStateMap,u::AbstractVector,φ::AbstractVector)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return update_adjoint_caches!(φ_to_u,uh,φh)
end

"""
    adjoint_solve!(φ_to_u::AbstractFEStateMap,du::AbstractVector)

Evaluate the solution to the adjoint problem given a RHS vector `∂F∂u` denoted `du`.
This should solve the linear problem `dRduᵀ*λ = ∂F∂uᵀ`.
"""
function adjoint_solve!(φ_to_u::AbstractFEStateMap,du::AbstractVector)
  @abstractmethod
end

"""
    dRdφ(φ_to_u::AbstractFEStateMap,uh,vh,φh)

Compute the derivative with respect to `φh` of the residual R.
"""
function dRdφ(φ_to_u::AbstractFEStateMap,uh,vh,φh)
  @abstractmethod
end

function dRdφ(φ_to_u::AbstractFEStateMap,u::AbstractVector,v::AbstractVector,φ::AbstractVector)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  vh = FEFunction(get_test_space(φ_to_u),v)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return dRdφ(φ_to_u,uh,vh,φh)
end

"""
    pullback(φ_to_u::AbstractFEStateMap,uh,φh,du;updated)

Compute `∂F∂u*dudφ` at `φh` and `uh` using the adjoint method. I.e., let

`∂F∂u*dudφ = -λᵀ*dRdφ`

and solve the adjoint problem `dRduᵀ*λ = ∂F∂uᵀ` using [`adjoint_solve!`](@ref).
"""
function pullback(φ_to_u::AbstractFEStateMap,uh,φh,du;updated=false)
  dudφ_vec, assem_deriv = φ_to_u.plb_caches
  U_reg = get_deriv_space(φ_to_u)

  ## Adjoint Solve
  if !updated
    update_adjoint_caches!(φ_to_u,uh,φh)
  end
  λ  = adjoint_solve!(φ_to_u,du)
  λh = FEFunction(get_test_space(φ_to_u),λ)

  ## Compute grad
  dudφ_vecdata = collect_cell_vector(U_reg,dRdφ(φ_to_u,uh,λh,φh))
  assemble_vector!(dudφ_vec,assem_deriv,dudφ_vecdata)
  rmul!(dudφ_vec, -1)

  return (NoTangent(),dudφ_vec)
end

function pullback(φ_to_u::AbstractFEStateMap,u::AbstractVector,φ::AbstractVector,du::AbstractVector;updated=false)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return pullback(φ_to_u,uh,φh,du;updated=updated)
end

"""
    rrule(φ_to_u::AbstractFEStateMap,φh)

Return the evaluation of a `AbstractFEStateMap` and a
a function for evaluating the pullback of `φ_to_u`. This enables
compatiblity with `ChainRules.jl`
"""
function ChainRulesCore.rrule(φ_to_u::AbstractFEStateMap,φh)
  u  = forward_solve!(φ_to_u,φh)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  update_adjoint_caches!(φ_to_u,uh,φh)
  return u, du -> pullback(φ_to_u,uh,φh,du;updated=true)
end

function ChainRulesCore.rrule(φ_to_u::AbstractFEStateMap,φ::AbstractVector)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return ChainRulesCore.rrule(φ_to_u,φh)
end

# IO
function Base.show(io::IO,object::AbstractFEStateMap)
  print(io,"$(nameof(typeof(object)))")
end