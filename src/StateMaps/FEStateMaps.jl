"""
    abstract type AbstractFEStateMap{N}

Types inheriting from this abstract type should enable the evaluation and differentiation of
the solution to an FE problem `u` that implicitly depends on an auxiliary parameter `φ`. `N`
specifies the order of differentiation supported by the state map.
"""
abstract type AbstractFEStateMap{N} end

"""
    get_diff_order(::AbstractFEStateMap{N}) where N
Return the order of differentiation supported by an FEStateMap.
"""
get_diff_order(::AbstractFEStateMap{N}) where N = N

"""
    get_state(m::AbstractFEStateMap)

Return the solution/state `u` to the FE problem.
"""
get_state(::AbstractFEStateMap) = @abstractmethod

get_state(m::Vector{<:AbstractFEStateMap}) = get_state.(m)

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
get_deriv_space(m::AbstractFEStateMap) = get_aux_space(m)

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
    update_incremental_state_partials!(φ_to_u,res,u,φ)

Update the incremental state partial `∂R/∂φ`
"""
function update_incremental_state_partials!(φ_to_u::AbstractFEStateMap{2},φh)
  U,V,V_p = φ_to_u.spaces
  u̇, assem_∂R∂φ, ∂R∂φ_mat = φ_to_u.cache.inc_state_cache
  res = get_res(φ_to_u)
  uh = get_state(φ_to_u)
  dv = get_fe_basis(V)
  ∂R∂φ = Gridap.jacobian(φ->res(uh,dv,φ),φh)
  assem_∂R∂φ = SparseMatrixAssembler(V_p,V)
  assemble_matrix!(∂R∂φ,∂R∂φ_mat,assem_∂R∂φ,V_p,V)
  return ∂R∂φ_mat
end

update_incremental_state_partials!(::AbstractFEStateMap{1},φh) = nothing

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
    update_incremental_adjoint_partials(res,uh,φh,λh,spaces)

Update the incremental adjoint partials used in the second order derivative computations.
"""
function update_incremental_adjoint_partials!(φ_to_u::AbstractFEStateMap{2},uh,φh,λh)
  U,V,V_p = φ_to_u.spaces
  res = get_res(φ_to_u)

  if !is_cache_built(φ_to_u.cache)
    build_cache!(φ_to_u,φh)
  end
  _, _,   assem_∂2R∂u2, ∂2R∂u2_mat,   assem_∂2R∂u∂φ,∂2R∂u∂φ_mat,  assem_∂2R∂φ2,∂2R∂φ2_mat,  assem_∂2R∂φ∂u,∂2R∂φ∂u_mat = φ_to_u.cache.inc_adjoint_cache

  # ∂²R / ∂u² * u̇ * λ
  ∂2R∂u2 = Gridap.hessian(uh->res(uh,λh,φh),uh)
  assemble_matrix!(∂2R∂u2,∂2R∂u2_mat,assem_∂2R∂u2,U,U)

  # ∂/∂φ (∂R/∂u * λ) * ṗ
  ∂R∂u_λ(uh,φh) = Gridap.gradient(uh->res(uh,λh,φh),uh)
  ∂2R∂u∂φ = Gridap.jacobian(φ->∂R∂u_λ(uh,φ),φh)
  assemble_matrix!(∂2R∂u∂φ,∂2R∂u∂φ_mat,assem_∂2R∂u∂φ,V_p,V)

  # ∂²R / ∂φ² * ṗ * λ
  ∂2R∂φ2 = Gridap.hessian(φh->res(uh,λh,φh),φh)
  assemble_matrix!(∂2R∂φ2,∂2R∂φ2_mat,assem_∂2R∂φ2,V_p,V_p)

  # ∂/∂u (∂R/∂φ * λ) * ṗ
  ∂R∂φ_λ(uh,φh) = Gridap.gradient(φh->res(uh,λh,φh),φh)
  ∂2R∂φ∂u = Gridap.jacobian(uh->∂R∂φ_λ(uh,φh),uh)
  assemble_matrix!(∂2R∂φ∂u,∂2R∂φ∂u_mat,assem_∂2R∂φ∂u,U,V_p)

  return ∂2R∂u2_mat, ∂2R∂u∂φ_mat, ∂2R∂φ2_mat, ∂2R∂φ∂u_mat
end

update_incremental_adjoint_partials!(::AbstractFEStateMap{1},uh,φh,λh) = nothing

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

function get_plb_cache(::AbstractFEStateMap)
  @abstractmethod
end

"""
    pullback(φ_to_u::AbstractFEStateMap,uh,φh,du;updated)

Compute `∂F∂u*dudφ` at `φh` and `uh` using the adjoint method. I.e., let

`∂F∂u*dudφ = -λᵀ*dRdφ`

and solve the adjoint problem `dRduᵀ*λ = ∂F∂uᵀ` using [`adjoint_solve!`](@ref).
"""
function pullback(φ_to_u::AbstractFEStateMap,uh,φh,du;updated=false)
  dudφ_vec, assem_deriv = get_plb_cache(φ_to_u)
  V_φ = get_deriv_space(φ_to_u)

  ## Adjoint Solve
  if !updated
    update_adjoint_caches!(φ_to_u,uh,φh)
  end

  λ  = adjoint_solve!(φ_to_u,du)
  λh = FEFunction(get_test_space(φ_to_u),conj!(λ))

  update_incremental_adjoint_partials!(φ_to_u,uh,φh,λh)

  ## Compute grad
  dudφ_vecdata = collect_cell_vector(V_φ,dRdφ(φ_to_u,uh,λh,φh))
  assemble_vector!(dudφ_vec,assem_deriv,dudφ_vecdata)
  rmul!(dudφ_vec, -1)

  return (NoTangent(),copy(dudφ_vec)) # avoid aliasing issues
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

## Caching
mutable struct FEStateMapCache
  cache_built::Bool
  solvers::Tuple
  fwd_cache::Tuple
  adj_cache::Tuple
  plb_cache::Tuple
  inc_state_cache::Tuple
  inc_adjoint_cache::Tuple
  state_updated:: Bool
  adjoint_updated:: Bool
end

function FEStateMapCache(fwd_solver,adjoint_solver)
  FEStateMapCache(false,(fwd_solver,adjoint_solver),(),(),(),(),(),false,false)
end

is_cache_built(c::FEStateMapCache) = c.cache_built

"""
    build_cache!(::AbstractFEStateMap,φh)

Build the FEStateMapCache (see AffineFEStateMap for an example)
"""
function build_inc_cache(state_map::AbstractFEStateMap{2},φh,uh,u̇,λ⁻)
  U,V,V_p = state_map.spaces
  res = get_res(state_map)

  # incremental state cache
  dv = get_fe_basis(V)
  ∂R∂φ = Gridap.jacobian(φ->res(uh,dv,φ),φh)
  assem_∂R∂φ = SparseMatrixAssembler(V_p,V)
  ∂R∂φ_mat = assemble_matrix(∂R∂φ,assem_∂R∂φ,V_p,V)
  inc_state_cache = (u̇, assem_∂R∂φ, ∂R∂φ_mat)

  # incremental adjoint cache
  λh = FEFunction(V,λ⁻)
  # ∂²R / ∂u² * u̇ * λ
  ∂2R∂u2 = Gridap.hessian(uh->res(uh,λh,φh),uh)
  assem_∂2R∂u2 = SparseMatrixAssembler(U,U)
  ∂2R∂u2_mat = assemble_matrix(∂2R∂u2,assem_∂2R∂u2,U,U)
  # ∂/∂φ (∂R/∂u * λ) * ṗ
  ∂R∂u_λ(uh,φh) = Gridap.gradient(uh->res(uh,λh,φh),uh)
  ∂2R∂u∂φ = Gridap.jacobian(φ->∂R∂u_λ(uh,φ),φh)
  assem_∂2R∂u∂φ = SparseMatrixAssembler(V_p,V)
  ∂2R∂u∂φ_mat = assemble_matrix(∂2R∂u∂φ,assem_∂2R∂u∂φ,V_p,V)
  # ∂²R / ∂φ² * ṗ * λ
  ∂2R∂φ2 = Gridap.hessian(φh->res(uh,λh,φh),φh)
  assem_∂2R∂φ2 = SparseMatrixAssembler(V_p,V_p)
  ∂2R∂φ2_mat = assemble_matrix(∂2R∂φ2,assem_∂2R∂φ2,V_p,V_p)
  # ∂/∂u (∂R/∂φ * λ) * ṗ
  ∂R∂φ_λ(uh,φh) = Gridap.gradient(φh->res(uh,λh,φh),φh)
  ∂2R∂φ∂u = Gridap.jacobian(uh->∂R∂φ_λ(uh,φh),uh)
  assem_∂2R∂φ∂u = SparseMatrixAssembler(U,V_p)
  ∂2R∂φ∂u_mat = assemble_matrix(∂2R∂φ∂u,assem_∂2R∂φ∂u,U,V_p)
  # incremental adjoint cotangent
  dṗ_from_u = allocate_in_domain(∂2R∂φ2_mat)
  inc_adjoint_cache = (λ⁻, dṗ_from_u,   assem_∂2R∂u2, ∂2R∂u2_mat,   assem_∂2R∂u∂φ,∂2R∂u∂φ_mat,  assem_∂2R∂φ2,∂2R∂φ2_mat,  assem_∂2R∂φ∂u,∂2R∂φ∂u_mat)

  return inc_state_cache, inc_adjoint_cache
end

build_inc_cache(state_map::AbstractFEStateMap{1},φh,uh,u̇,λ⁻) = ((),())

function build_cache!(::AbstractFEStateMap,φh)
  @abstractmethod
end

"""
    delete_cache!(c::FEStateMapCache)

Delete the contents of FEStateMapCache and mark for build.
"""
function delete_cache!(c::FEStateMapCache)
  c.cache_built = false
  c.fwd_cache = ()
  c.adj_cache = ()
  c.plb_cache = ()
  c.inc_state_cache = ()
  c.inc_adjoint_cache = ()
  return
end

# IO
function Base.show(io::IO,object::AbstractFEStateMap)
  print(io,"$(nameof(typeof(object)))")
end