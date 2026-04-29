###########################################################################
# ṗ->u̇ : Solving the "incremental state equation" ∂R/∂u * u̇ = - ∂R/∂p * ṗ #
###########################################################################

# getters
get_ns(m::AffineFEStateMap) = m.cache.fwd_cache[1]
get_ns(nls_cache::NewtonRaphsonCache) = nls_cache.ns
get_ns(nls_cache::NLSolversCache) = nls_cache.ns
get_ns(nls_cache::NewtonCache) = nls_cache.ns

function get_ns(m::NonlinearFEStateMap)
  nls_cache = m.cache.fwd_cache[2]
  ns = get_ns(nls_cache)
end

# helpers to check if the inc caches have been updated for the current point p
function fwd_pass_ran(p_to_u::AbstractFEStateMap,p::AbstractVector)
  is_cache_built(p_to_u.cache) ? nothing : return false # return false if cache is not built (it will get built in the forward pass)
  get_free_dof_values(get_parameter(p_to_u)) == p && p_to_u.cache.state_updated
end

function bwd_pass_ran(p_to_u::AbstractFEStateMap,p::AbstractVector)
  is_cache_built(p_to_u.cache) ? nothing : return false # return false if cache is not built (it will get built in the backward pass)
  get_free_dof_values(get_parameter(p_to_u)) == p && p_to_u.cache.adjoint_updated
end

function _mapreduce_partials(pᵋ,∂R∂p_mat)
  mapreduce(ForwardDiff.partials, vcat, pᵋ)
end

function _mapreduce_partials(pᵋ::PVector,∂R∂p_mat)
  pvec = allocate_in_domain(∂R∂p_mat)
  v = map(Base.Fix2(_mapreduce_partials,nothing), local_views(pᵋ))
  pvec_ids = pvec.index_partition
  pᵋ_ids = pᵋ.index_partition
  map(_map_dofs_to_rhs!,local_views(pvec),local_views(pvec_ids),v,local_views(pᵋ_ids))
  consistent!(pvec) |> wait
  return pvec
end

function _build_duals(tag, u, u̇)
  map(u, eachrow(u̇)) do v, p
    ForwardDiff.Dual{tag}(v, p...)
  end
end

function _build_duals(tag, u::PVector, u̇::PVector)
  ids = u̇.index_partition
  v = map((x,y)->_build_duals(tag,x,y), local_views(u), local_views(u̇))
  u_dual = PVector(v,ids)
  consistent!(u_dual) |> wait
  return u_dual
end

function incremental_state_map(p_to_u::AbstractFEStateMap{N}, res,  pᵋ::AbstractVector{ForwardDiff.Dual{T,VT,PT}}) where {N,T,VT,PT}
  @assert N == 2 "You're trying to compute the Hessian-vector product for a state map that only expects first order derivatives.
    You should set diff_order = 2 in the FEStateMap and StateParamMap constructors to enable second order differentiation."
  u̇, assem_∂R∂p, ∂R∂p_mat = p_to_u.cache.inc_state_cache
  p = ForwardDiff.value.(pᵋ)
  ṗ = _mapreduce_partials(pᵋ,∂R∂p_mat)

  # solve state (if needed): once per outer iteration - should have been done already as the optimiser should first call the forward pass (to compute the gradient) before computing HVP's
  if !fwd_pass_ran(p_to_u,p)
    @warn "You are not calling the forward pass (state) before computing HVP's"
    u = p_to_u(p) # will also update the incremental state partial ∂R∂p
  end

  u = get_free_dof_values(get_state(p_to_u)) # current solution
  ns = get_ns(p_to_u) # numerical factorisation for the incremental state system is the same as the state system in the forward pass

  # solve incremental state: once per inner iteration (only thing changing is ṗ)
  solve!(u̇, ns, -∂R∂p_mat*ṗ) # incremental state equation
  return _build_duals(T, u, u̇)
end

function (p_to_u::AffineFEStateMap)(pᵋ::AbstractVector{ForwardDiff.Dual{T,VT,PT}}) where {T,VT,PT}
  res = (u,v,p) -> p_to_u.biform(u,v,p) - p_to_u.liform(v,p)
  incremental_state_map(p_to_u, res, pᵋ)
end

function (p_to_u::NonlinearFEStateMap)(pᵋ::AbstractVector{ForwardDiff.Dual{T,VT,PT}}) where {T,VT,PT}
  res = p_to_u.res
  incremental_state_map(p_to_u, res, pᵋ)
end

function incremental_adjoint_pullback(p_to_u,res,uᵋ,pᵋ::AbstractVector{ForwardDiff.Dual{T,VT,PT}},duᵋ) where {T,VT,PT}
  U,V,V_p = p_to_u.spaces
  adjoint_ns, _, λ = p_to_u.cache.adj_cache
  dp_from_u, assem_deriv = p_to_u.cache.plb_cache
  λ⁻, dṗ_from_u,   assem_∂2R∂u2, ∂2R∂u2_mat,   assem_∂2R∂u∂p,∂2R∂u∂p_mat,  assem_∂2R∂p2,∂2R∂p2_mat,  assem_∂2R∂p∂u,∂2R∂p∂u_mat = p_to_u.cache.inc_adjoint_cache

  p = ForwardDiff.value.(pᵋ)
  ṗ = _mapreduce_partials(pᵋ,∂2R∂u∂p_mat)
  u = ForwardDiff.value.(uᵋ)
  u̇ = _mapreduce_partials(uᵋ,∂2R∂u2_mat)
  du = ForwardDiff.value.(duᵋ)
  du̇ = tangent_from_dual(duᵋ)

  ## pullback the value  (solve the adjoint equation) - once per outer iteration
  if !bwd_pass_ran(p_to_u,p)
    @warn "You are not calling the backwards pass (state) before computing HVP's"
    _, dp_from_u = GridapTopOpt.pullback(p_to_u,u,p,du) # This will update λ, dp_from_u and the incremental adjoint partials - it would be better if these objects were returned so that we know they were updated
  end

  ## pullback the dual component (solve the incremental adjoint equation) - once per inner iteration
  #du̇ .= du̇ - (∂2R∂u2_mat*u̇ + ∂2R∂u∂p_mat*ṗ)
  mul!(du̇, ∂2R∂u2_mat, u̇, -1, 1)  # du̇ := du̇ - ∂2R∂u2_mat*u̇
  mul!(du̇, ∂2R∂u∂p_mat, ṗ, -1, 1) # du̇ := du̇ - ∂2R∂u∂p_mat*ṗ

  λ⁻ = solve!(λ⁻,adjoint_ns,du̇) # solve the incremental adjoint equation
  uh = FEFunction(U,u)
  λ⁻h = FEFunction(V,λ⁻)
  ph = FEFunction(V_p,p)
  ∂R∂p_λ⁻_vecdata = collect_cell_vector(V_p,GridapTopOpt.dRdφ(p_to_u,uh,λ⁻h,ph))
  assemble_vector!(dṗ_from_u,assem_deriv,∂R∂p_λ⁻_vecdata)

  #dṗ_from_u .= - dṗ_from_u - (∂2R∂p2_mat*ṗ + ∂2R∂p∂u_mat*u̇)
  rmul!(dṗ_from_u, -1)                    # dṗ_from_u := -dṗ_from_u
  mul!(dṗ_from_u, ∂2R∂p2_mat, ṗ, -1, 1)   # dṗ_from_u -= ∂2R∂p2_mat*ṗ
  mul!(dṗ_from_u, ∂2R∂p∂u_mat, u̇, -1, 1)  # dṗ_from_u -= ∂2R∂p∂u_mat*u̇

  dpᵋ = _build_duals(T, dp_from_u, dṗ_from_u)
  ( NoTangent(), dpᵋ)
end

function ChainRulesCore.rrule(p_to_u::NonlinearFEStateMap,pᵋ::AbstractVector{ForwardDiff.Dual{T,VT,PT}}) where {T,VT,PT}
  res = p_to_u.res
  uᵋ = p_to_u(pᵋ)
  return uᵋ, duᵋ -> incremental_adjoint_pullback(p_to_u,res,uᵋ,pᵋ,duᵋ)
end

function ChainRulesCore.rrule(p_to_u::AffineFEStateMap,pᵋ::AbstractVector{ForwardDiff.Dual{T,VT,PT}}) where {T,VT,PT}
  res = (u,v,p) -> p_to_u.biform(u,v,p) - p_to_u.liform(v,p)
  uᵋ = p_to_u(pᵋ)
  return uᵋ, duᵋ -> incremental_adjoint_pullback(p_to_u,res,uᵋ,pᵋ,duᵋ)
end

#####################################################################
# u̇ -> du̇, dṗ: Computing the increments of the objective functional #
#####################################################################

function fwd_pass_ran(u_to_j::StateParamMap,u,p)
  is_cache_built(u_to_j.cache) ? nothing : return false
  u_to_j.cache.fwd_cache[1] == u && u_to_j.cache.fwd_cache[2] == p && u_to_j.cache.fwd_ran
end

function bwd_pass_ran(u_to_j::StateParamMap,u,p)
  is_cache_built(u_to_j.cache) ? nothing : return false
  u_to_j.cache.fwd_cache[1] == u && u_to_j.cache.fwd_cache[2] == p && u_to_j.cache.bwd_ran
end

function (u_to_j::StateParamMap)(uᵋ::AbstractVector{ForwardDiff.Dual{T1,V1,P1}},pᵋ::AbstractVector{ForwardDiff.Dual{T2,V2,P2}}) where {T1,V1,P1,T2,V2,P2}
  F = u_to_j.F
  U,V_p = u_to_j.spaces

  u = ForwardDiff.value.(uᵋ)
  u̇ = ForwardDiff.partials.(uᵋ)
  p = ForwardDiff.value.(pᵋ)
  ṗ = ForwardDiff.partials.(pᵋ)

  # pushforward the value # skip if already computed at the point p
  if !fwd_pass_ran(u_to_j,u,p)
    @warn "You are not calling the forward pass (objective) before computing HVP's"
    j = u_to_j(u,p) # will also update ∂j∂u_vec and ∂j∂φ_vec
  end

  ∂j∂u_vec,∂j∂φ_vec,_,_ = u_to_j.cache.plb_cache
  u,p,j = u_to_j.cache.fwd_cache

  # pushforward the dual component
  J̇ = ∂j∂φ_vec ⋅ ṗ + ∂j∂u_vec ⋅ u̇
  Jᵋ = ForwardDiff.Dual{T2}(j[], J̇)
  return  Jᵋ
end

function ChainRulesCore.rrule(u_to_j::StateParamMap,uᵋ::AbstractVector{ForwardDiff.Dual{T1,V1,P1}},pᵋ::AbstractVector{ForwardDiff.Dual{T2,V2,P2}}) where {T1,V1,P1,T2,V2,P2}
  spaces = u_to_j.spaces
  U,V_p = spaces
  F = u_to_j.F
  ∂j∂u_vec,∂j∂φ_vec,_,_ = u_to_j.cache.plb_cache

  u = ForwardDiff.value.(uᵋ)
  p = ForwardDiff.value.(pᵋ)

  function u_to_j_pullback(dJᵋ)
    # pullback the value # skip if already computed at the point p
    dJ = ForwardDiff.value(dJᵋ)
    dJ̇ = ForwardDiff.partials(dJᵋ)

    if !bwd_pass_ran(u_to_j,u,p)
      @warn "You are not calling the backwards pass (objective) before computing HVP's"
      _, ∂j∂u_vec, ∂j∂φ_vec = GridapTopOpt.pullback(u_to_j,u,p,dJ)
    end

    # pullback the dual component

    # once per outer iteration
    dṗ_from_j, du̇_from_j, _, ∂2J∂u2_mat, _, ∂2J∂u∂p_mat, _, ∂2J∂p2_mat,  _, ∂2J∂p∂u_mat = u_to_j.cache.inc_obj_cache
    ṗ = _mapreduce_partials(pᵋ,∂2J∂p2_mat)
    u̇ = _mapreduce_partials(uᵋ,∂2J∂u2_mat)

    # once per inner iteration
    # dṗ_from_j .=  (∂2J∂p2_mat*ṗ + ∂2J∂p∂u_mat*u̇)

    mul!(dṗ_from_j, ∂2J∂p2_mat, ṗ, 1, 0)   # dṗ_from_j := ∂2J∂p2_mat*ṗ
    mul!(dṗ_from_j, ∂2J∂p∂u_mat, u̇, 1, 1)   # dṗ_from_j += ∂2J∂p∂u_mat*u̇

    mul!(du̇_from_j, ∂2J∂u2_mat, u̇, 1, 0)   # du̇_from_j := ∂2J∂u2_mat*u̇
    mul!(du̇_from_j, ∂2J∂u∂p_mat, ṗ, 1, 1)   # du̇_from_j += ∂2J∂u∂p_mat*ṗ

    Du̇ = _build_duals(T1, ∂j∂u_vec, du̇_from_j)
    Dṗ = _build_duals(T2, ∂j∂φ_vec, dṗ_from_j)
    (  NoTangent(), Du̇, Dṗ )
  end

  return u_to_j(uᵋ,pᵋ), u_to_j_pullback
end

tangent_from_dual(pᵋ) = first.(ForwardDiff.partials.(pᵋ))