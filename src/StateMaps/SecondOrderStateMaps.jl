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

# helpers
function fwd_pass_ran(p_to_u::AbstractFEStateMap,p::AbstractVector)
  get_free_dof_values(get_parameter(p_to_u)) == p && p_to_u.cache.state_updated
end

function bwd_pass_ran(p_to_u::AbstractFEStateMap,p::AbstractVector)
  get_free_dof_values(get_parameter(p_to_u)) == p && p_to_u.cache.adjoint_updated
end

function incremental_state_map(p_to_u::AbstractFEStateMap, res,  pᵋ::AbstractVector{ForwardDiff.Dual{T,VT,PT}}) where {T,VT,PT}
  u̇, assem_∂R∂p, ∂R∂p_mat = p_to_u.cache.inc_state_cache
  ns = get_ns(p_to_u) # numerical factorisation for the incremental state system is the same as the state system in the forward pass
  
  p = ForwardDiff.value.(pᵋ) 
  ṗ =  mapreduce(ForwardDiff.partials, vcat, pᵋ)'
  u = get_free_dof_values(get_state(p_to_u)) # current solution


  println("Running HVP at p = $(sum(p)) and ṗ = $(sum(ṗ))")

  # solve state (if needed): once per outer iteration - should have been done already as the optimiser should first call the forward pass (to compute the gradient) before computing HVP's
  if !fwd_pass_ran(p_to_u,p)
    @warn "You are not calling the forward pass (state) before computing HVP's"
    u = p_to_u(p) # will also update the incremental state partial ∂R∂p
  end

  # solve incremental state: once per inner iteration (only thing changing is ṗ)
  solve!(u̇, ns, -∂R∂p_mat*ṗ') # incremental state equation

  return map(u, eachrow(u̇)) do v, p
    ForwardDiff.Dual{T}(v, p...)
  end
end

function (p_to_u::AffineFEStateMap)(pᵋ::AbstractVector{ForwardDiff.Dual{T,VT,PT}}) where {T,VT,PT}
  res = (u,v,p) -> p_to_u.biform(u,v,p) - p_to_u.liform(v,p)
  ns = p_to_u.cache.fwd_cache[1] 
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
  ṗ =  vec(mapreduce(ForwardDiff.partials, hcat, pᵋ))
  u = ForwardDiff.value.(uᵋ)
  u̇ = vec(mapreduce(ForwardDiff.partials, hcat, uᵋ))
  du = ForwardDiff.value.(duᵋ)
  du̇ = vec(mapreduce(ForwardDiff.partials, hcat, duᵋ))  

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

  dpᵋ = map(dp_from_u, eachrow(dṗ_from_u)) do v, p
    ForwardDiff.Dual{T}(v, p...)
  end
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

######################################################################
# u̇ -> du̇, dṗ: Computing the increments of the objective functional #
######################################################################

function fwd_pass_ran(u_to_j::StateParamMap,u,p)
  u_to_j.caches[5] == u && u_to_j.caches[6] == p && u_to_j.cache2.fwd_ran 
end

function bwd_pass_ran(u_to_j::StateParamMap,u,p)
  u_to_j.caches[5] == u && u_to_j.caches[6] == p && u_to_j.cache2.bwd_ran
end

function (u_to_j::StateParamMap)(uᵋ::Vector{ForwardDiff.Dual{T1,V1,P1}},pᵋ::Vector{ForwardDiff.Dual{T2,V2,P2}}) where {T1,V1,P1,T2,V2,P2}
  F = u_to_j.F
  U,V_p = u_to_j.spaces
  ∂j∂u_vec,∂j∂φ_vec,_,_,_,_,j = u_to_j.caches

  u = ForwardDiff.value.(uᵋ)
  u̇ = ForwardDiff.partials.(uᵋ)
  p = ForwardDiff.value.(pᵋ)
  ṗ = ForwardDiff.partials.(pᵋ)
  
  # pushforward the value # skip if already computed at the point p 
  if !fwd_pass_ran(u_to_j,u,p)
    @warn "You are not calling the forward pass (objective) before computing HVP's"
    j = u_to_j(u,p) # will also update ∂j∂u_vec and ∂j∂φ_vec
  end 

  # pushforward the dual component
  J̇ = ∂j∂φ_vec ⋅ ṗ + ∂j∂u_vec ⋅ u̇
  Jᵋ = ForwardDiff.Dual{T2}(j[], J̇)
  return  Jᵋ
end

function ChainRulesCore.rrule(u_to_j::StateParamMap,uᵋ::Vector{ForwardDiff.Dual{T1,V1,P1}},pᵋ::Vector{ForwardDiff.Dual{T2,V2,P2}}) where {T1,V1,P1,T2,V2,P2}
  spaces = u_to_j.spaces
  U,V_p = spaces
  F = u_to_j.F
  ∂j∂u_vec,∂j∂φ_vec,_,_,_,_,j = u_to_j.caches

  u = ForwardDiff.value.(uᵋ)
  p = ForwardDiff.value.(pᵋ)
  u̇ = mapreduce(ForwardDiff.partials, hcat, uᵋ)'
  ṗ = mapreduce(ForwardDiff.partials, hcat, pᵋ)'

  function u_to_j_pullback(dJᵋ)
    # pullback the value # skip if already computed at the point p
    dJ = ForwardDiff.value(dJᵋ)
    if !bwd_pass_ran(u_to_j,u,p)
      @warn "You are not calling the backwards pass (objective) before computing HVP's"
      _, ∂j∂u_vec, ∂j∂φ_vec = GridapTopOpt.pullback(u_to_j,u,p,dJ) 
    end

    # pullback the dual component

    # once per outer iteration
    #∂2J∂u2_mat, ∂2J∂u∂p_mat, ∂2J∂p2_mat, ∂2J∂p∂u_mat = incremental_objective_partials(F,uh,ph,spaces)
    _, ∂2J∂u2_mat, _, ∂2J∂u∂p_mat, _, ∂2J∂p2_mat,  _, ∂2J∂p∂u_mat = u_to_j.inc_obj_cache
   
    # once per inner iteration
    dṗ = ∂2J∂p2_mat * ṗ + ∂2J∂p∂u_mat * u̇ 
    du̇ = ∂2J∂u2_mat * u̇ + ∂2J∂u∂p_mat * ṗ 

    Du̇ = map(∂j∂u_vec, eachrow(du̇)) do v, p
      ForwardDiff.Dual{T1}(v, p...)
    end
    Dṗ = map(∂j∂φ_vec, eachrow(dṗ)) do v, p
      ForwardDiff.Dual{T2}(v, p...)
    end
    (  NoTangent(), Du̇, Dṗ )
  end

  return u_to_j(uᵋ,pᵋ), u_to_j_pullback
end