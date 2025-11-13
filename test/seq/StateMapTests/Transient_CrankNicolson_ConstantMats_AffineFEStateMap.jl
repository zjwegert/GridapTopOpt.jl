module Transient_CrankNicolson_ConstantMats_AffineFEStateMap

using Gridap, Gridap.ODEs, Gridap.FESpaces, Gridap.CellData, Gridap.Arrays, Gridap.Helpers
using GridapTopOpt
using Test

model = CartesianDiscreteModel((-1,1,-1,1), (8,8))
Ω = Triangulation(model)
t0 = 0.0
tF = 0.1
N = 10
Δt = (tF-t0)/N
order = 1
dΩ = Measure(Ω, 2order)
reffe = ReferenceFE(lagrangian, Float64, order)

g(t) = x -> x[1] * (t + 0.1) * (x[2]^2 - 1)
f(t) = x -> sin(t) * sinpi(x[1]) * sinpi(x[2])
V = FESpace(model, reffe, dirichlet_tags="boundary")
U = TransientTrialFESpace(V, g)
uh0 = interpolate(g(t0), U(t0))
u0 = get_free_dof_values(uh0)
V_α = FESpace(model, reffe)
αf(x) = x[1]^2+x[2]^2+1
αh = interpolate(αf,V_α)

m(dtu, v, α) = ∫(α * v * dtu)dΩ
a(u, v, α) = ∫(α * (∇(v) ⋅ ∇(u)))dΩ
l(t, v, α) = ∫(v * f(t))dΩ

# Test
op = TransientLinearFEOperator((t,u,v)->a(u,v,αh), (t,dtu,v)->m(dtu,v,αh), (t, v)->l(t, v, αh), U, V; constant_forms=(true,true))
solver_rk = RungeKutta(LUSolver(), LUSolver(), Δt, :DIRK_CrankNicolson_2_2)
uh = solve(solver_rk, op, t0, tF, uh0)
_t = collect(t0:Δt:tF)
us_odes = [u0,[similar(u0) for t in _t[2:end]]...]
for (i,(tn, uhn)) in enumerate(uh)
  # @show i
  @assert tn ≈ _t[i+1] "tn = $tn != $(_t[i+1]) = t[i+1]"
  copy!(us_odes[i+1],get_free_dof_values(uhn))
end

# Crank-Nicolson
ts = collect(t0:Δt:tF);
A(tₙ,tₙ₋₁,uₙ,v,(α,uₙ₋₁)) = 1/Δt*m(uₙ, v, α) + 1/2*a(uₙ,v,α)
B(tₙ,tₙ₋₁,v,(α,uₙ₋₁)) = 1/2*l(tₙ,v,α) + 1/2*l(tₙ₋₁,v,α) + 1/Δt*m(uₙ₋₁, v, α) - 1/2*a(uₙ₋₁,v,α)

# AffineFEStateMaps
V_αuₙ₋₁(tₙ₋₁) = MultiFieldFESpace([V_α,U(tₙ₋₁)])
αuₙ₋₁_to_uₙ(tₙ,tₙ₋₁,isfirst) = AffineFEStateMap((u,v,p)->A(tₙ,tₙ₋₁,u,v,p),(v,p)->B(tₙ,tₙ₋₁,v,p),U(tₙ),V,V_αuₙ₋₁(tₙ₋₁),
  reassemble_matrix=isfirst,reassemble_adjoint=isfirst,precompute_uhd=true)

# Build first map
αu₁_to_u₂ = αuₙ₋₁_to_uₙ(ts[2],ts[1],true)
αu₁ = combine_fields(GridapTopOpt.get_deriv_space(αu₁_to_u₂),get_free_dof_values(αh),get_free_dof_values(uh0));
αu₁h = FEFunction(GridapTopOpt.get_deriv_space(αu₁_to_u₂),αu₁)
GridapTopOpt.build_cache!(αu₁_to_u₂,αu₁h);

# Build remaining maps and re-use cache
αuᵢ₋₁_to_uᵢ = [αu₁_to_u₂; [αuₙ₋₁_to_uₙ(ts[i],ts[i-1],false) for i in 3:length(ts)]...];

αu₁_to_u₂_adj_cache = αu₁_to_u₂.cache.adj_cache
αu₁_to_u₂_fwd_cache = αu₁_to_u₂.cache.fwd_cache
αu₁_to_u₂_plb_cache = αu₁_to_u₂.cache.plb_cache
for i in 2:length(αuᵢ₋₁_to_uᵢ)
  αuᵢ₋₁_to_uᵢ[i].cache.cache_built = true
  αuᵢ₋₁_to_uᵢ[i].cache.adj_cache = αu₁_to_u₂_adj_cache
  αuᵢ₋₁_to_uᵢ[i].cache.fwd_cache = (αu₁_to_u₂_fwd_cache[1:3]...,similar(αu₁_to_u₂_fwd_cache[4]),αu₁_to_u₂_fwd_cache[end]);
  αuᵢ₋₁_to_uᵢ[i].cache.plb_cache = αu₁_to_u₂_plb_cache
end;

## Check solution
function check_solution(α)
  @test u0 ≈ us_odes[1]
  αu0 = combine_fields(GridapTopOpt.get_deriv_space(αu₁_to_u₂),α,u0);
  # Comptue u₁ and store in uᵢ₋₁
  uᵢ₋₁ = αu₁_to_u₂(αu0);
  @test uᵢ₋₁ ≈ us_odes[2]
  for i in 2:length(αuᵢ₋₁_to_uᵢ)
    # Compute uᵢ given uᵢ₋₁
    αuᵢ₋₁ = combine_fields(V_αuₙ₋₁(ts[i-1]),α,uᵢ₋₁);
    uᵢ    = αuᵢ₋₁_to_uᵢ[i](αuᵢ₋₁);
    @test uᵢ ≈ us_odes[i+1]
    uᵢ₋₁ = uᵢ
  end
end
check_solution(get_free_dof_values(αh))

## Obj
J(u, α) = ∫(α * ∇(u) ⋅ ∇(u))dΩ
J_spm = GridapTopOpt.StateParamMap(J,V,V_α,SparseMatrixAssembler(V,V),SparseMatrixAssembler(V_α,V_α))

## Integrate in time
# function α_to_j_trapz(α)
#   αu0 = combine_fields(GridapTopOpt.get_aux_space(αuᵢ₋₁_to_uᵢ[1]),α,u0);
#   uᵢ = αuᵢ₋₁_to_uᵢ[1](αu0);
#   j = 0.0
#   for i in 1:length(ts)-2
#     # Solve for new time step
#     αuᵢ = combine_fields(GridapTopOpt.get_aux_space(αuᵢ₋₁_to_uᵢ[i+1]),α,uᵢ);
#     uᵢ₊₁  = αuᵢ₋₁_to_uᵢ[i+1](αuᵢ);
#     # Integrate in time
#     jᵢ = J_spm(uᵢ,α)
#     jᵢ₊₁ = J_spm(uᵢ₊₁,α)
#     j += (jᵢ₊₁ + jᵢ)*(ts[i+1]-ts[i])/2
#     uᵢ = uᵢ₊₁
#   end
#   return j
# end

## Sum in time
function α_to_j_sum(α)
  j = J_spm(u0,α)
  αu0 = combine_fields(GridapTopOpt.get_aux_space(αuᵢ₋₁_to_uᵢ[1]),α,u0);
  # Comptue u₁ and store in uᵢ₋₁
  uᵢ₋₁ = αuᵢ₋₁_to_uᵢ[1](αu0);
  for i in 2:length(αuᵢ₋₁_to_uᵢ)
    # Compute uᵢ given uᵢ₋₁
    αuᵢ₋₁ = combine_fields(GridapTopOpt.get_aux_space(αuᵢ₋₁_to_uᵢ[i]),α,uᵢ₋₁);
    uᵢ    = αuᵢ₋₁_to_uᵢ[i](αuᵢ₋₁);
    j    += J_spm(uᵢ,α)
    # Store uᵢ in uᵢ₋₁
    uᵢ₋₁ = uᵢ
  end
  return j/N
end

# α_to_j_sum(get_free_dof_values(αh))
val_and_grad = GridapTopOpt.val_and_gradient(α_to_j_sum,get_free_dof_values(αh))
ad_grad = val_and_grad.grad[1]

using FiniteDiff
fdm_grad = FiniteDiff.finite_difference_gradient(α_to_j_sum, get_free_dof_values(αh))

norm(ad_grad - fdm_grad,Inf)/norm(fdm_grad,Inf)
@test ad_grad ≈ fdm_grad

end