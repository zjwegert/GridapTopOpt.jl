using GridapTopOpt
using Gridap, Gridap.MultiField
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using FiniteDiff
using Test

verbose = false
model = CartesianDiscreteModel((0,1,0,1),(8,8))
order = 2
reffe = ReferenceFE(lagrangian,Float64,order)
Ω = Triangulation(model)

V_φ = TestFESpace(Ω,reffe)
φf(x) = x[1]*x[2]+1
φh = interpolate(φf,V_φ)
V_reg = TestFESpace(Ω,reffe)
U_reg = TrialFESpace(V_reg)

V = FESpace(Ω,reffe;dirichlet_tags="boundary")

rhs = [x -> x[1], x -> x[2], x -> x[1] + x[2], x -> x[1] - x[2]]
sol = [x -> rhs[1](x), x -> rhs[2](x)*φf(x), x -> rhs[3](x), x -> rhs[4](x)*φf(x)]
U1 = TrialFESpace(V,sol[1])
U2 = TrialFESpace(V,sol[2])
U3 = TrialFESpace(V,sol[3])
U4 = TrialFESpace(V,sol[4])

# Define weakforms
dΩ = Measure(Ω,2*order)

a1((),u1,v1,φ) = ∫(φ * u1 * v1)dΩ
l1((),v1,φ) = ∫(φ * rhs[1] * v1)dΩ

a2((u1,),(u2,u3),(v2,v3),φ) = ∫(u1 * u2 * v2)dΩ + ∫(u3 * v3)dΩ
l2((u1,),(v2,v3),φ) = ∫(φ * rhs[2] * u1 * v2)dΩ + ∫(rhs[3] * v3)dΩ

a3((u1,(u2,u3)),u4,v4,φ) = ∫(φ * (u1 + u2) * u4 * v4)dΩ
l3((u1,(u2,u3)),v4,φ) = ∫(φ *φ * rhs[4] * (u1 + u2) * v4)dΩ

# Create operator from components
UB1, VB1 = U1, V
UB2, VB2 = MultiFieldFESpace([U2,U3]), MultiFieldFESpace([V,V])
UB3, VB3 = U4, V
op = StaggeredAffineFEOperator([a1,a2,a3],[l1,l2,l3],[UB1,UB2,UB3],[VB1,VB2,VB3])

φ_to_u = StaggeredAffineFEStateMap(op,V_φ,U_reg,φh)

# Test solution
GridapTopOpt.forward_solve!(φ_to_u,φh)
xh = get_state(φ_to_u)

xh_exact = interpolate(sol,op.trial)
for k in 1:length(sol)
  eh_k = xh[k] - xh_exact[k]
  e_k = sqrt(sum(∫(eh_k * eh_k)dΩ))
  verbose && println("Error in field $k: $e_k")
  @test e_k < 1e-10
end

# Test gradient
F((u1,(u2,u3),u4),φ) = ∫(u1*u2*u3*u4*φ)dΩ
pcf = PDEConstrainedFunctionals(F,φ_to_u)
_,_,_dF,_ = evaluate!(pcf,φh)

function φ_to_j(φ)
  u = φ_to_u(φ)
  pcf.J(u,φ)
end

fdm_grad = FiniteDiff.finite_difference_gradient(φ_to_j, get_free_dof_values(φh))
rel_error = norm(_dF - fdm_grad, Inf)/norm(fdm_grad,Inf)

verbose && println("Relative error in gradient: $rel_error")
!isnan(rel_error) && @test rel_error < 1e-10