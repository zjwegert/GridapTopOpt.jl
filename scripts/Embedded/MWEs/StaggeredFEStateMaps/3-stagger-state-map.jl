include("core_old_caching.jl")
include("extensions.jl")

model = CartesianDiscreteModel((0,1,0,1),(8,8))
order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
Ω = Triangulation(model)

V_φ = TestFESpace(Ω,reffe)
φh = interpolate(x->x[1]*x[2]+1,V_φ)
V_reg = TestFESpace(Ω,reffe)
U_reg = TrialFESpace(V_reg)

V = FESpace(Ω,reffe;dirichlet_tags="boundary")

sol = [x -> x[1], x -> x[2], x -> x[1] + x[2], x -> x[1] - x[2]]
U1 = TrialFESpace(V,sol[1])
U2 = TrialFESpace(V,sol[2])
U3 = TrialFESpace(V,sol[3])
U4 = TrialFESpace(V,sol[4])

# Define weakforms
dΩ = Measure(Ω,3*order)

a1((),u1,v1,φ) = ∫(φ * u1 * v1)dΩ
l1((),v1,φ) = ∫(sol[1] * v1)dΩ

a2((u1,),(u2,u3),(v2,v3),φ) = ∫(u1 * u2 * v2)dΩ + ∫(u3 * v3)dΩ
l2((u1,),(v2,v3),φ) = ∫(φ * sol[2] * u1 * v2)dΩ + ∫(sol[3] * v3)dΩ

a3((u1,(u2,u3)),u4,v4,φ) = ∫(φ * φ * (u1 + u2) * u4 * v4)dΩ
l3((u1,(u2,u3)),v4,φ) = ∫(φ * sol[4] * (u1 + u2) * v4)dΩ

# Create operator from components
UB1, VB1 = U1, V
UB2, VB2 = MultiFieldFESpace([U2,U3]), MultiFieldFESpace([V,V])
UB3, VB3 = U4, V
op = StaggeredAffineFEOperator([a1,a2,a3],[l1,l2,l3],[UB1,UB2,UB3],[VB1,VB2,VB3])

φ_to_u = StaggeredAffineFEStateMap(op,V_φ,U_reg,φh)

forward_solve!(φ_to_u,φh)
xh = get_state(φ_to_u)

F((u1,(u2,u3),u4),φ) = ∫(u1*u2*u3*u4*φ)dΩ
_F = StaggeredStateParamMap(F,φ_to_u)

# With statemaps
u, u_pullback = GridapTopOpt.rrule(φ_to_u,φh);
uh = FEFunction(get_trial_space(φ_to_u),u)
j_val, j_pullback = GridapTopOpt.rrule(_F,uh,φh)   # Compute functional and pull back
_, dFdu, dFdφ     = j_pullback(1)    # Compute dFdu, dFdφ
_, dφ_adj         = u_pullback(dFdu) # Compute -dFdu*dudφ via adjoint
_dF = dφ_adj + dFdφ

function φ_to_j(φ)
  u = φ_to_u(φ)
  _F(u,φ)
end

using FiniteDifferences
fdm_grad = FiniteDifferences.grad(central_fdm(5, 1), φ_to_j, get_free_dof_values(φh))[1]
norm(_dF - fdm_grad, Inf)/norm(fdm_grad,Inf)
