include("core_old_caching.jl")
include("extensions.jl")

model = CartesianDiscreteModel((0,1,0,1),(8,8))
order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
Ω = Triangulation(model)

V_φ = TestFESpace(Ω,reffe)
φf(x) = x[1]+1
φh = interpolate(φf,V_φ)
V_reg = TestFESpace(Ω,reffe)
U_reg = TrialFESpace(V_reg)

V = FESpace(Ω,reffe;dirichlet_tags="boundary")

rhs = [x -> x[1], x -> (x[1] - x[2])]
sol = [x -> rhs[1](x)*φf(x), x -> rhs[2](x)*φf(x)]
U1 = TrialFESpace(V,sol[1])
U2 = TrialFESpace(V,sol[2])

# Define weakforms
dΩ = Measure(Ω,2*order)

a1((),u1,v1,φ) = ∫(u1 * v1)dΩ
l1((),v1,φ) = ∫(φ * rhs[1] * v1)dΩ

a2((u1,),u2,v2,φ) = ∫(u1 * u2 * v2)dΩ
l2((u1,),v2,φ) = ∫(φ * rhs[2] * u1 * v2)dΩ

# Create operator from components
op = StaggeredAffineFEOperator([a1,a2],[l1,l2],[U1,U2],[V,V])

φ_to_u = StaggeredAffineFEStateMap(op,V_φ,U_reg,φh)

forward_solve!(φ_to_u,φh)
xh = get_state(φ_to_u)

F((u1,u2),φ) = ∫(u1 + u2 + φ)dΩ
_F = StaggeredStateParamMap(F,φ_to_u)

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