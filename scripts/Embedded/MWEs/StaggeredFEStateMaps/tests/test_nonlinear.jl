using Gridap, GridapTopOpt, FiniteDifferences

model = CartesianDiscreteModel((0,1,0,1),(8,8))
order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
Ω = Triangulation(model)

V_φ = TestFESpace(Ω,reffe)
φf(x) = x[1]*x[2]+1
φh = interpolate(φf,V_φ)
V_reg = TestFESpace(Ω,reffe)
U_reg = TrialFESpace(V_reg)

V = FESpace(Ω,reffe;dirichlet_tags="boundary")

_rhs(x) = x[1] - x[2]
_sol(x) = _rhs(x)*φf(x)
U = TrialFESpace(V,_sol)

# Define weakforms
dΩ = Measure(Ω,3*order)

L(u::Function) = x -> (u(x) + 1) * u(x) * (u(x) + 2)
L(u) = (u + 1) * u * (u + 2)

r(u1,v1,φ) = ∫((φ*L(u1) - L(sol[1])) * v1)dΩ

# Create operator from components
φ_to_u = NonlinearFEStateMap(r,U,V,V_φ,U_reg,φh)
forward_solve!(φ_to_u,φh)

F(u1,φ) = ∫(u1*φ)dΩ
_F = GridapTopOpt.StateParamMap(F,φ_to_u)

# With statemaps
u, u_pullback = GridapTopOpt.rrule(φ_to_u,φh)
uh = FEFunction(U,u)
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

