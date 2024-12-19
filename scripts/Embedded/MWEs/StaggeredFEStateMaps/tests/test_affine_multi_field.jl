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

sol = [x->x[1],x->x[1]-x[2]]
U1, U2 = TrialFESpace(V,sol[1]), TrialFESpace(V,sol[2])
UB = MultiFieldFESpace([U1,U2])
VB = MultiFieldFESpace([V,V])

# Define weakforms
dΩ = Measure(Ω,3*order)

a1((u1,u2),(v1,v2),φ) = ∫(φ * u1 * v1 + u2 * v2)dΩ
l1((v1,v2),φ) = ∫(φ* φ * _rhs * v1 + v2)dΩ

# Create operator from components
φ_to_u = AffineFEStateMap(a1,l1,UB,VB,V_φ,U_reg,φh)

F((u1,u2),φ) = ∫(u1*u2*φ)dΩ
_F = GridapTopOpt.StateParamMap(F,φ_to_u)

# With statemaps
u, u_pullback = GridapTopOpt.rrule(φ_to_u,φh)
uh = FEFunction(UB,u)
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