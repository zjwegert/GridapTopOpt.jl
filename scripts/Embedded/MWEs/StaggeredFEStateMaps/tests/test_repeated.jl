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
U = TrialFESpace(V,x->x[1] - x[2])

rhs = [x -> x[1], x -> x[2]]

# Define weakforms
dΩ = Measure(Ω,3*order)

a1(u1,v1,φ) = ∫(φ * u1 * v1)dΩ
l1(v1,φ) = ∫(sin∘(φ) * rhs[1] * v1)dΩ
l2(v1,φ) = ∫(cos∘(φ) * rhs[2] * v1)dΩ

# Create operator from components
φ_to_u = RepeatingAffineFEStateMap(2,a1,[l1,l2],U,V,V_φ,U_reg,φh)

F(x,φ) = ∫(x[1]*x[2]*φ)dΩ
_F = GridapTopOpt.StateParamMap(F,φ_to_u)

# With statemaps
u, u_pullback = GridapTopOpt.rrule(φ_to_u,φh)
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