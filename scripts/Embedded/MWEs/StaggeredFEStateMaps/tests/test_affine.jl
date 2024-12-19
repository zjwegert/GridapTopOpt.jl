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

a1(u1,v1,φ) = ∫(φ * u1 * v1)dΩ
l1(v1,φ) = ∫(φ* φ * _rhs * v1)dΩ

# Create operator from components
φ_to_u = AffineFEStateMap(a1,l1,U,V,V_φ,U_reg,φh)

forward_solve!(φ_to_u,φh)
xh = get_state(φ_to_u)

xh_exact = interpolate(_sol,U)
eh = xh - xh_exact
e = sqrt(sum(∫(eh * eh)dΩ))

F(u1,φ) = ∫(u1*φ)dΩ
_F = GridapTopOpt.StateParamMap(F,φ_to_u)

## Adjoint
∂R∂u(u1,du1,dv1) = a1(dv1,du1,φh)
∂R∂u_mat = assemble_matrix((du,dv)->∂R∂u(xh,du,dv),V,U)

∂F∂u(du1,u1,φ) = ∫(du1*φ)dΩ
∂F∂u_vec = assemble_vector(du->∂F∂u(du,xh,φh),U)

λ = ∂R∂u_mat \ ∂F∂u_vec

_dRdφ(dφ,u1,v1,φ) = ∫(dφ * u1 * v1)dΩ - ∫(2dφ * φ * _rhs * v1)dΩ
dRdφ_vec = assemble_vector(dφ->_dRdφ(dφ,xh,FEFunction(V,λ),φh),V_reg)

∂dF∂φ(dφ,u1,φ) = ∫(dφ*u1)dΩ
∂dF∂φ_vec = assemble_vector(dφ->∂dF∂φ(dφ,xh,φh),V_reg)

_dFdφ = ∂dF∂φ_vec - dRdφ_vec

# With statemaps
u, u_pullback = GridapTopOpt.rrule(φ_to_u,φh)
uh = FEFunction(U,u)
j_val, j_pullback = GridapTopOpt.rrule(_F,uh,φh)   # Compute functional and pull back
_, dFdu, dFdφ     = j_pullback(1)    # Compute dFdu, dFdφ
_, dφ_adj         = u_pullback(dFdu) # Compute -dFdu*dudφ via adjoint
_dF = dφ_adj + dFdφ

norm(_dFdφ - _dF,Inf)

function φ_to_j(φ)
  u = φ_to_u(φ)
  _F(u,φ)
end

fdm_grad = FiniteDifferences.grad(central_fdm(5, 1), φ_to_j, get_free_dof_values(φh))[1]
norm(_dF - fdm_grad, Inf)/norm(fdm_grad,Inf)