module SecondOrderTests

using Test, Gridap, GridapTopOpt
using FiniteDifferences
using Zygote
using ForwardDiff

# FE setup
order = 1 
xmax = ymax = 1.0
dom = (0,xmax,0,ymax)
el_size = (2,2)
model = CartesianDiscreteModel(dom,el_size)
Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe_scalar;dirichlet_tags=[1,2,3,4,5,6]) # 3 dofs
U = TrialFESpace(V,0.0)
V_p = TestFESpace(model,reffe_scalar;dirichlet_tags=[2,3,4,5,6,7,8]) # 2 dofs

#########################################
# Second order partial derivative tests #
#########################################

J(u,p) = ∫(u*u*p*p)dΩ # keep p term otherwise dual error
u = rand(num_free_dofs(U))
p = rand(num_free_dofs(V_p))
λ = rand(num_free_dofs(V))
uh = FEFunction(U,u)
ph = FEFunction(V_p,p)
λh = FEFunction(V,λ)
spaces = (U,V_p)
_,_,_, ∂2J∂u2_mat, _, ∂2J∂u∂p_mat, _, ∂2J∂p2_mat, _, ∂2J∂p∂u_mat = GridapTopOpt.build_inc_obj_cache(J,uh,ph,spaces,Val(2))
#∂2J∂u2_mat, ∂2J∂u∂p_mat, ∂2J∂p2_mat, ∂2J∂p∂u_mat = SecondOrderTopOpt.incremental_objective_partials(J,uh,ph,spaces)

# ∂²J / ∂u² * u̇
dv = get_fe_basis(V)
du = get_trial_fe_basis(U)
dp = get_fe_basis(V_p)
dp_ = get_trial_fe_basis(V_p)

∂2∂u2_analytical(uh) = ∫( 2*ph*ph*du⋅dv )dΩ
∂2∂u2_matrix_analytical = assemble_matrix(∂2∂u2_analytical(uh),U,U)
@test ∂2∂u2_matrix_analytical ≈ ∂2J∂u2_mat

# ∂/∂p (∂J/∂u ) * ṗ
∂2J∂u∂p_analytical(uh,ph) = ∫( 4*ph*uh*dp_⋅dv )dΩ
∂2J∂u∂p_matrix_analytical = assemble_matrix(∂2J∂u∂p_analytical(uh,ph),V_p,U)
@test ∂2J∂u∂p_matrix_analytical  ≈ ∂2J∂u∂p_mat

# ∂²J / ∂p² * ṗ
∂2J∂p2_analytical(uh) = ∫( 2*uh*uh*dp⋅dp_ )dΩ
∂2J∂p2_matrix_analytical = assemble_matrix(∂2J∂p2_analytical(uh),V_p,V_p)
@test ∂2J∂p2_matrix_analytical  ≈ ∂2J∂p2_mat

# ∂/∂u (∂J / ∂p) * u̇
∂2J∂p∂u_analytical(uh,ph) = ∫( 4*uh*ph*du⋅dp )dΩ
∂2J∂p∂u_matrix_analytical = assemble_matrix(∂2J∂p∂u_analytical(uh,ph),U,V_p)
@test ∂2J∂p∂u_matrix_analytical  ≈ ∂2J∂p∂u_mat

f(x) = 1.0
res(u,v,p) = ∫( p*∇(u)⋅∇(v) - f*v )dΩ
state_map = NonlinearFEStateMap(res,U,V,V_p,diff_order=2)
diff_order = GridapTopOpt.get_diff_order(state_map)
∂2R∂u2_mat, ∂2R∂u∂p_mat, ∂2R∂p2_mat, ∂2R∂p∂u_mat = GridapTopOpt.update_incremental_adjoint_partials!(state_map,uh,ph,λh,diff_order)

# ∂²R / ∂u² * u̇ * λ
∂2∂u2R_analytical(uh,λh,ph) = ∫( 0*du*dv )dΩ
∂2∂u2R_matrix_analytical = assemble_matrix(∂2∂u2R_analytical(uh,λh,ph),U,U)
@test ∂2∂u2R_matrix_analytical ≈ ∂2R∂u2_mat

# ∂/∂p (∂R/∂u * λ) * ṗ
∂2R∂u∂p_analytical(uh,λh,ph) = ∫( dp_* ∇(dv) ⋅ ∇(λh)  )dΩ
∂2R∂u∂p_matrix_analytical = assemble_matrix(∂2R∂u∂p_analytical(uh,λh,ph),V_p,U)
@test ∂2R∂u∂p_matrix_analytical ≈ ∂2R∂u∂p_mat
# ∂²R / ∂p² * ṗ * λ
∂2R∂p2_analytical(uh,λh) = ∫( 0*dp⋅dp_ )dΩ
∂2R∂p2_matrix_analytical = assemble_matrix(∂2R∂p2_analytical(uh,λh),V_p,V_p)
@test ∂2R∂p2_matrix_analytical ≈ ∂2R∂p2_mat   

# ∂/∂u (∂R/∂p * λ) * ṗ
∂2R∂p∂u_analytical(uh,λh,ph) = ∫( dp * ∇(du) ⋅ ∇(λh) )dΩ   
∂2R∂p∂u_matrix_analytical = assemble_matrix(∂2R∂p∂u_analytical(uh,λh,ph),U,V_p)
@test ∂2R∂p∂u_matrix_analytical ≈ ∂2R∂p∂u_mat

######################
# Self-adjoint tests #
######################

f(x) = 1.0
res(u,v,p) = ∫( p*∇(u)⋅∇(v)-f*v )dΩ   
J(u,p) = ∫( f*u + 0*p )dΩ # p term to avoid dual error - should be fixed in the future
state_map = NonlinearFEStateMap(res,U,V,V_p,diff_order=2)
objective = GridapTopOpt.StateParamMap(J,state_map,diff_order=2)
ṗ = [0.16337618888610783,1.54235]
p = [0.3253596201459815,2.45346264]
ph = FEFunction(V_p,p)
u = copy(state_map(p))

uh = FEFunction(U,u)
Zygote.gradient(p->objective(state_map(p),p),p) # update λ
λ = state_map.cache.adj_cache[3]

@test u ≈ λ # the adjoint should equal the solution for a self-adjoint problem

λh = FEFunction(V,λ)
T = ForwardDiff.Tag(()->(),typeof(p))
pᵋ = map(p, ṗ) do v, p
    ForwardDiff.Dual{T}(v, p...)
end
uᵋ = state_map(pᵋ)
ForwardDiff.value.(uᵋ) ≈ u
u̇ = vec(mapreduce(ForwardDiff.partials, hcat, uᵋ))

∇f = p->Zygote.gradient(p->objective(state_map(p),p),p)[1]
Hṗ_FOR = ForwardDiff.derivative(α -> ∇f(p + α*ṗ), 0)
λ⁻ = state_map.cache.inc_adjoint_cache[1]

@test λ⁻ ≈ u̇ # the incremental adjoint should equal the incremental state for a self-adjoint problem

function p_to_j(p)
    ph = FEFunction(V_p,p)
    op = FEOperator((u,v)->res(u,v,ph),U,V)
    uh = solve(op)
    sum(J(uh,ph))
end
g_fd = p->FiniteDifferences.jacobian(central_fdm(5,1),p_to_j,p)
Hṗ_fd = FiniteDifferences.jacobian(central_fdm(5,1),g_fd,p)[1]*ṗ

p_to_j(p) = objective((state_map(p)),p)
@test Hṗ_fd ≈ Hvp(p_to_j,p,ṗ) # the Hessian-vector product computed using the pullback should match the finite difference approximation of the Hessian-vector product (this is a test of the entire incremental map, including the adjoint part)

########################################################
# Unit and integration tests for the pushforward rules #
########################################################

J(u,p) = ∫( f*(1.0(sin∘(2π*u))+1)*(1.0(cos∘(2π*p))+1)*p)dΩ 
objective = GridapTopOpt.StateParamMap(J,state_map,diff_order=2)

# incremental objective (and pullback) test (u̇->du̇)
N = num_free_dofs(V)
function up_to_j(up)
    u = up[1:N]
    p = up[N+1:end]
    j = objective(u,p)
end
up = vcat(u,p)
u̇ṗ = vcat(u̇,ṗ)
∇f = up->Zygote.gradient(up_to_j,up)[1]
du̇dṗ =  ForwardDiff.derivative(α -> ∇f(up + α*u̇ṗ), 0)
du̇dṗ_FD =FiniteDifferences.jacobian(central_fdm(5,1),up->Zygote.gradient(up_to_j,up)[1],up)[1]*vcat(u̇,ṗ)

@test du̇dṗ_FD ≈ du̇dṗ # the pullback of the incremental objective should match the finite difference approximation of the pullback of the incremental objective

# Nonlinear state map tests 
res(u,v,p) = ∫( (u+1)*(p)*∇(u)⋅∇(v) - f*v )dΩ
state_map = NonlinearFEStateMap(res,U,V,V_p,diff_order=2)
Zygote.gradient(p->objective(state_map(p),p),p) # update λ and u

# incremental state test (ṗ->u̇)
function p_to_u(p)
    ph = FEFunction(V_p,p)
    op = FEOperator((u,v)->res(u,v,ph),U,V)
    uh = solve(op)
    return get_free_dof_values(uh)
end
uᵋ = state_map(pᵋ)
u̇ = vec(mapreduce(ForwardDiff.partials, hcat, uᵋ))
∂u_∂p_FD = FiniteDifferences.jacobian(central_fdm(5,1),p_to_u,p)[1]
∂u_∂p_FD_ṗ = ∂u_∂p_FD * ṗ
@test u̇ ≈ ∂u_∂p_FD_ṗ rtol = 1e-7 # the pullback of the incremental state should match the finite difference approximation of the pullback of the incremental state 

# entire incremental map (including the adjoint part) (ṗ->dṗ)
function p_to_j(p)
    ph = FEFunction(V_p,p)
    op = FEOperator((u,v)->res(u,v,ph),U,V)
    uh = solve(op)
    sum(J(uh,ph))
end
g_fd = p->FiniteDifferences.jacobian(central_fdm(5,1),p_to_j,p)
Hṗ_fd = FiniteDifferences.jacobian(central_fdm(5,1),g_fd,p)[1]*ṗ

p_to_j(p) = objective((state_map(p)),p)
@test Hṗ_fd ≈ Hvp(p_to_j,p,ṗ) # the Hessian-vector product computed with AD should match the finite difference approximation of the Hessian-vector product (this is a test of the entire incremental map, including the adjoint part)

#Affine state map Tests
a(u,v,p) = ∫( p*(p+1)*∇(u)⋅∇(v) )dΩ
l(v,p) = ∫( f*v )dΩ
state_map = AffineFEStateMap(a,l,U,V,V_p,diff_order=2)
Zygote.gradient(p->objective(state_map(p),p),p) # update λ and u

# incremental state test (ṗ->u̇)
function p_to_u(p)
    ph = FEFunction(V_p,p)
    op = FEOperator((u,v)->a(u,v,ph)-l(v,ph),U,V)
    uh = solve(op)
    return get_free_dof_values(uh)
end
uᵋ = state_map(pᵋ)
u̇ = vec(mapreduce(ForwardDiff.partials, hcat, uᵋ))
∂u_∂p_FD = FiniteDifferences.jacobian(central_fdm(5,1),p_to_u,p)[1]
∂u_∂p_FD_ṗ = ∂u_∂p_FD * ṗ
@test u̇ ≈ ∂u_∂p_FD_ṗ 

# entire incremental map (including the adjoint part) (ṗ->dṗ)
function p_to_j(p)
    ph = FEFunction(V_p,p)
    op = AffineFEOperator((u,v)->a(u,v,ph),v->l(v,ph),U,V)
    uh = solve(op)
    sum(J(uh,ph))
end
g_fd = p->FiniteDifferences.jacobian(central_fdm(5,1),p_to_j,p)
Hṗ_fd = FiniteDifferences.jacobian(central_fdm(5,1),g_fd,p)[1]*ṗ

p_to_j(p) = objective((state_map(p)),p)
@test Hṗ_fd ≈ Hvp(p_to_j,p,ṗ) # the Hessian-vector product computed using AD should match the finite difference approximation of the Hessian-vector product (this is a test of the entire incremental map, including the adjoint part)

end

