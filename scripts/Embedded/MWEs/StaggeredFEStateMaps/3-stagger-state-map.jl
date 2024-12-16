include("core.jl")
include("extensions.jl")

model = CartesianDiscreteModel((0,1,0,1),(8,8))
order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
Ω = Triangulation(model)

V_φ = TestFESpace(Ω,reffe)
φh = interpolate(x->(x[1]+1)/2,V_φ)
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

a2((u1,),(u2,u3),(v2,v3),φ) = ∫(φ * u1 * u2 * v2)dΩ + ∫(u3 * v3)dΩ
l2((u1,),(v2,v3),φ) = ∫(sol[2] * u1 * v2)dΩ + ∫(sol[3] * v3)dΩ

a3((u1,(u2,u3)),u4,v4,φ) = ∫(φ * (u1 + u2) * u4 * v4)dΩ
l3((u1,(u2,u3)),v4,φ) = ∫(sol[4] * (u1 + u2) * v4)dΩ

# Create operator from components
UB1, VB1 = U1, V
UB2, VB2 = MultiFieldFESpace([U2,U3]), MultiFieldFESpace([V,V])
UB3, VB3 = U4, V
op = StaggeredAffineFEOperator([a1,a2,a3],[l1,l2,l3],[UB1,UB2,UB3],[VB1,VB2,VB3])

φ_to_u = StaggeredAffineFEStateMap(op,V_φ,U_reg)

xh, cache = forward_solve!(φ_to_u,φh,nothing)
xh, cache = forward_solve!(φ_to_u,φh,cache)

# function _fix_ith_residual_at_jth_xhs(op::StaggeredAffineFEOperator{NB}, xhs, φh, i, j) where NB
#   @assert NB >= i
#   # i == j
#   a(uk,vk) = op.biforms[i](xhs,uk,vk,φh)
#   l(vk) = op.liforms[i](xhs,vk,φh)
#   return (uk,vk) -> a(uk,vk) - l(vk)
# end

function _get_solutions(op::StaggeredFEOperator{NB},xh) where NB
  map(i->get_solution(op,xh,i),Tuple(1:NB))
end

function _get_∂res_k_at_xhi(op::StaggeredAffineFEOperator{NB}, xh_comb, i, k) where NB
  @assert NB >= k && 1 <= i < k
  ak_at_xhi(xhi,vk) = op.biforms[k]((xh_comb[1:i-1]...,xhi,xh_comb[i+1:end-1]...),xh_comb[k],vk)
  lk_at_xhi(xhi,vk) = op.liforms[k]((xh_comb[1:i-1]...,xhi,xh_comb[i+1:end-1]...),vk)
  res_k_at_xhi(xhi,vk) = ak_at_xhi(xhi,vk) - lk_at_xhi(xhi,vk)
  ∂res_k_at_xhi(vk) = ∇(res_k_at_xhi,[xh_comb[i],vk],1)
end

F((u1,(u2,u3),u4),φ) = ∫(1u1*u1 + 2u2*u2 + 3u3*u3 + 4u4*u4 + 5φ*φ)dΩ
xh_comb = _get_solutions(op,xh)
_dFdxj(j) = ∇((xj->F((xh_comb[1:j-1]...,xj,xh_comb[j+1:end]...),φh)))(xh_comb[j])

_biforms, _liforms, _spaces, _assmes, _solvers = φ_to_u.biforms,
  φ_to_u.liforms, φ_to_u.spaces, φ_to_u.assems, φ_to_u.solvers
a_at_φ = map(a->((xhs,uk,vk) -> a(xhs,uk,vk,φh)),_biforms)
l_at_φ = map(l->((xhs,vk) -> l(xhs,vk,φh)),_liforms)
_op = StaggeredAffineFEOperator(a_at_φ,l_at_φ,_spaces.trials,_spaces.tests,_assmes.assmes)

# k = 3
a_adj3((),λ3,Λ3) = _op.biforms[3](xh_comb[1:end-1],Λ3,λ3)
l_adj3((),Λ3) = _dFdxj(3)
# k = 2
a_adj2((λ3,),λ2,Λ2) = _op.biforms[2](xh_comb[1:end-2],Λ2,λ2)
∂res_3_at_xh2 = _get_∂res_k_at_xhi(_op,xh_comb,2,3)
l_adj2((λ3,),Λ2) = _dFdxj(2) - ∂res_3_at_xh2(λ3)
# k = 1
a_adj1((λ3,λ2),λ1,Λ1) = _op.biforms[1](xh_comb[1:end-3],Λ1,λ1)
∂res_3_at_xh1 = _get_∂res_k_at_xhi(_op,xh_comb,1,3)
∂res_2_at_xh1 = _get_∂res_k_at_xhi(_op,xh_comb,1,2)
l_adj1((λ3,λ2),Λ1) = _dFdxj(1) - ∂res_3_at_xh1(λ3) - ∂res_2_at_xh1(λ2)

a_adjs = [a_adj3,a_adj2,a_adj1]
l_adjs = [l_adj3,l_adj2,l_adj1]
# op_adjoint = StaggeredAffineFEOperator(a_adjs,l_adjs,reverse(_spaces.tests),reverse(_spaces.trials))
# op_adjoint = StaggeredAffineFEOperator(a_adjs,l_adjs,reverse(_spaces.trials),reverse(_spaces.tests))
op_adjoint = StaggeredAdjointAffineFEOperator(a_adjs,l_adjs,reverse(_spaces.trials),reverse(_spaces.tests))

λh = zero(op_adjoint.test)
λh, cache = solve!(λh,φ_to_u.solvers.adjoint_solver,op_adjoint,nothing);

# Analytic adjoint
∂F∂u1(du1,(u1,(u2,u3),u4)) = ∫(2du1*u1)dΩ
∂F∂u23((du2,du3),(u1,(u2,u3),u4)) = ∫(4u2*du2 + 6u3*du3)dΩ
∂F∂u4(du4,(u1,(u2,u3),u4)) = ∫(8du4*u4)dΩ

∂F∂u1_vec = assemble_vector(du->∂F∂u1(du,xh_comb),UB1)
∂F∂u1_AD = assemble_vector(_dFdxj(1),UB1)
@assert norm(∂F∂u1_vec - ∂F∂u1_AD) < 1.e-15

∂F∂u23_vec = assemble_vector(du->∂F∂u23(du,xh_comb),UB2)
∂F∂u23_AD = assemble_vector(_dFdxj(2),UB2)
@assert norm(∂F∂u23_vec - ∂F∂u23_AD) < 1.e-15

∂F∂u4_vec = assemble_vector(du->∂F∂u4(du,xh_comb),UB3)
∂F∂u4_AD = assemble_vector(_dFdxj(3),UB3)
@assert norm(∂F∂u4_vec - ∂F∂u4_AD) < 1.e-15

norm(∂F∂u4_vec - get_vector(cache[2][1]))

# Stiffness matrices
∂R1∂u1((),u1,du1,dv1) = a1((),dv1,du1,φh)
∂R1∂u1_mat = assemble_matrix((du,dv)->∂R1∂u1((),xh_comb[1],du,dv),VB1,UB1)

∂R2∂u2((u1,),(u2,u3),(du2,du3),(dv2,dv3)) = a2((u1,),(dv2,dv3),(du2,du3),φh)
∂R2∂u2_mat = assemble_matrix((du,dv)->∂R2∂u2((xh_comb[1],),xh_comb[2],du,dv),VB2,UB2)

∂R3∂u3((u1,(u2,u3)),u4,du4,dv4) = a3((u1,(u2,u3)),dv4,du4,φh)
∂R3∂u3_mat = assemble_matrix((du,dv)->∂R3∂u3((xh_comb[1:2]...,),xh_comb[3],du,dv),VB3,UB3)

# Operator order: λ3 problem, λ2 problem, λ1 problem
adjoint_op3 = cache[2][1]
adjoint_op2 = cache[2][2]
adjoint_op1 = cache[2][3]

norm(get_matrix(adjoint_op1) - ∂R1∂u1_mat,Inf)
norm(get_matrix(adjoint_op2) - ∂R2∂u2_mat,Inf)
norm(get_matrix(adjoint_op3) - ∂R3∂u3_mat,Inf)

# RHS & solutions
λ3 = ∂R3∂u3_mat\∂F∂u4_vec
norm(λ3 - get_free_dof_values(get_solution(op,λh,1)),Inf)/norm(λ3,Inf)

∂R3∂u23((du2,du3),(u1,(u2,u3)),u4,v4) = ∫(φh * du2 * u4 * v4)dΩ - ∫(sol[4] * du2 * v4)dΩ
∂R3∂u23_vec = assemble_vector(du->∂R3∂u23(du,(xh_comb[1:2]...,),xh_comb[3],FEFunction(UB3,λ3)),UB2)

λ2 = ∂R2∂u2_mat\(∂F∂u23_vec - ∂R3∂u23_vec)
get_free_dof_values(get_solution(op_adjoint,λh,2))
norm(λ2 - get_free_dof_values(get_solution(op_adjoint,λh,2)),Inf)/norm(λ2,Inf)

∂R3∂u1(du1,(u1,(u2,u3)),u4,v4) = ∫(φh * du1 * u4 * v4)dΩ - ∫(sol[4] * du1 * v4)dΩ
∂R3∂u1_vec = assemble_vector(du->∂R3∂u1(du,(xh_comb[1:2]...,),xh_comb[3],FEFunction(UB3,λ3)),UB1)
∂R2∂u1(du1,(u1,),(u2,u3),(v2,v3)) = ∫(φh * du1 * u2 * v2)dΩ - ∫(sol[2] * du1 * v2)dΩ
∂R2∂u1_vec = assemble_vector(du->∂R2∂u1(du,(xh_comb[1],),xh_comb[2],FEFunction(UB2,λ2)),UB1)

λ1 = ∂R1∂u1_mat\(∂F∂u1_vec - ∂R2∂u1_vec - ∂R3∂u1_vec)
norm(λ1 - get_free_dof_values(get_solution(op_adjoint,λh,3)),Inf)/norm(λ1,Inf)