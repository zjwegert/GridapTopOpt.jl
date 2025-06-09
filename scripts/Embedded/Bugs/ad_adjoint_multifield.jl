using Gridap, Gridap.FESpaces, Gridap.CellData
using Test

function _transpose_contributions(b::DomainContribution)
  c = DomainContribution()
  for (trian,array_old) in b.dict
    array_new = lazy_map(transpose,array_old)
    add_contribution!(c,trian,array_new)
  end
  return c
end

model = CartesianDiscreteModel((0,1,0,1),(4,4))
dΩ = Measure(get_triangulation(model),2)
V = FESpace(model,ReferenceFE(lagrangian,Float64,1);dirichlet_tags="boundary")
U = TrialFESpace(V,x->x[1])

UB = MultiFieldFESpace([U,U])
VB = MultiFieldFESpace([V,V])
uh = interpolate([x->x[1],x->x[1]],UB)

r((u1,u2),(v1,v2)) = ∫(u1*u1*v1 + u2*u2*v2 + u2*v1)dΩ
j(u,du,v) = jacobian((u,v)->r(u,v),[u,v],1)
j_anal((u1,u2),(du1,du2),(v1,v2)) = ∫(2u1*du1*v1 + 2u2*du2*v2 + du2*v1)dΩ

# Assemble jacobian
u = get_trial_fe_basis(UB)
v = get_fe_basis(VB)
K = assemble_matrix(j(uh,u,v),UB,VB)
K_anal = assemble_matrix(j_anal(uh,u,v),UB,VB)

@test norm(K - K_anal,Inf) == 0
@test norm(K - K',Inf) > 0
@warn """
  Analytic matrix has $(length(K_anal.nzval)) non-zero entries,
      while matrix from AD has $(length(K.nzval)) non-zero entries.
  """ # (1) The jacobian's have different sparsity patterns

# Assemble adjoint of jacobian
K_adjoint = assemble_matrix(j(uh,v,u),V,U) # (2) This produces an error!
K_adjoint_direct = assemble_matrix(_transpose_contributions(j(uh,u,v)),UB,VB)
K_adjoint_anal = assemble_matrix(j_anal(uh,v,u),VB,UB)

@test norm(K' - K_adjoint_direct,Inf) == 0
@test norm(K' - K_adjoint_anal,Inf) == 0