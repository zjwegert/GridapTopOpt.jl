using GridapTopOpt
using Gridap, Gridap.MultiField
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using FiniteDiff
using Test

using GridapDistributed, PartitionedArrays, Gridap.Arrays
using GridapTopOpt: ordered_distributed_model_from_serial_model, test_serial_and_distributed_fields

verbose = true

mesh_parts = (2,2)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(mesh_parts),)))
end

model = CartesianDiscreteModel(ranks,mesh_parts,(0,1,0,1),(8,8))
order = 2
reffe = ReferenceFE(lagrangian,Float64,order)
Ω = Triangulation(model)

V_φ = TestFESpace(model,reffe)
φf(x) = x[1]+1
φh = interpolate(φf,V_φ)

V = FESpace(model,reffe;dirichlet_tags="boundary")

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

# Test derivative
F((u1,u2),φ) = ∫(u1 + u2 + φ)dΩ

## Zygote
# Spaces and assems
V_u1φ = MultiFieldFESpace([U1,V_φ];style=BlockMultiFieldStyle())
U_u1u2 = MultiFieldFESpace([U1,U2];style=BlockMultiFieldStyle())
V_u1u2 = MultiFieldFESpace([V,V];style=BlockMultiFieldStyle())
assem_U = SparseMatrixAssembler(U_u1u2,V_u1u2)
assem_V_φ = SparseMatrixAssembler(V_φ,V_φ)
# Weak forms
_a1(u1,v1,φ) = a1((),u1,v1,φ)
_l1(v1,φ) = l1((),v1,φ)
_a2(u2,v2,(u1,φ)) = a2((u1,),u2,v2,φ)
_l2(v2,(u1,φ)) = l2((u1,),v2,φ)
# StateMaps
φ_to_u1 = AffineFEStateMap(_a1,_l1,U1,V,V_φ,φh)
φ_to_u2 = AffineFEStateMap(_a2,_l2,U2,V,V_u1φ,one(V_u1φ))
# StateParamMap
_F2 = GridapTopOpt.StateParamMap(F,V_u1u2,V_φ,U_u1u2,assem_U,assem_V_φ)

Base.getindex(a::GridapDistributed.BlockPVector,i) = a.blocks[i]

using ChainRulesCore
Zygote.@adjoint GridapDistributed.BlockPVector(a,b) = GridapDistributed.BlockPVector(a,b), y->(nothing,y)

function φ_to_j2(φ)
  u1 = φ_to_u1(φ)
  u1φ = GridapDistributed.BlockPVector([u1,φ],[U1.gids, V_φ.gids])
  u2 = φ_to_u2(u1φ)
  u1u2 = GridapDistributed.BlockPVector([u1,u2],[U1.gids, U2.gids])
  _F2(u1u2,φ)
end

cpcf = CustomPDEConstrainedFunctionals(φ_to_j2,0,φ_to_u1)

_,_,_dF2,_ = evaluate!(cpcf,φh);

op = StaggeredAffineFEOperator([a1,a2],[l1,l2],[U1,U2],[V,V])
φ_to_u = StaggeredAffineFEStateMap(op,V_φ,φh)
pcf = PDEConstrainedFunctionals(F,φ_to_u)

function φ_to_j(φ)
    u = φ_to_u(φ)
    pcf.J(u,φ)
  end

_,_,_dF,_ = evaluate!(pcf,φh);

_dF
_dF2