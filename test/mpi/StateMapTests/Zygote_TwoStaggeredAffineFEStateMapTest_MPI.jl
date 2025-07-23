module Zygote_TwoStaggeredAffineFEStateMapTestMPI

using GridapTopOpt
using Gridap, Gridap.MultiField
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using Test

using GridapDistributed, PartitionedArrays

function main(model)
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

  op = StaggeredAffineFEOperator([a1,a2],[l1,l2],[U1,U2],[V,V])
  φ_to_u = StaggeredAffineFEStateMap(op,V_φ,φh)
  pcf = PDEConstrainedFunctionals(F,φ_to_u)

  function φ_to_j(φ)
      u = φ_to_u(φ)
      pcf.J(u,φ)
    end

  _,_,_dF,_ = evaluate!(pcf,φh);

  ## Zygote
  for style in (BlockMultiFieldStyle(),ConsecutiveMultiFieldStyle())
    # Spaces and assems
    V_u1φ = MultiFieldFESpace([U1,V_φ];style)
    U_u1u2 = MultiFieldFESpace([U1,U2];style)
    V_u1u2 = MultiFieldFESpace([V,V];style)
    assem_U = SparseMatrixAssembler(U_u1u2,V_u1u2)
    assem_V_φ = SparseMatrixAssembler(V_φ,V_φ)
    # Weak forms
    _a1(u1,v1,φ) = a1((),u1,v1,φ)
    _l1(v1,φ) = l1((),v1,φ)
    _a2(u2,v2,(u1,φ)) = a2((u1,),u2,v2,φ)
    _l2(v2,(u1,φ)) = l2((u1,),v2,φ)
    # StateMaps
    φ_to_u1 = AffineFEStateMap(_a1,_l1,U1,V,V_φ)
    u1φ_to_u2 = AffineFEStateMap(_a2,_l2,U2,V,V_u1φ)
    # StateParamMap
    _F = GridapTopOpt.StateParamMap(F,U_u1u2,V_φ,assem_U,assem_V_φ)

    function φ_to_j2(φ)
      u1 = φ_to_u1(φ)
      u1φ = combine_fields(V_u1φ,u1,φ)
      u2 = u1φ_to_u2(u1φ)
      u1u2 = combine_fields(U_u1u2,u1,u2)
      _F(u1u2,φ)
    end

    cpcf = CustomPDEConstrainedFunctionals(φ_to_j2,0;state_map=φ_to_u1)
    _,_,_dF2,_ = evaluate!(cpcf,φh);

    @test _dF ≈ _dF2
  end
end

function run_test(ranks,mesh_parts,n)
  model = CartesianDiscreteModel(ranks,mesh_parts,(0,1,0,1),(n,n))
  main(model)
end

with_mpi() do distribute
  mesh_parts = (2,2)
  ranks = distribute(LinearIndices((prod(mesh_parts),)))
  run_test(ranks,mesh_parts,8)
end

end