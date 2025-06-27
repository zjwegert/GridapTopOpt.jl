module AffineFEStateMap_ZygoteJacobiansTestMPI

using Gridap, Gridap.FESpaces, Gridap.CellData, Gridap.Helpers
using GridapDistributed, PartitionedArrays
using GridapTopOpt
using Test

using GridapTopOpt: val_and_jacobian, val_and_gradient

function main(model)
  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  Ω = Triangulation(model)
  V_φ = TestFESpace(model,reffe)
  φf(x) = x[1]*x[2]+1
  φh = interpolate(φf,V_φ)

  V = FESpace(model,reffe;dirichlet_tags="boundary")
  _rhs(x) = x[1] - x[2]
  _sol(x) = _rhs(x)*φf(x)
  U = TrialFESpace(V,_sol)

  # Define weakforms
  dΩ = Measure(Ω,2*order)

  a1(u1,v1,φ) = ∫(φ * u1 * v1)dΩ
  l1(v1,φ) = ∫(φ* φ * _rhs * v1)dΩ

  # Create operator from components
  φ_to_u = AffineFEStateMap(a1,l1,U,V,V_φ,φh)

  # Compute gradient
  F(u,φ) = ∫(u*φ+1)dΩ
  _F = GridapTopOpt.StateParamMap(F,φ_to_u)
  F2(u,φ) = ∫(2u*φ+1)dΩ
  _F2 = GridapTopOpt.StateParamMap(F2,φ_to_u)

  function φ_to_j(φ)
    u = φ_to_u(φ)
    _F(u,φ)^2
  end

  out1 = val_and_gradient(φ_to_j, get_free_dof_values(φh))

  function φ_to_jc(φ)
    u = φ_to_u(φ)
    [_F(u,φ),_F2(u,φ)^2]
  end

  out2 = val_and_jacobian(φ_to_jc, get_free_dof_values(φh))
  out3 = val_and_jacobian(φ_to_jc, get_free_dof_values(φh);ignore_pullback=2)

  return out1, out2, out3, V_φ
end

function run_test(ranks,model_serial)
  out1,out2,out3,V = main(model_serial);

  model = GridapTopOpt.ordered_distributed_model_from_serial_model(ranks,model_serial);
  dout1,dout2,dout3,dV = main(model);

  @test out1.val ≈ dout1.val
  @test out2.val ≈ dout2.val
  @test out3.val ≈ dout3.val

  gradh_1 = FEFunction(V,out1.grad[1])
  dgradh_1 = FEFunction(dV,dout1.grad[1])
  deriv_test1 = GridapTopOpt.test_serial_and_distributed_fields(dgradh_1,dV,gradh_1,V)

  gradh_2_1 = FEFunction(V,out2.grad[1][1])
  dgradh_2_1 = FEFunction(dV,dout2.grad[1][1])
  deriv_test21 = GridapTopOpt.test_serial_and_distributed_fields(dgradh_2_1,dV,gradh_2_1,V)

  gradh_2_2 = FEFunction(V,out2.grad[1][2])
  dgradh_2_2 = FEFunction(dV,dout2.grad[1][2])
  deriv_test22 = GridapTopOpt.test_serial_and_distributed_fields(dgradh_2_2,dV,gradh_2_2,V)

  @test iszero(out3.grad[1][2]) # Check that the second term is zero as pullback not computed
  gradh_3_2 = FEFunction(V,out3.grad[1][2])
  dgradh_3_2 = FEFunction(dV,dout3.grad[1][2])
  deriv_test33 = GridapTopOpt.test_serial_and_distributed_fields(dgradh_3_2,dV,gradh_3_2,V)

  map_main(deriv_test1,deriv_test21,deriv_test22,deriv_test33) do deriv_test1,
      deriv_test21,deriv_test22,deriv_test33
    @test deriv_test1
    @test deriv_test21
    @test deriv_test22
    @test deriv_test33
    nothing
  end
end

with_mpi() do distribute
  mesh_parts = (2,2)
  ranks = distribute(LinearIndices((prod(mesh_parts),)))
  model_serial = CartesianDiscreteModel((0,1,0,1),(8,8));
  run_test(ranks,model_serial)
end

end