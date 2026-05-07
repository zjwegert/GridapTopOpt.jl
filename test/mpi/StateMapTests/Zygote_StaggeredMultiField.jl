using GridapTopOpt
using Gridap, Gridap.MultiField
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using Test
using FiniteDiff

using GridapDistributed, PartitionedArrays, Gridap.Arrays
using GridapTopOpt: ordered_distributed_model_from_serial_model, test_serial_and_distributed_fields

function driver(model)
  order = 2
  reffe = ReferenceFE(lagrangian,Float64,order)
  Ω = Triangulation(model)

  V_φ = MultiFieldFESpace([TestFESpace(model,reffe),TestFESpace(model,reffe)])
  φf1(x) = x[1]*x[2]+1;
  φf2(x) = x[2]-x[1];
  φh = interpolate([φf1,φf2],V_φ)
  V = FESpace(model,reffe;dirichlet_tags="boundary")

  rhs = [x -> x[1], x -> (x[1] - x[2])]
  sol = [x -> rhs[1](x)*φf1(x), x -> rhs[2](x)*φf2(x)*φf1(x)]
  U1 = TrialFESpace(V,sol[1])
  U2 = TrialFESpace(V,sol[2])

  # Define weakforms
  dΩ = Measure(Ω,2*order)

  a1(u1,v1,(φ1,φ2)) = ∫(u1 * v1)dΩ
  l1(v1,(φ1,φ2)) = ∫(φ1 * rhs[1] * v1)dΩ

  a2(u2,v2,(u1,φ1,φ2)) = ∫(u1 * u2 * v2)dΩ
  l2(v2,(u1,φ1,φ2)) = ∫(φ1 * φ2 * rhs[2] * u1 * v2)dΩ

  # Test derivative
  F((u1,u2),(φ1,φ2)) = ∫(u1*u2*φ1*φ2)dΩ

  ## Zygote
  for style in (BlockMultiFieldStyle(),ConsecutiveMultiFieldStyle())
    # Spaces and assems
    V_u1φ1φ2 = MultiFieldFESpace([U1,V_φ...];style)
    U_u1u2 = MultiFieldFESpace([U1,U2];style)
    V_u1u2 = MultiFieldFESpace([V,V];style)
    assem_U = SparseMatrixAssembler(U_u1u2,V_u1u2)
    assem_V_φ = SparseMatrixAssembler(V_φ,V_φ)
    # StateMaps
    φ_to_u1 = AffineFEStateMap(a1,l1,U1,V,V_φ)
    u1φ1φ2_to_u2 = AffineFEStateMap(a2,l2,U2,V,V_u1φ1φ2)
    # StateParamMap
    _F = GridapTopOpt.StateParamMap(F,U_u1u2,V_φ,assem_U,assem_V_φ)

    function φ_to_j(φ)
      u1 = φ_to_u1(φ)
      φ1 = restrict(V_φ,φ,1)
      φ2 = restrict(V_φ,φ,2)
      u1φ1φ2 = combine_fields(V_u1φ1φ2,u1,φ1,φ2)
      u2 = u1φ1φ2_to_u2(u1φ1φ2)
      u1u2 = combine_fields(U_u1u2,u1,u2)
      _F(u1u2,φ)
    end

    cpcf = CustomPDEConstrainedFunctionals(φ_to_j,0;state_map=φ_to_u1)
    _,_,dF,_ = evaluate!(cpcf,φh);

    return dF,V_φ
  end
end

function main(distribute,mesh_partition)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  model_serial = CartesianDiscreteModel((0,1,0,1),(8,8));
  dF_serial,V_deriv_serial = driver(model_serial);

  model = ordered_distributed_model_from_serial_model(ranks,model_serial);
  dF,V_deriv = driver(model);

  @test length(dF_serial) ≈ length(dF)
  @test norm(dF_serial) ≈ norm(dF)

  dFh = FEFunction(V_deriv,dF)
  dFh_serial = FEFunction(V_deriv_serial,dF_serial)
  deriv_test = test_serial_and_distributed_fields(dFh,V_deriv,dFh_serial,V_deriv_serial)

  map_main(deriv_test) do deriv_test
    @test deriv_test
    nothing
  end
end

with_mpi() do distribute
  main(distribute,(2,2))
end
