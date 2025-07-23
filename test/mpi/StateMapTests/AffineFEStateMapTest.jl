module AffineFEStateMapTestMPI

using GridapTopOpt
using Gridap, Gridap.MultiField
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using FiniteDiff
using Test

using GridapDistributed, PartitionedArrays, Gridap.Arrays
using GridapTopOpt: ordered_distributed_model_from_serial_model, test_serial_and_distributed_fields

function driver(model,verbose)
  order = 2
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
  dΩ = Measure(Ω,3*order)

  a1(u1,v1,φ) = ∫(φ * u1 * v1)dΩ
  l1(v1,φ) = ∫(φ* φ * _rhs * v1)dΩ

  # Create operator from components
  φ_to_u = AffineFEStateMap(a1,l1,U,V,V_φ)

  # Test solution
  GridapTopOpt.forward_solve!(φ_to_u,φh)
  xh = get_state(φ_to_u)

  xh_exact = interpolate(_sol,U)
  eh = xh - xh_exact
  e = sqrt(sum(∫(eh * eh)dΩ))
  verbose && println("Error in field: $e")
  @test e < 1e-15

  # Compute gradient
  F(u1,φ) = ∫(u1*φ+1)dΩ
  pcf = PDEConstrainedFunctionals(F,φ_to_u)
  _,_,dF,_ = evaluate!(pcf,φh);

  function φ_to_j(φ)
    u = φ_to_u(φ)
    pcf.J(u,φ)
  end

  cpcf = CustomPDEConstrainedFunctionals(φ_to_j,0;state_map=φ_to_u)
  _,_,cdF,_ = evaluate!(cpcf,φh)
  @test cdF ≈ dF

  function φ_to_j_v2(φ)
    u = φ_to_u(φ)
    [pcf.J(u,φ)]
  end

  cpcf = CustomPDEConstrainedFunctionals(φ_to_j_v2,0;state_map=φ_to_u)
  _,_,cdF,_ = evaluate!(cpcf,φh)
  @test cdF ≈ dF

  function φ_to_j3(φ)
    u = φ_to_u(φ)
    [pcf.J(u,φ),pcf.J(u,φ)^2]
  end

  cpcf = CustomPDEConstrainedFunctionals(φ_to_j3,1;state_map=φ_to_u)
  evaluate!(cpcf,φh)
  cpcf = CustomPDEConstrainedFunctionals(φ_to_j3,1;state_map=φ_to_u,analytic_dC=[(dC,φ)->dC])
  evaluate!(cpcf,φh)

  return dF,V_φ
end

function main(distribute,mesh_partition)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  model_serial = CartesianDiscreteModel((0,1,0,1),(8,8));
  dF_serial,V_deriv_serial = driver(model_serial,false);

  model = ordered_distributed_model_from_serial_model(ranks,model_serial);
  dF,V_deriv = driver(model,false);

  @test length(dF_serial) == length(dF)
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

end