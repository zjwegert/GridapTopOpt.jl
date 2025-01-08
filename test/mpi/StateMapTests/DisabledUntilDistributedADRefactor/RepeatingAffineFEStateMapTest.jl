module RepeatingAffineFEStateMapTestMPI

using GridapTopOpt
using Gridap, Gridap.MultiField
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using FiniteDifferences
using Test

using GridapDistributed, PartitionedArrays, Gridap.Arrays
using GridapTopOpt: ordered_distributed_model_from_serial_model, test_serial_and_distributed_fields

function driver(model,verbose)
  order = 2
  reffe = ReferenceFE(lagrangian,Float64,order)
  Ω = Triangulation(model)

  V_φ = TestFESpace(Ω,reffe)
  φh = interpolate(1,V_φ)
  V_reg = TestFESpace(Ω,reffe)
  U_reg = TrialFESpace(V_reg)

  V = FESpace(Ω,reffe;dirichlet_tags="boundary")

  rhs = x -> x[1]
  sol = x -> rhs(x)
  U = TrialFESpace(V,sol)

  # Define weakforms
  dΩ = Measure(Ω,3*order)

  a1(u1,v1,φ) = ∫(φ * u1 * v1)dΩ
  l1(v1,φ) = ∫(φ * rhs * v1)dΩ
  l2(v1,φ) = ∫(φ * φ * rhs * v1)dΩ

  # Create operator from components
  φ_to_u = RepeatingAffineFEStateMap(2,a1,[l1,l2],U,V,V_φ,U_reg,φh)

  # Test solution
  GridapTopOpt.forward_solve!(φ_to_u,φh)
  xh = get_state(φ_to_u)

  xh_exact = interpolate(sol,U)
  for k in 1:2
    eh_k = xh[k] - xh_exact
    e_k = sqrt(sum(∫(eh_k * eh_k)dΩ))
    verbose && println("Error in field $k: $e_k")
    @test e_k < 1e-10
  end

  φh = interpolate(x->x[1]*x[2]+1,V_φ)

  # Compute gradient
  F(x,φ) = ∫(x[1]*x[2]*φ)dΩ
  pcf = PDEConstrainedFunctionals(F,φ_to_u)
  _,_,dF,_ = evaluate!(pcf,φh)

  return dF,U_reg
end

function main(distribute,mesh_partition)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  model_serial = CartesianDiscreteModel((0,1,0,1),(8,8));
  dF_serial,V_deriv_serial = driver(model_serial,false);

  model = ordered_distributed_model_from_serial_model(ranks,model_serial);
  dF,V_deriv = driver(model,false);

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

end