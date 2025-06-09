module NonlinearFEStateMapTestMPI

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

  V_φ = TestFESpace(Ω,reffe)
  φh = interpolate(1,V_φ)
  V_reg = TestFESpace(Ω,reffe)
  U_reg = TrialFESpace(V_reg)

  V = FESpace(Ω,reffe;dirichlet_tags="boundary")

  _sol(x) = x[1] + 1
  U = TrialFESpace(V,_sol)

  # Define weakforms
  dΩ = Measure(Ω,3*order)

  L(u::Function) = x -> (u(x) + 1) * u(x)
  L(u) = (u + 1) * u

  r(u1,v1,φ) = ∫((φ*L(u1) - L(_sol)) * v1)dΩ

  # Create operator from components
  lsolver = LUSolver()
  solver = NewtonSolver(lsolver;rtol=1.e-10,verbose)
  φ_to_u = NonlinearFEStateMap(r,U,V,V_φ,U_reg,φh;nls=solver)

  ## Test solution
  GridapTopOpt.forward_solve!(φ_to_u,φh)
  xh = get_state(φ_to_u)
  xh_exact = interpolate(_sol,U)
  eh = xh - xh_exact
  e = sqrt(sum(∫(eh * eh)dΩ))
  verbose && println("Error in field: $e")
  @test e < 1e-10

  ## Update LSF for testing gradient
  φh = interpolate(x->x[1]*x[2]+1,V_φ)

  F(u1,φ) = ∫(u1*φ)dΩ
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