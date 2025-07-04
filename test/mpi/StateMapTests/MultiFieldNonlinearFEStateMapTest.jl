module MultiFieldNonlinearFEStateMapTestMPI

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

  rhs = [x -> x[2], x -> x[1] + x[2]]
  sol = [x -> rhs[1](x), x -> rhs[2](x) + rhs[1](x)*φf(x)]
  U1 = TrialFESpace(V,sol[1])
  U2 = TrialFESpace(V,sol[2])
  UB = MultiFieldFESpace([U1,U2])
  VB = MultiFieldFESpace([V,V])

  # Define weakforms
  dΩ = Measure(Ω,3*order)

  L(u::Function) = x -> (u(x) + 1) * u(x)
  L(u) = (u + 1) * u

  r((u1,u2),(v1,v2),φ) = ∫(φ * (L(u1) - L(sol[1])) * v1)dΩ + ∫(u2 * v2 - u1*v2*φ - v2*rhs[2])dΩ

  # Create operator from components
  lsolver = LUSolver()
  solver = NewtonSolver(lsolver;rtol=1.e-10,verbose)
  φ_to_u = NonlinearFEStateMap(r,UB,VB,V_φ,φh;nls=solver)

  ## Test solution
  GridapTopOpt.forward_solve!(φ_to_u,φh)
  xh = get_state(φ_to_u)
  xh_exact = interpolate(sol,UB)
  for k in 1:length(sol)
    eh_k = xh[k] - xh_exact[k]
    e_k = sqrt(sum(∫(eh_k * eh_k)dΩ))
    verbose && println("Error in field $k: $e_k")
    @test e_k < 1e-10
  end

  ## Update LSF for testing gradient
  F((u1,u2),φ) = ∫(u2*u1*φ)dΩ
  pcf = PDEConstrainedFunctionals(F,φ_to_u)
  _,_,dF,_ = evaluate!(pcf,φh)

  return dF,V_φ
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