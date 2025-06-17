module TwoStaggeredAffineFEStateMapTestMPI

using GridapTopOpt
using Gridap, Gridap.MultiField
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using FiniteDiff
using Test

using GridapDistributed, PartitionedArrays, Gridap.Arrays
using GridapTopOpt: ordered_distributed_model_from_serial_model, test_serial_and_distributed_fields

function driver(model,verbose,analytic_partials)
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

  # Create operator from components
  op = StaggeredAffineFEOperator([a1,a2],[l1,l2],[U1,U2],[V,V])

  if analytic_partials
    ∂R2∂xh1(du1,(u1,),u2,v2,φ) = ∫(du1 * u2 * v2)dΩ - ∫(φ * rhs[2] * du1 * v2)dΩ
    ∂Rk∂xhi = ((∂R2∂xh1,),)
    φ_to_u = StaggeredAffineFEStateMap(op,∂Rk∂xhi,V_φ,φh)
  else
    φ_to_u = StaggeredAffineFEStateMap(op,V_φ,φh)
  end

  # Test solution
  GridapTopOpt.forward_solve!(φ_to_u,φh)
  xh = get_state(φ_to_u)
  xh_exact = interpolate(sol,MultiFieldFESpace([U1,U2]))
  for k in 1:length(sol)
    eh_k = xh[k] - xh_exact[k]
    e_k = sqrt(sum(∫(eh_k * eh_k)dΩ))
    verbose && println("Error in field $k: $e_k")
    @test e_k < 1e-10
  end

  # Test derivative
  F((u1,u2),φ) = ∫(u1 + u2 + φ)dΩ

  if analytic_partials
    ∂F∂u12(du1,(u1,u2),φ) = ∫(du1)dΩ
    ∂F∂u3(du2,(u1,u2),φ) = ∫(du2)dΩ
    pcf = PDEConstrainedFunctionals(F,(∂F∂u12,∂F∂u3),φ_to_u)
  else
    pcf = PDEConstrainedFunctionals(F,φ_to_u)
  end
  _,_,dF,_ = evaluate!(pcf,φh);

  return dF,V_φ
end

function main(distribute,mesh_partition,analytic_partials)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  model_serial = CartesianDiscreteModel((0,1,0,1),(8,8));
  dF_serial,V_deriv_serial = driver(model_serial,false,analytic_partials);

  model = ordered_distributed_model_from_serial_model(ranks,model_serial);
  dF,V_deriv = driver(model,false,analytic_partials);

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
  main(distribute,(2,2),true)
  main(distribute,(2,2),false)
end

end