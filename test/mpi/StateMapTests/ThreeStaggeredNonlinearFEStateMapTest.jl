module ThreeStaggeredNonlinearFEStateMapTestMPI

using GridapTopOpt
using Gridap, Gridap.MultiField
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using FiniteDiff
using Test

using GridapDistributed, PartitionedArrays, Gridap.Arrays
using GridapTopOpt: ordered_distributed_model_from_serial_model, test_serial_and_distributed_fields

function driver(model,verbose,analytic_partials)
  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  Ω = Triangulation(model)
  V = FESpace(model,reffe;dirichlet_tags="boundary")

  sol = [x -> x[1], x -> x[2], x -> x[1] + x[2], x -> 2.0*x[1]]
  U1 = TrialFESpace(V,sol[1])
  U2 = TrialFESpace(V,sol[2])
  U3 = TrialFESpace(V,sol[3])
  U4 = TrialFESpace(V,sol[4])

  # Define weakforms
  dΩ = Measure(Ω,4*order)

  V_φ = TestFESpace(model,reffe)
  φh = interpolate(1,V_φ)

  F(u::Function) = x -> (u(x) + 1) * u(x)
  F(u) = (u + 1) * u
  dF(u,du) = 2.0 * u * du + du

  j1((),u1,du1,dv1,φ) = ∫(dF(u1,du1) * dv1)dΩ
  r1((),u1,v1,φ) = ∫((F(u1) - φ * F(sol[1])) * v1)dΩ

  j2((u1,),(u2,u3),(du2,du3),(dv2,dv3),φ) = ∫(u1 * φ *  dF(u2,du2) * dv2)dΩ + ∫(dF(u3,du3) * dv3)dΩ
  r2((u1,),(u2,u3),(v2,v3),φ) = ∫(φ * u1 * (F(u2) - F(sol[2])) * v2)dΩ + ∫((F(u3) - F(sol[3])) * v3)dΩ

  j3((u1,(u2,u3)),u4,du4,dv4,φ) = ∫(φ * u3 * dF(u4,du4) * dv4)dΩ
  r3((u1,(u2,u3)),u4,v4,φ) = ∫(u3 * (φ * F(u4) - F(sol[4])) * v4)dΩ

  # Define solver: Each block will be solved with a LU solver
  lsolver = LUSolver()
  nlsolver = NewtonSolver(lsolver;rtol=1.e-10,verbose)
  solver = StaggeredFESolver(fill(nlsolver,3))

  # Create operator from full spaces
  mfs = BlockMultiFieldStyle(3,(1,2,1))
  X = MultiFieldFESpace([U1,U2,U3,U4];style=mfs)
  Y = MultiFieldFESpace([V,V,V,V];style=mfs)
  op = StaggeredNonlinearFEOperator([r1,r2,r3],[j1,j2,j3],X,Y)

  if analytic_partials
    ∂R2∂xh1(du1,(u1,),(u2,u3),(v2,v3),φ) = ∫(φ * du1 * (F(u2) - F(sol[2])) * v2)dΩ
    ∂R3∂xh1(du1,(u1,(u2,u3)),u4,v4,φ) = ∫(0du1)dΩ
    ∂R3∂xh2((du2,du3),(u1,(u2,u3)),u4,v4,φ) = ∫(du3 * (φ * F(u4) - F(sol[4])) * v4)dΩ
    ∂Rk∂xhi = ((∂R2∂xh1,),(∂R3∂xh1,∂R3∂xh2))
    φ_to_u = StaggeredNonlinearFEStateMap(op,∂Rk∂xhi,V_φ,φh;solver)
  else
    φ_to_u = StaggeredNonlinearFEStateMap(op,V_φ,φh;solver)
  end

  ## Test solution
  GridapTopOpt.forward_solve!(φ_to_u,φh)
  xh = get_state(φ_to_u)
  xh_exact = interpolate(sol,op.trial)
  for k in 1:length(sol)
    eh_k = xh[k] - xh_exact[k]
    e_k = sqrt(sum(∫(eh_k * eh_k)dΩ))
    verbose && println("Error in field $k: $e_k")
    @test e_k < 1e-10
  end

  ## Update LSF for testing gradient
  φh = interpolate(x->x[1]*x[2]+1,V_φ)
  GridapTopOpt.forward_solve!(φ_to_u,φh)
  xh = get_state(φ_to_u)

  ## Test gradient
  F((u1,(u2,u3),u4),φ) = ∫(u1*u2*u3*u4*φ)dΩ

  if analytic_partials
    ∂F∂u1(du1,(u1,(u2,u3),u4),φ) = ∫(du1*u2*u3*u4*φ)dΩ
    ∂F∂u23((du2,du3),(u1,(u2,u3),u4),φ) = ∫(u1*du2*u3*u4*φ)dΩ + ∫(u1*u2*du3*u4*φ)dΩ
    ∂F∂u4(du4,(u1,(u2,u3),u4),φ) = ∫(u1*u2*u3*du4*φ)dΩ
    pcf = PDEConstrainedFunctionals(F,(∂F∂u1,∂F∂u23,∂F∂u4),φ_to_u)
  else
    pcf = PDEConstrainedFunctionals(F,φ_to_u)
  end
  _,_,dF,_ = evaluate!(pcf,φh)

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