module TwoStaggeredAffineFEStateMapTest

using GridapTopOpt
using Gridap, Gridap.MultiField
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using FiniteDiff
using Test

function main(;verbose,analytic_partials)
  model = CartesianDiscreteModel((0,1,0,1),(8,8))
  order = 2
  reffe = ReferenceFE(lagrangian,Float64,order)
  Ω = Triangulation(model)

  V_φ = TestFESpace(Ω,reffe)
  φf(x) = x[1]+1
  φh = interpolate(φf,V_φ)
  V_reg = TestFESpace(Ω,reffe)
  U_reg = TrialFESpace(V_reg)

  V = FESpace(Ω,reffe;dirichlet_tags="boundary")

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
    φ_to_u = StaggeredAffineFEStateMap(op,∂Rk∂xhi,V_φ,U_reg,φh)
  else
    φ_to_u = StaggeredAffineFEStateMap(op,V_φ,U_reg,φh)
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
  _,_,_dF,_ = evaluate!(pcf,φh);

  function φ_to_j(φ)
    u = φ_to_u(φ)
    pcf.J(u,φ)
  end

  fdm_grad = FiniteDiff.finite_difference_gradient(φ_to_j, get_free_dof_values(φh))
  rel_error = norm(_dF - fdm_grad, Inf)/norm(fdm_grad,Inf)

  verbose && println("Relative error in gradient: $rel_error")
  @test rel_error < 1e-8
end

main(verbose = false, analytic_partials = true)
main(verbose = false, analytic_partials = false)

end