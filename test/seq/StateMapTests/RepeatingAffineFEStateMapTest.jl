module RepeatingAffineFEStateMapTest

using GridapTopOpt
using Gridap, Gridap.MultiField
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using FiniteDiff
using Test

function main(verbose)
  model = CartesianDiscreteModel((0,1,0,1),(8,8))
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
  _,_,_dF,_ = evaluate!(pcf,φh)

  function φ_to_j(φ)
    u = φ_to_u(φ)
    pcf.J(u,φ)
  end

  fdm_grad = FiniteDiff.finite_difference_gradient(φ_to_j, get_free_dof_values(φh))
  rel_error = norm(_dF - fdm_grad, Inf)/norm(fdm_grad,Inf)

  verbose && println("Relative error in gradient: $rel_error")
  @test rel_error < 1e-8
end

main(false)

end