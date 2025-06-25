module AffineFEStateMapTest

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
  φ_to_u = AffineFEStateMap(a1,l1,U,V,V_φ,φh)

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
  _,_,_dF,_ = evaluate!(pcf,φh);

  function φ_to_j(φ)
    u = φ_to_u(φ)
    pcf.J(u,φ)
  end

  cpcf = CustomPDEConstrainedFunctionals(φ_to_j,φ_to_u,φh)
  _,_,cdF,_ = evaluate!(cpcf,φh)
  @test cdF ≈ _dF

  fdm_grad = FiniteDiff.finite_difference_gradient(φ_to_j, get_free_dof_values(φh))
  rel_error = norm(_dF - fdm_grad, Inf)/norm(fdm_grad,Inf)

  verbose && println("Relative error in gradient: $rel_error")
  @test rel_error < 1e-8
end

main(false)

end