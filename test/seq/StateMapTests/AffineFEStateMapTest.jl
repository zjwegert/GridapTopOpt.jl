module AffineFEStateMapTest

using GridapTopOpt
using Gridap, Gridap.MultiField
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using FiniteDifferences
using Test

function main(verbose)
  model = CartesianDiscreteModel((0,1,0,1),(8,8))
  order = 2
  reffe = ReferenceFE(lagrangian,Float64,order)
  Ω = Triangulation(model)

  V_φ = TestFESpace(Ω,reffe)
  φf(x) = x[1]*x[2]+1
  φh = interpolate(φf,V_φ)
  V_reg = TestFESpace(Ω,reffe)
  U_reg = TrialFESpace(V_reg)

  V = FESpace(Ω,reffe;dirichlet_tags="boundary")

  _rhs(x) = x[1] - x[2]
  _sol(x) = _rhs(x)*φf(x)
  U = TrialFESpace(V,_sol)

  # Define weakforms
  dΩ = Measure(Ω,3*order)

  a1(u1,v1,φ) = ∫(φ * u1 * v1)dΩ
  l1(v1,φ) = ∫(φ* φ * _rhs * v1)dΩ

  # Create operator from components
  φ_to_u = AffineFEStateMap(a1,l1,U,V,V_φ,U_reg,φh)

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

  fdm_grad = FiniteDifferences.grad(central_fdm(5, 1), φ_to_j, get_free_dof_values(φh))[1]
  rel_error = norm(_dF - fdm_grad, Inf)/norm(fdm_grad,Inf)

  verbose && println("Relative error in gradient: $rel_error")
  @test rel_error < 1e-10
end

main(false)

end