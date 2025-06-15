module NonlinearFEStateMapTest

using GridapTopOpt
using Gridap, Gridap.MultiField
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using FiniteDiff
using Test

function main(verbose)
  model = CartesianDiscreteModel((0,1,0,1),(8,8))
  order = 1
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
  _,_,_dF,_ = evaluate!(pcf,φh)

  function φ_to_j(φ)
    u = φ_to_u(φ)
    pcf.J(u,φ)
  end

  fdm_grad = FiniteDiff.finite_difference_gradient(φ_to_j, get_free_dof_values(φh))
  rel_error = norm(_dF - fdm_grad, Inf)/norm(fdm_grad,Inf)

  verbose && println("Relative error in gradient: $rel_error")
  @test rel_error < 1e-6
end

main(false)

end