module MultiFieldAffineFEStateMapTest

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

  rhs = [x->x[1],x->x[1]-x[2]]
  sol = [x->rhs[1](x)+rhs[2](x)*φf(x),x->rhs[2](x)]
  U1, U2 = TrialFESpace(V,sol[1]), TrialFESpace(V,sol[2])
  UB = MultiFieldFESpace([U1,U2])
  VB = MultiFieldFESpace([V,V])

  # Define weakforms
  dΩ = Measure(Ω,3*order)

  a1((u1,u2),(v1,v2),φ) = ∫(u1 * v1 + u2 * v2)dΩ - ∫(u2*v1*φ)dΩ
  l1((v1,v2),φ) = ∫(rhs[1] * v1 + v2*rhs[2])dΩ

  # Create operator from components
  φ_to_u = AffineFEStateMap(a1,l1,UB,VB,V_φ)

  # Test solution
  GridapTopOpt.forward_solve!(φ_to_u,φh)
  xh = get_state(φ_to_u)

  xh_exact = interpolate(sol,UB)
  for k in 1:2
    eh_k = xh[k] - xh_exact[k]
    e_k = sqrt(sum(∫(eh_k * eh_k)dΩ))
    verbose && println("Error in field $k: $e_k")
    @test e_k < 1e-10
  end

  # Compute gradient
  F((u1,u2),φ) = ∫(u1*u2*φ)dΩ
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