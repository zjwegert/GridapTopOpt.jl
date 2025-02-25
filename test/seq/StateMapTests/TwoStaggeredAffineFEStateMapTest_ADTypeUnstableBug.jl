module TwoStaggeredAffineFEStateMapTest_ADTypeUnstableBug

using GridapTopOpt
using Gridap, Gridap.MultiField
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using FiniteDiff
using Test

function main(;verbose)
  model = CartesianDiscreteModel((0,1,0,1),(4,4))
  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  Ω = Triangulation(model)
  Ω1 = Triangulation(model,1:12)
  Ω2 = Triangulation(model,9:16)

  V_φ = TestFESpace(Ω,reffe)
  φf(x) = x[1]+1
  φh = interpolate(φf,V_φ)
  V_reg = TestFESpace(Ω,reffe)
  U_reg = TrialFESpace(V_reg)

  V1 = FESpace(Ω1,reffe;dirichlet_tags="boundary")
  VB1 = MultiFieldFESpace([V1,V1])
  VB2 = FESpace(Ω2,reffe;dirichlet_tags="boundary")

  rhs = [x -> 1,x -> 2, x -> 3]
  sol = [x -> rhs[1](x)*φf(x),x -> rhs[2](x)*φf(x), x -> rhs[3](x)*φf(x)]
  U1 = TrialFESpace(V1,sol[1])
  U2 = TrialFESpace(V1,sol[2])

  UB1 = MultiFieldFESpace([U1,U2])
  UB2 = TrialFESpace(VB2,sol[3])

  # Define weakforms
  dΩ = Measure(Ω,2*order)
  dΩ1 = Measure(Ω1,2*order)
  dΩ2 = Measure(Ω2,2*order)

  a1((),(u1,u2),(v1,v2),φ) = ∫(u1 * v1)dΩ1 + ∫(u2 * v2)dΩ1
  l1((),(v1,v2),φ) = ∫(φ * rhs[1] * v1)dΩ1 + ∫(φ * rhs[2] * v2)dΩ1

  a2(((u1,u2),),u3,v3,φ) = ∫(φ * (u1 + u2) * u3 * v3)dΩ2
  l2(((u1,u2),),v3,φ) = ∫(φ * φ * rhs[3] * (u1 + u2) * v3)dΩ2

  # Create operator from components
  op = StaggeredAffineFEOperator([a1,a2],[l1,l2],[UB1,UB2],[VB1,VB2])

  ## This fails due to Gridap#1062
  # op_at_φ = GridapTopOpt.get_staggered_operator_at_φ(op,φh)
  # _solver = StaggeredFESolver([LUSolver(),LUSolver()])
  # xh = solve(_solver,op_at_φ);
  # ubh1 = GridapSolvers.BlockSolvers.get_solution(op,xh,1)
  # ub2h = GridapSolvers.BlockSolvers.get_solution(op,xh,2)
  # ∂a2∂ub1(v3) = ∇(ub1->a2((ub1,),ub2h,v3,φh),ubh1)
  # ∂a2∂ub1(ub2h)
  ## Alternate
  ∂R2∂xh1((du1,du2),((u1,u2),),u3,v3,φ) = ∫(φ * (du1 + du2) * u3 * v3)dΩ2 - ∫(φ * φ * rhs[3] * (du1 + du2) * v3)dΩ2
  ∂Rk∂xhi = ((∂R2∂xh1,),)

  φ_to_u = StaggeredAffineFEStateMap(op,∂Rk∂xhi,V_φ,U_reg,φh)

  # Test solution
  GridapTopOpt.forward_solve!(φ_to_u,φh)
  xh = get_state(φ_to_u);

  dΩs = [dΩ1,dΩ1,dΩ2]
  xh_exact = interpolate(sol,op.trial);
  for k in 1:length(sol)
    eh_k = xh[k] - xh_exact[k]
    e_k = sqrt(sum(∫(eh_k * eh_k)dΩs[k]))
    verbose && println("Error in field $k: $e_k")
    @test e_k < 1e-10
  end

  # Test derivative
  F(((u1,u2),u3),φ) = ∫(u1 + u2 + u3 + φ)dΩ
  ∂F∂u12((du1,du2),((u1,u2),u3),φ) = ∫(du1 + du2)dΩ
  ∂F∂u3(du3,((u1,u2),u3),φ) = ∫(du3)dΩ

  pcf = PDEConstrainedFunctionals(F,(∂F∂u12,∂F∂u3),φ_to_u)
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

main(verbose=false)
end