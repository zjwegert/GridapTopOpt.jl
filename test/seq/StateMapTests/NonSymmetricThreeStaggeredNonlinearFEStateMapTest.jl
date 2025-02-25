module NonSymmetricThreeStaggeredNonlinearFEStateMapTest

using GridapTopOpt
using Gridap, Gridap.MultiField
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using FiniteDiff
using Test

function main(;verbose,analytic_partials)
  model = CartesianDiscreteModel((0,1,0,1),(8,8))
  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  Ω = Triangulation(model)
  V = FESpace(Ω,reffe;dirichlet_tags="boundary")

  sol = [x -> x[1], x -> x[2], x -> x[1] + x[2], x -> 2.0*x[1]]
  U1 = TrialFESpace(V,sol[1])
  U2 = TrialFESpace(V,sol[2])
  U3 = TrialFESpace(V,sol[3])
  U4 = TrialFESpace(V,sol[4])

  # Define weakforms
  dΩ = Measure(Ω,4*order)

  V_φ = TestFESpace(Ω,reffe)
  φh = interpolate(1,V_φ)
  V_reg = TestFESpace(Ω,reffe)
  U_reg = TrialFESpace(V_reg)

  F(u::Function) = x -> (u(x) + 1) * u(x)
  F(u) = (u + 1) * u
  dF(u,du) = 2.0 * u * du + du

  r1((),u1,v1,φ) = ∫((F(u1) - φ * F(sol[1])) * v1)dΩ
  r2((u1,),(u2,u3),(v2,v3),φ) = ∫(φ * u1 * (F(u2) - F(sol[2])) * v2)dΩ + ∫((F(u3) - F(sol[3])) * v3)dΩ + ∫(u2 * v3)dΩ
  r3((u1,(u2,u3)),u4,v4,φ) = ∫(u3 * (φ * F(u4) - F(sol[4])) * v4)dΩ

  if analytic_partials
    j1 = ((),u1,du1,dv1,φ) -> ∫(dF(u1,du1) * dv1)dΩ
    j2 = ((u1,),(u2,u3),(du2,du3),(dv2,dv3),φ) -> ∫(u1 * φ *  dF(u2,du2) * dv2)dΩ + ∫(dF(u3,du3) * dv3)dΩ + ∫(du2 * dv3)dΩ
    j3 = ((u1,(u2,u3)),u4,du4,dv4,φ) -> ∫(φ * u3 * dF(u4,du4) * dv4)dΩ
  else
    j1 = ((),u1,du1,v1,φ) -> Gridap.jacobian((u1,v1,φ)->r1((),u1,v1,φ),[u1,v1,φ],1)
    j2 = ((u1,),u23,du23,v23,φ) -> Gridap.jacobian((u23,v23,φ)->r2((u1,),u23,v23,φ),[u23,v23,φ],1)
    j3 = ((u1,u23),u4,du4,v4,φ) -> Gridap.jacobian((u4,v4,φ)->r3((u1,u23),u4,v4,φ),[u4,v4,φ],1)
  end

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
    φ_to_u = StaggeredNonlinearFEStateMap(op,∂Rk∂xhi,V_φ,U_reg,φh;solver)
  else
    φ_to_u = StaggeredNonlinearFEStateMap(op,V_φ,U_reg,φh;solver)
  end

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

main(verbose = false, analytic_partials = true)
main(verbose = false, analytic_partials = false)

end