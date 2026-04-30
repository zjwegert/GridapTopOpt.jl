# module SecondOrderTests
using Test, Gridap, GridapTopOpt
using Zygote
using ForwardDiff
using PartitionedArrays, GridapDistributed

# Compute hvp using FD + val_and_gradient - Note, this relies on val_and_gradient being correct, but this is tested in other places.
function fd_hvp(f, p, v; h::Real = cbrt(eps()))
  ∇f(q) = val_and_gradient(f, q).grad[1]
  g₋₂ = ∇f(p - 2h*v)
  g₋₁ = ∇f(p - h*v)
  g₊₁ = ∇f(p + h*v)
  g₊₂ = ∇f(p + 2h*v)
  return (g₋₂ - 8*g₋₁ + 8*g₊₁ - g₊₂) / (12h)
end

function main(distribute,mesh_partition)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  # FE setup
  order = 1
  xmax = ymax = 1.0
  dom = (0,xmax,0,ymax)
  el_size = (4,4)
  CartesianDiscreteModel(dom,el_size)
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe_scalar;dirichlet_tags=[1,2,3,4,5,6])
  U = TrialFESpace(V,0.0)
  V_p = TestFESpace(model,reffe_scalar;dirichlet_tags=[2,3,4,5,6,7,8])

  #########################################
  # Second order partial derivative tests #
  #########################################

  J(u,p) = ∫(u*u*p*p)dΩ # keep p term otherwise dual error
  uh = interpolate(x->rand(), U)
  ph = interpolate(x->rand(), V_p)
  λh = interpolate(x->rand(), V)
  spaces = (U,V_p)
  _,_,_, ∂2J∂u2_mat, _, ∂2J∂u∂p_mat, _, ∂2J∂p2_mat, _, ∂2J∂p∂u_mat,_,_ = GridapTopOpt.build_inc_obj_cache(J,uh,ph,spaces);
  #∂2J∂u2_mat, ∂2J∂u∂p_mat, ∂2J∂p2_mat, ∂2J∂p∂u_mat = SecondOrderTopOpt.incremental_objective_partials(J,uh,ph,spaces)

  # ∂²J / ∂u² * u̇
  dv = get_fe_basis(V)
  du = get_trial_fe_basis(U)
  dp = get_fe_basis(V_p)
  dp_ = get_trial_fe_basis(V_p)

  ∂2∂u2_analytical(uh) = ∫( 2*ph*ph*du⋅dv )dΩ
  ∂2∂u2_matrix_analytical = assemble_matrix(∂2∂u2_analytical(uh),U,U)
  @test reduce(&,map(≈,partition(∂2∂u2_matrix_analytical),partition(∂2J∂u2_mat)))

  # ∂/∂p (∂J/∂u ) * ṗ
  ∂2J∂u∂p_analytical(uh,ph) = ∫( 4*ph*uh*dp_⋅dv )dΩ
  ∂2J∂u∂p_matrix_analytical = assemble_matrix(∂2J∂u∂p_analytical(uh,ph),V_p,U)
  @test reduce(&,map(≈,partition(∂2J∂u∂p_matrix_analytical),partition(∂2J∂u∂p_mat)))

  # ∂²J / ∂p² * ṗ
  ∂2J∂p2_analytical(uh) = ∫( 2*uh*uh*dp⋅dp_ )dΩ
  ∂2J∂p2_matrix_analytical = assemble_matrix(∂2J∂p2_analytical(uh),V_p,V_p)
  @test reduce(&,map(≈,partition(∂2J∂p2_matrix_analytical),partition(∂2J∂p2_mat)))

  # ∂/∂u (∂J / ∂p) * u̇
  ∂2J∂p∂u_analytical(uh,ph) = ∫( 4*uh*ph*du⋅dp )dΩ
  ∂2J∂p∂u_matrix_analytical = assemble_matrix(∂2J∂p∂u_analytical(uh,ph),U,V_p)
  @test reduce(&,map(≈,partition(∂2J∂p∂u_matrix_analytical),partition(∂2J∂p∂u_mat)))

  f(x) = 1.0
  res(u,v,p) = ∫( p*∇(u)⋅∇(v) - f*v )dΩ
  state_map = NonlinearFEStateMap(res,U,V,V_p,diff_order=2)
  ∂2R∂u2_mat, ∂2R∂u∂p_mat, ∂2R∂p2_mat, ∂2R∂p∂u_mat = GridapTopOpt.update_incremental_adjoint_partials!(state_map,uh,ph,λh)

  # ∂²R / ∂u² * u̇ * λ
  ∂2∂u2R_analytical(uh,λh,ph) = ∫( 0*du*dv )dΩ
  ∂2∂u2R_matrix_analytical = assemble_matrix(∂2∂u2R_analytical(uh,λh,ph),U,U)
  @test reduce(&,map(≈,partition(∂2∂u2R_matrix_analytical),partition(∂2R∂u2_mat)))

  # ∂/∂p (∂R/∂u * λ) * ṗ
  ∂2R∂u∂p_analytical(uh,λh,ph) = ∫( dp_* ∇(dv) ⋅ ∇(λh)  )dΩ
  ∂2R∂u∂p_matrix_analytical = assemble_matrix(∂2R∂u∂p_analytical(uh,λh,ph),V_p,U)
  @test reduce(&,map(≈,partition(∂2R∂u∂p_matrix_analytical),partition(∂2R∂u∂p_mat)))
  # ∂²R / ∂p² * ṗ * λ
  ∂2R∂p2_analytical(uh,λh) = ∫( 0*dp⋅dp_ )dΩ
  ∂2R∂p2_matrix_analytical = assemble_matrix(∂2R∂p2_analytical(uh,λh),V_p,V_p)
  @test reduce(&,map(≈,partition(∂2R∂p2_matrix_analytical),partition(∂2R∂p2_matrix_analytical)))

  # ∂/∂u (∂R/∂p * λ) * ṗ
  ∂2R∂p∂u_analytical(uh,λh,ph) = ∫( dp * ∇(du) ⋅ ∇(λh) )dΩ
  ∂2R∂p∂u_matrix_analytical = assemble_matrix(∂2R∂p∂u_analytical(uh,λh,ph),U,V_p)
  @test reduce(&,map(≈,partition(∂2R∂p∂u_matrix_analytical),partition(∂2R∂p∂u_mat)))

  ######################
  # Self-adjoint tests #
  ######################

  f2(x) = 1.0
  res2(u,v,p) = ∫( p*∇(u)⋅∇(v)-f2*v )dΩ
  J2(u,p) = ∫( f2*u + 0*p )dΩ # p term to avoid dual error - should be fixed in the future
  state_map = NonlinearFEStateMap(res2,U,V,V_p,diff_order=2)
  objective = GridapTopOpt.StateParamMap(J2,state_map,diff_order=2)
  ph = interpolate(x->rand(), V_p)
  ṗh = interpolate(x->rand(), V_p)
  u = state_map(get_free_dof_values(ph))

  uh = FEFunction(U,u)
  Zygote.gradient(p->objective(state_map(p),p),get_free_dof_values(ph)); # update λ
  λ = state_map.cache.adj_cache[3]

  @test u ≈ λ # the adjoint should equal the solution for a self-adjoint problem

  p = get_free_dof_values(ph)
  ṗ = get_free_dof_values(ṗh)
  λh = FEFunction(V,λ)
  T = ForwardDiff.Tag(()->(),typeof(p))
  pᵋ = GridapTopOpt._build_duals(T,p,ṗ);
  uᵋ = state_map(pᵋ)
  ForwardDiff.value.(uᵋ) ≈ u

  function _mapreduce_partials(pᵋ::PVector)
    v = map(Base.Fix2(GridapTopOpt._mapreduce_partials,nothing), local_views(pᵋ))
    pvec_ids = pᵋ.index_partition
    return PVector(v,pvec_ids)
  end

  u̇ = _mapreduce_partials(uᵋ)

  ∇f = p->GridapTopOpt.val_and_gradient(p->objective(state_map(p),p),p).grad[1]
  Hṗ_FOR = ForwardDiff.derivative(α -> ∇f(p + α*ṗ), 0)
  λ⁻ = state_map.cache.inc_adjoint_cache[1]

  @test λ⁻ ≈ u̇ # the incremental adjoint should equal the incremental state for a self-adjoint problem

  Hṗ_fd = fd_hvp(p->objective(state_map(p),p),p,ṗ)
  Hṗ = Hvp(p->objective(state_map(p),p),p,ṗ)
  @test Hṗ_fd ≈ Hṗ

  # ########################################################
  # # Unit and integration tests for the pushforward rules #
  # ########################################################

  J3(u,p) = ∫( f*(1.0(sin∘(2π*u))+1)*(1.0(cos∘(2π*p))+1)*p)dΩ
  objective = GridapTopOpt.StateParamMap(J3,state_map,diff_order=2)

  # !! Nonlinear state map tests
  res3(u,v,p) = ∫( (u+1)*(p)*∇(u)⋅∇(v) - f*v )dΩ
  state_map = NonlinearFEStateMap(res3,U,V,V_p,diff_order=2)
  Zygote.gradient(p->objective(state_map(p),p),p); # update λ and u

  # entire incremental map (including the adjoint part) (ṗ->dṗ)
  p_to_j1(p) = objective((state_map(p)),p)
  Hṗ_fd = fd_hvp(p_to_j1,p,ṗ)
  Hṗ = Hvp(p_to_j1,p,ṗ)
  @show maximum(abs,Hṗ-Hṗ_fd)/maximum(abs,Hṗ)
  @test Hṗ_fd ≈ Hṗ

  # !! Affine state map Tests
  a(u,v,p) = ∫( p*(p+1)*∇(u)⋅∇(v) )dΩ
  l(v,p) = ∫( f*v )dΩ
  state_map = AffineFEStateMap(a,l,U,V,V_p,diff_order=2)
  Zygote.gradient(p->objective(state_map(p),p),p); # update λ and u

  # entire incremental map (including the adjoint part) (ṗ->dṗ)
  p_to_j2(p) = objective((state_map(p)),p)
  Hṗ_fd = fd_hvp(p_to_j2,p,ṗ)
  Hṗ = Hvp(p_to_j2,p,ṗ)
  @test Hṗ_fd ≈ Hṗ

  # ! only StateParamMap
  model = CartesianDiscreteModel(ranks,mesh_partition,(0,1,0,1), (3,3))
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  reffe = ReferenceFE(lagrangian, Float64, 1)
  K = TestFESpace(model, reffe)
  κ0h = interpolate(x->x[1], K)
  κ = get_free_dof_values(κ0h)
  vh = interpolate(0.5, K)
  v = get_free_dof_values(vh)
  assem = SparseMatrixAssembler(K,K)
  l2_norm = StateParamMap((u, κ) -> ∫(u + κ*κ)dΩ,K,K,assem,assem;diff_order=2) # (!!)
  u_obs = interpolate(x -> sin(2π*x[1]), K) |> get_free_dof_values
  function κ_to_J1(κ)
    sqrt(l2_norm(u_obs, κ))
  end
  val, grad = val_and_gradient(κ_to_J1, κ);
  # Hessian-vector product
  Hv = Hvp(κ_to_J1, κ, v)

  # Test full case
  Hv_fd = fd_hvp(κ_to_J1, κ, v)
  @show maximum(abs,Hv-Hv_fd)/maximum(abs,Hv)
  @test Hv ≈ Hv_fd

  # ! Doc example
  f3(x) = x[2]
  g(x) = x[1]

  model = CartesianDiscreteModel(ranks,mesh_partition,(0,1,0,1), (3,3))
  Ω = Triangulation(model)
  dΩ = Measure(Ω, 2)
  reffe = ReferenceFE(lagrangian, Float64, 1)
  K = TestFESpace(model, reffe)
  κ0h = interpolate(x->x[1], K)
  κ = get_free_dof_values(κ0h)
  vh = interpolate(0.5, K)
  v = get_free_dof_values(vh)
  K = TestFESpace(model, reffe)
  V = TestFESpace(model, reffe; dirichlet_tags="boundary")
  U = TrialFESpace(V,g)
  a3(u, v, κ) = ∫(κ * ∇(v) ⋅ ∇(u))dΩ
  b3(v, κ) = ∫(v*f3)dΩ
  κ_to_u = AffineFEStateMap(a3,b3,U,V,K;diff_order=2)
  l2_norm = StateParamMap((u, κ) -> ∫(u ⋅ u + 0κ)dΩ,κ_to_u;diff_order=2) # (!!)
  u_obs = interpolate(x -> sin(2π*x[1]), V) |> get_free_dof_values
  function κ_to_J2(κ)
    u = κ_to_u(κ)
    sqrt(l2_norm(u-u_obs, κ))
  end
  val, grad = val_and_gradient(κ_to_J2, κ);
  # Hessian-vector product
  Hv = Hvp(κ_to_J2, κ, v)

  # !! Tests
  Hv_fd = fd_hvp(κ_to_J2, κ, v)
  @show maximum(abs,Hv-Hv_fd)/maximum(abs,Hv)
  @test Hv ≈ Hv_fd
  nothing
end

with_mpi() do distribute
  main(distribute,(2,2))
end

# function driver(model)
#   f(x) = x[2]
#   g(x) = x[1]

#   Ω = Triangulation(model)
#   dΩ = Measure(Ω, 2)
#   reffe = ReferenceFE(lagrangian, Float64, 1)
#   K = TestFESpace(model, reffe)
#   V = TestFESpace(model, reffe; dirichlet_tags="boundary")
#   U = TrialFESpace(V,g)
#   a(u, v, κ) = ∫(κ * ∇(v) ⋅ ∇(u))dΩ
#   b(v, κ) = ∫(v*f)dΩ
#   κ_to_u = AffineFEStateMap(a,b,U,V,K;diff_order=2)
#   l2_norm = StateParamMap((u, κ) -> ∫(u ⋅ u + 0κ)dΩ,κ_to_u;diff_order=2) # (!!)
#   u_obs = interpolate(x -> sin(2π*x[1]), V) |> get_free_dof_values
#   function J(κ)
#     u = κ_to_u(κ)
#     sqrt(l2_norm(u-u_obs, κ))
#   end
#   κ0h = interpolate(1.0, K)
#   val, grad = val_and_gradient(J, get_free_dof_values(κ0h));
#   # Hessian-vector product
#   vh = interpolate(0.5, K);
#   Hv = Hvp(J, get_free_dof_values(κ0h),get_free_dof_values(vh));

#   return grad[1], Hv, K
# end

# model_serial = CartesianDiscreteModel((0,1,0,1),(8,8));
# dF_serial,ddF_serial,V_deriv_serial = driver(model_serial);

# model = GridapTopOpt.ordered_distributed_model_from_serial_model(ranks,model_serial);
# dF,ddF,V_deriv = driver(model);

# norm(dF_serial)
# norm(dF)
# norm(ddF_serial)
# norm(ddF)

# dFh = FEFunction(V_deriv,dF)
# dFh_serial = FEFunction(V_deriv_serial,dF_serial)
# grad_test = GridapTopOpt.test_serial_and_distributed_fields(dFh,V_deriv,dFh_serial,V_deriv_serial)
# map(grad_test) do grad_test
#   @test grad_test
#   nothing
# end

# ddFh = FEFunction(V_deriv,ddF)
# ddFh_serial = FEFunction(V_deriv_serial,ddF_serial)
# hes_test = GridapTopOpt.test_serial_and_distributed_fields(ddFh,V_deriv,ddFh_serial,V_deriv_serial)
# map(hes_test) do hes_test
#   @test hes_test
#   nothing
# end
# # end