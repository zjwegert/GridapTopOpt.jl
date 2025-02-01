# module AutodiffTests

using Test
using Gridap, Gridap.Algebra
using GridapDistributed
using PartitionedArrays

function main_sf(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  domain = (0,4,0,4)
  cells = (4,4)
  model = CartesianDiscreteModel(ranks,parts,domain,cells)

  u((x,y)) = (x+y)^k
  σ(∇u) = (1.0+∇u⋅∇u)*∇u
  dσ(∇du,∇u) = (2*∇u⋅∇du)*∇u + (1.0+∇u⋅∇u)*∇du
  f(x) = -divergence(y->σ(∇(u,y)),x)

  k = 1
  reffe = ReferenceFE(lagrangian,Float64,k)
  V = TestFESpace(model,reffe,dirichlet_tags="boundary")
  U = TrialFESpace(u,V)

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*k)
  r(u,v) = ∫( ∇(v)⋅(σ∘∇(u)) - v*f )dΩ
  j(u,du,v) = ∫( ∇(v)⋅(dσ∘(∇(du),∇(u))) )dΩ

  op = FEOperator(r,j,U,V)
  op_AD = FEOperator(r,U,V)

  uh = interpolate(1.0,U)
  A = jacobian(op,uh)
  A_AD = jacobian(op_AD,uh)
  @test reduce(&,map(≈,partition(A),partition(A_AD)))

  g(v) = ∫(0.5*v⋅v)dΩ
  dg(v) = ∫(uh⋅v)dΩ
  b = assemble_vector(dg,U)
  b_AD = assemble_vector(gradient(g,uh),U)
  @test b ≈ b_AD
end

function main_mf(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),(4,4))

  k = 2
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},k)
  reffe_p = ReferenceFE(lagrangian,Float64,k-1;space=:P)

  u(x) = VectorValue(x[2],-x[1])
  V = TestFESpace(model,reffe_u,dirichlet_tags="boundary")
  U = TrialFESpace(V,u)
  Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean)

  X = MultiFieldFESpace([U,Q])
  Y = MultiFieldFESpace([V,Q])

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*(k+1))

  ν = 1.0
  f = VectorValue(0.0,0.0)

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
  dc(u,du,dv) = ∫(dv⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

  biform((du,dp),(dv,dq)) = ∫(ν*∇(dv)⊙∇(du) - (∇⋅dv)*dp - (∇⋅du)*dq)dΩ
  liform((dv,dq)) = ∫(dv⋅f)dΩ

  r((u,p),(v,q)) = biform((u,p),(v,q)) + c(u,v) - liform((v,q))
  j((u,p),(du,dp),(dv,dq)) = biform((du,dp),(dv,dq)) + dc(u,du,dv)

  op = FEOperator(r,j,X,Y)
  op_AD = FEOperator(r,X,Y)

  xh = interpolate([VectorValue(1.0,1.0),1.0],X)
  uh, ph = xh
  A = jacobian(op,xh)
  A_AD = jacobian(op_AD,xh)
  @test reduce(&,map(≈,partition(A),partition(A_AD)))

  g((v,q)) = ∫(0.5*v⋅v + 0.5*q*q)dΩ
  dg((v,q)) = ∫(uh⋅v + ph*q)dΩ
  b = assemble_vector(dg,X)
  b_AD = assemble_vector(gradient(g,xh),X)
  @test b ≈ b_AD
end

function main_mf_Ω(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),(4,4))

  k = 2
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},k)
  reffe_p = ReferenceFE(lagrangian,Float64,k-1;space=:P)

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*(k+1))

  u(x) = VectorValue(x[2],-x[1])
  V = TestFESpace(Ω,reffe_u,dirichlet_tags="boundary")            # Taking Ω instead of model
  U = TrialFESpace(V,u)
  Q = TestFESpace(Ω,reffe_p;conformity=:L2,constraint=:zeromean)  # Taking Ω instead of model

  X = MultiFieldFESpace([U,Q])
  Y = MultiFieldFESpace([V,Q])

  ν = 1.0
  f = VectorValue(0.0,0.0)

  conv(u,∇u) = (∇u')⋅u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v) = ∫(v⊙(conv∘(u,∇(u))))dΩ
  dc(u,du,dv) = ∫(dv⊙(dconv∘(du,∇(du),u,∇(u))))dΩ

  biform((du,dp),(dv,dq)) = ∫(ν*∇(dv)⊙∇(du) - (∇⋅dv)*dp - (∇⋅du)*dq)dΩ
  liform((dv,dq)) = ∫(dv⋅f)dΩ

  r((u,p),(v,q)) = biform((u,p),(v,q)) + c(u,v) - liform((v,q))
  j((u,p),(du,dp),(dv,dq)) = biform((du,dp),(dv,dq)) + dc(u,du,dv)

  op = FEOperator(r,j,X,Y)
  op_AD = FEOperator(r,X,Y)

  xh = interpolate([VectorValue(1.0,1.0),1.0],X)
  uh, ph = xh
  A = jacobian(op,xh)
  A_AD = jacobian(op_AD,xh)
  @test reduce(&,map(≈,partition(A),partition(A_AD)))

  g((v,q)) = ∫(0.5*v⋅v + 0.5*q*q)dΩ
  dg((v,q)) = ∫(uh⋅v + ph*q)dΩ
  b = assemble_vector(dg,X)
  b_AD = assemble_vector(gradient(g,xh),X)
  @test b ≈ b_AD
end

function main(distribute,parts)
  # main_sf(distribute,parts)
  # main_mf(distribute,parts)
  main_mf_Ω(distribute,parts)
end

with_debug() do distribute
  parts = (2,2)
  ranks = distribute(LinearIndices((prod(parts),)))
  main(distribute,parts)
end

# end

# using Gridap, GridapDistributed, PartitionedArrays
# using Test

# mesh_partition = (2,2);
# ranks = with_debug() do distribute
#   distribute(LinearIndices((prod(mesh_partition),)))
# end

# _model = CartesianDiscreteModel(ranks,mesh_partition,(0,1,0,1),(10,10))
# base_model = UnstructuredDiscreteModel(_model)
# ref_model = refine(base_model, refinement_method = "barycentric")
# model = Adaptivity.get_model(ref_model)

# Ω = Triangulation(model)

# reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},1,space=:P)
# reffe_p = ReferenceFE(lagrangian,Float64,1,space=:P)

# V = TestFESpace(Ω,reffe_u,conformity=:H1)
# Q = TestFESpace(Ω,reffe_p,conformity=:H1)

# VQ = MultiFieldFESpace([V,Q])

# #
# uh = zero(VQ)
# get_triangulation(VQ) # Produces check failed error

# #
# @test get_triangulation(V) === get_triangulation(Q)
# t = map((u,v)->u===v,local_views(get_triangulation(V)),local_views(get_triangulation(Q)))
# @test all(collect(t))
# @test Triangulation(model) === Triangulation(model)

# using Gridap.Helpers, Gridap.CellData

# function test_triangulation(Ω1,Ω2)
#   @assert typeof(Ω1.grid) == typeof(Ω2.grid)
#   t = map(fieldnames(typeof(Ω1.grid))) do field
#     getfield(Ω1.grid,field) == getfield(Ω2.grid,field)
#   end
#   all(t)
#   a = Ω1.model === Ω2.model
#   b = Ω1.tface_to_mface == Ω2.tface_to_mface
#   a && b && all(t)
# end

# # function test_triangulation(Ω1::GridapDistributed.DistributedTriangulation,Ω2::GridapDistributed.DistributedTriangulation)
# #   t = map((_Ω1,_Ω2)->test_triangulation(_Ω1,_Ω2),local_views(Ω1),local_views(Ω2))
# #   all(collect(t))
# # end

# function CellData.get_triangulation(f::MultiFieldCellField)
#   s1 = first(f.single_fields)
#   trian = get_triangulation(s1)
#   # @check all(map(i->trian===get_triangulation(i),f.single_fields))
#   @check all(map(i->test_triangulation(trian,get_triangulation(i)),f.single_fields))
#   trian
# end

# # function CellData.get_triangulation(a::GridapDistributed.DistributedMultiFieldCellField)
# #   trians = map(get_triangulation,a.field_fe_fun)
# #   # @check all(map(t -> t === first(trians), trians))
# #   @check all(map(t -> test_triangulation(t,first(trians)), trians))
# #   return first(trians)
# # end