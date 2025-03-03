using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapPETSc

using GridapDistributed,PartitionedArrays,GridapPETSc

function main(ranks,mesh_partition,n)
  path = "./results/Testing PETSc with empty parts/"
  i_am_main(ranks) && mkpath(path)

  domain = (0,1,0,1)
  cell_partition = (n,n)
  base_model = UnstructuredDiscreteModel(CartesianDiscreteModel(ranks,mesh_partition,domain,cell_partition))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)
  writevtk(model,path*"model")

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)

  lsf(x) = sqrt(x[1]^2+(x[2]-0.5)^2)-0.3
  φh = interpolate(lsf,V_φ)
  writevtk(get_triangulation(φh),path*"initial_lsf",cellfields=["φh"=>φh])

  # Cut
  order = 1
  degree = 2*(order+1)

  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)

  Ω = Triangulation(cutgeo,PHYSICAL)
  Ω_act = Triangulation(cutgeo,ACTIVE)
  dΩ = Measure(Ω,degree)

  Γ  = EmbeddedBoundary(cutgeo)
  dΓ  = Measure(Γ,degree)

  Γg = GhostSkeleton(cutgeo)
  n_Γg = get_normal_vector(Γg)
  dΓg = Measure(Γg,degree)

  writevtk(Ω,path*"omega")

  # Weak form

  γg = 0.1
  h = 1/n
  g = 1
  a(u,v) =
    ∫( ∇(v)⋅∇(u) )dΩ +
    ∫( (γg*h)*(jump(n_Γg⋅∇(v))⋅jump(n_Γg⋅∇(u))) )dΓg

  l(v) = ∫( g*v )dΓ

  reffe = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(Ω_act,reffe,conformity=:H1,dirichlet_tags=[5,])
  U = TrialFESpace(V)

  op = AffineFEOperator(a,l,U,V)

  uh = solve(op)
  writevtk(Ω,path*"omega_sol",cellfields=["uh"=>uh])
end

with_debug() do distribute
  mesh_partition = (2,2)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  main(ranks,mesh_partition,30)
end
