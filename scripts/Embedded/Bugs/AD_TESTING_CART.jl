using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt

using GridapDistributed,PartitionedArrays,GridapPETSc

function main(ranks,mesh_partition,n)
  path = "./results/Testing Deriv/"
  i_am_main(ranks) && mkpath(path)

  domain = (0,4,0,1,0,1)
  cell_partition = (4n,n,n)
  base_model = UnstructuredDiscreteModel(CartesianDiscreteModel(ranks,mesh_partition,domain,cell_partition))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)
  writevtk(model,path*"model")

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)

  lsf(x) = sqrt((x[1]-2.0)^2+(x[2]-0.35)^2+(x[3]-0.5)^2)-0.3 # Sphere
  φh = interpolate(lsf,V_φ)

  # Correct LSF
  _φ = get_free_dof_values(φh)
  map(local_views(_φ)) do φ
    idx = findall(isapprox(0.0;atol=eps()),φ)
    if !isempty(idx)
      i_am_main(ranks) && println("    Correcting level values at $(length(idx)) nodes")
    end
    φ[idx] .+= 100*eps(eltype(φ))
  end
  consistent!(_φ) |> wait

  # Cut
  order = 1
  degree = 2*(order+1)

  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)
  Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
  dΓ = Measure(Γ,degree)

  writevtk(get_triangulation(φh),path*"initial_lsf",cellfields=["φh"=>φh])
  writevtk(Γ,path*"Gamma")

  function J(φ)
    n = get_normal_vector(Γ)
    return ∫(n⋅∇(φ))dΓ
  end

  sum(J(φh))
  gradient(J,φh)
end

with_debug() do distribute
  mesh_partition = (2,4,3) # This fails
  # mesh_partition = (1,1,6) # This fails
  # mesh_partition = (2,4,2) # This works
  # mesh_partition = (2,2,2) # This works
  # mesh_partition = (1,1,5) # This works
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  main(ranks,mesh_partition,10)
end