using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt

using GridapDistributed,PartitionedArrays,GridapPETSc

function test_mesh(model)
  grid_topology = Geometry.get_grid_topology(model)
  D = num_cell_dims(grid_topology)
  d = 0
  vertex_to_cells = Geometry.get_faces(grid_topology,d,D)
  bad_vertices = findall(i->i==0,map(length,vertex_to_cells))
  @assert isempty(bad_vertices) "Bad vertices detected: re-generate your mesh with a different resolution"
end

function main(ranks)
  path = "./results/Testing Deriv Gmsh/"
  i_am_main(ranks) && mkpath(path)

  model = GmshDiscreteModel(ranks,(@__DIR__)*"/../Meshes/mesh_3d_finer.msh")
  map(test_mesh,local_views(model))
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
  ncpus = 30
  ranks = distribute(LinearIndices((ncpus,)))
  petsc_options = "-ksp_converged_reason -ksp_error_if_not_converged true"
  GridapPETSc.with(;args=split(petsc_options)) do
    main(ranks)
  end
end