using Gridap, GridapDistributed, PartitionedArrays

using GridapEmbedded, GridapEmbedded.Distributed, GridapEmbedded.LevelSetCutters
using GridapEmbedded.Distributed: DistributedDiscreteGeometry

function main(distribute,mesh_partition;name)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  model = CartesianDiscreteModel(ranks,mesh_partition,(0,1,0,1),(100,100));
  V_φ = TestFESpace(model,ReferenceFE(lagrangian,Float64,1))
  φh = interpolate(x->-cos(4π*x[1])*cos(4*pi*x[2])/4-0.2/4,V_φ)

  geo = DistributedDiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)
  Ω = Triangulation(cutgeo,PHYSICAL)
  writevtk(Ω,"./results/discrete_geo_$(name)_2d",cellfields=["φh"=>φh])
end

# with_debug() do distribute
#   mesh_partition = (2,3);
#   main(distribute,mesh_partition;name="DebugMPI")
# end

with_mpi() do distribute
  mesh_partition = (2,3);
  main(distribute,mesh_partition;name="MPI")
end