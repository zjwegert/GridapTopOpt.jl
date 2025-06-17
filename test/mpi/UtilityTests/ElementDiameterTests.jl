module ElementDiameterTests
using Test
using GridapTopOpt
using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.Arrays

using GridapGmsh,GridapDistributed, PartitionedArrays

function main(distribute,mesh_partition)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  model = GmshDiscreteModel(ranks,(@__DIR__)*"/../../meshes/mesh.msh")

  h = get_element_diameters(model)
  hₕ = get_element_diameter_field(model)

  model_serial = GmshDiscreteModel((@__DIR__)*"/../../meshes/mesh.msh")
  h_serial = get_element_diameters(model_serial)
  @test test_array(h_serial,collect(h))
  true
end

with_mpi() do distribute
  @test main(distribute,(2,2))
end

end