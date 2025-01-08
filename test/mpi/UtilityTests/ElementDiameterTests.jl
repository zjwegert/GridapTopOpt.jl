module ElementDiameterTests
using Test
using GridapTopOpt
using GridapTopOpt: _get_tri_circumdiameter, _get_tet_circumdiameter
using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.Arrays

using GridapGmsh,GridapDistributed, PartitionedArrays

function main(distribute,mesh_partition)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  model = GmshDiscreteModel(ranks,"mesh.msh")

  h = get_element_diameters(model)
  hₕ = get_element_diameter_field(model)

  model_serial = GmshDiscreteModel("mesh.msh")
  h_serial = get_element_diameters(model_serial)
  @test test_array(h_serial,collect(h))
end

with_mpi() do distribute
  @test main(distribute,(2,2))
end

end