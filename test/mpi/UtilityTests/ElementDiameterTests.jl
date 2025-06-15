module ElementDiameterTests
using Test
using GridapTopOpt
using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.Arrays

using GridapGmsh,GridapDistributed, PartitionedArrays

function main(distribute,mesh_partition,D,n)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  base_model = UnstructuredDiscreteModel(CartesianDiscreteModel(ranks,mesh_partition,domain,cell_partition))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)

  h = get_element_diameters(model)
  hₕ = get_element_diameter_field(model)

  _base_model = UnstructuredDiscreteModel(CartesianDiscreteModel(domain,cell_partition))
  _ref_model = refine(_base_model, refinement_method = "barycentric")
  model_serial = Adaptivity.get_model(_ref_model)
  h_serial = get_element_diameters(model_serial)
  @test test_array(h_serial,collect(h))
end

with_mpi() do distribute
  @test main(distribute,(2,2),2,4)
  @test main(distribute,(2,2),3,4)
end

end