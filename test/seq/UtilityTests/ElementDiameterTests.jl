module ElementDiameterTests
using Test
using GridapTopOpt
using Gridap, Gridap.Geometry, Gridap.Adaptivity

function generate_model(D,n)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  base_model = UnstructuredDiscreteModel((CartesianDiscreteModel(domain,cell_partition)))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = ref_model.model
  return model
end

model = generate_model(2,2)
@test all(isequal(0.5),get_element_diameters(model))

model = generate_model(3,2)
@test all(isequal(0.5),get_element_diameters(model))

end