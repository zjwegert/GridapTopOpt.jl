module ElementDiameterTests
using Test
using GridapTopOpt
using GridapTopOpt: _get_tri_circumdiameter, _get_tet_circumdiameter

function generate_model(D,n)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  base_model = UnstructuredDiscreteModel((CartesianDiscreteModel(domain,cell_partition)))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = ref_model.model
  return model
end

tri_coords = [
  VectorValue(0.2948854413908402,0.2331928283531077),
  VectorValue(0.2886157430114769,0.2306953723644822),
  VectorValue(0.2908695476768671,0.2387503094465989)
];

@test abs(_get_tri_circumdiameter(tri_coords)/2 - 0.00431269)/0.0043126 < 1e-5

tet_coords = [
  VectorValue(0.2,0.2,0.1),
  VectorValue(1,0.2,0.1),
  VectorValue(2.3,1,0.1),
  VectorValue(1.2,3/2,1)
];

@test abs(_get_tet_circumdiameter(tet_coords)/2 - 2.64105)/2.64105 < 1e-5

model = generate_model(2,1)
@test all(isone,get_element_diameters(model))

get_cell_coordinates(model)

model = generate_model(3,1)
@test all(abs.(get_element_diameters(model)/2 .- 0.559017) .< 1e-8)

end