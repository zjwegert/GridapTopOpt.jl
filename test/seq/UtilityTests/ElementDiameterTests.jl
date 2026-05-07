module ElementDiameterTests
using Test
using GridapTopOpt
using Gridap, Gridap.Geometry, Gridap.Adaptivity

function generate_model(D,n,ref)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  base_model = CartesianDiscreteModel(domain,cell_partition)
  !ref && return base_model
  ref_model = refine(UnstructuredDiscreteModel(base_model), refinement_method = "barycentric")
  model = ref_model.model
  return model
end

model = generate_model(2,2,false)
@test all(isequal(0.5),get_element_diameters(model))

model = generate_model(2,2,true)
@test all(isequal(0.5),get_element_diameters(model))

model = generate_model(3,2,false)
@test all(isequal(0.5),get_element_diameters(model))

model = generate_model(3,2,true)
@test all(isequal(0.5),get_element_diameters(model))

# Test empty trian
model = generate_model(2,2,true)
Ω = Triangulation(model,Int[1])
dΩ = Measure(Ω,2)
hₕ = get_element_diameter_field(model)
∫(hₕ)dΩ

end