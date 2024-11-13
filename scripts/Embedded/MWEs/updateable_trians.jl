using GridapTopOpt
using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity

function generate_model(D,n)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  base_model = UnstructuredDiscreteModel((CartesianDiscreteModel(domain,cell_partition)))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = ref_model.model
  return model
end

φ(r) = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-r,V_φ) # Circle

struct EmbeddedTriangulationCollection
  func::Function
  trians::Dict{Symbol,Triangulation}
  bgmodel::DiscreteModel
  cutgeo::EmbeddedDiscretization
end

function EmbeddedTriangulationCollection(
  func::Function,bgmodel::DiscreteModel,φ0
)
  geo = DiscreteGeometry(φ0,bgmodel)
  cutgeo = cut(bgmodel,geo)
  trians = Dict(pairs(func(cutgeo)))
  return EmbeddedTriangulationCollection(func,trians,bgmodel,cutgeo)
end

function update!(c::EmbeddedTriangulationCollection,φh)
  geo = DiscreteGeometry(φh,c.bgmodel)
  cutgeo = cut(c.bgmodel,geo)
  trians = func(cutgeo)
  for (key,value) in pairs(trians)
    c.trians[key] = value
  end
  return c
end

function Base.getindex(c::EmbeddedTriangulationCollection,key)
  return c.trians[key]
end

order = 1
model = generate_model(2,10)

Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)

reffe = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe)

φ0 = φ(0.2)
Ωs = EmbeddedTriangulationCollection(model,φ0) do cutgeo
  (; 
    :Ωin  => Triangulation(cutgeo,PHYSICAL_IN),
    :Ωout => Triangulation(cutgeo,PHYSICAL_OUT),
    :Γ    => EmbeddedBoundary(cutgeo)
  )
end

area(Ωs) = sum(∫(1.0)*Measure(Ωs[:Ωin],2))
contour(Ωs) = sum(∫(1.0)*Measure(Ωs[:Γ],2))

for r in 0.2:0.1:0.5
  update!(Ωs,φ(r))
  println(" >> Radius: $r")
  println(" >> Area: $(area(Ωs)) (expected: $(π*r^2))")
  println(" >> Contour: $(contour(Ωs)) (expected: $(2π*r))")
end
