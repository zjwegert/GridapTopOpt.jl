module EmbeddedCollectionsTests
using Test

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

function main()
  order = 1
  model = generate_model(2,40)

  reffe = ReferenceFE(lagrangian,Float64,order)
  V_φ = TestFESpace(model,reffe)

  φ(r) = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-r,V_φ) # Circle

  φ0 = φ(0.2)
  Ωs = EmbeddedCollection(model,φ0) do cutgeo
    Ω = Triangulation(cutgeo,PHYSICAL_IN)
    Γ = EmbeddedBoundary(cutgeo)
    (; 
      :Ω  => Ω,
      :Γ  => Γ,
      :dΩ => Measure(Ω,2*order),
      :dΓ => Measure(Γ,2*order)
    )
  end

  area(Ωs) = sum(∫(1.0)*Ωs[:dΩ])
  contour(Ωs) = sum(∫(1.0)*Ωs[:dΓ])

  for r in 0.2:0.1:0.5
    update_collection!(Ωs,φ(r))
    A = area(Ωs)
    C = contour(Ωs)
    println(" >> Radius: $r")
    println(" >> Area: $(A) (expected: $(π*r^2))")
    println(" >> Contour: $(C) (expected: $(2π*r))")
    @test abs(A - π*r^2) < 1e-3
    @test abs(C - 2π*r) < 1e-3
    println("---------------------------------")
  end
end

main()

end