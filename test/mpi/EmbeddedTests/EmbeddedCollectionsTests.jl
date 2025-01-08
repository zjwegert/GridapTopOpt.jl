module EmbeddedCollectionsTestsMPI
using Test

using GridapTopOpt
using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity

using GridapDistributed, PartitionedArrays

function generate_model(D,n,ranks,mesh_partition)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  base_model = UnstructuredDiscreteModel(CartesianDiscreteModel(ranks,mesh_partition,domain,cell_partition))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)
  return model
end

function main(distribute,mesh_partition)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  order = 1
  model = generate_model(2,40,ranks,mesh_partition)

  reffe = ReferenceFE(lagrangian,Float64,order)
  V_φ = TestFESpace(model,reffe)

  φ(r) = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-r,V_φ) # Circle

  φ0 = φ(0.2)
  Ωs = EmbeddedCollection(model,φ0) do cutgeo,_
    Ω = Triangulation(cutgeo,PHYSICAL_IN)
    (;
      :Ω  => Ω,
      :dΩ => Measure(Ω,2*order),
    )
  end

  function r_Γ(cutgeo,cutgeo_facet)
    Γ = EmbeddedBoundary(cutgeo)
    (;
      :Γ  => Γ,
      :dΓ => Measure(Γ,2*order)
    )
  end
  add_recipe!(Ωs,r_Γ)

  area(Ωs) = sum(∫(1.0)*Ωs.dΩ)
  contour(Ωs) = sum(∫(1.0)*Ωs.dΓ)

  for r in 0.2:0.1:0.5
    update_collection!(Ωs,φ(r))
    A = area(Ωs)
    C = contour(Ωs)
    i_am_main(ranks) && println(" >> Radius: $r")
    i_am_main(ranks) && println(" >> Area: $(A) (expected: $(π*r^2))")
    i_am_main(ranks) && println(" >> Contour: $(C) (expected: $(2π*r))")
    @test abs(A - π*r^2) < 1e-3
    @test abs(C - 2π*r) < 1e-3
    println("---------------------------------")
  end
end

with_mpi() do distribute
  main(distribute,(2,2))
end

end