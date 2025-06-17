module CutFEMEvolveTestMPI
using Test

using GridapTopOpt
using Gridap, Gridap.Geometry, Gridap.Adaptivity
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers

using GridapDistributed, PartitionedArrays

function main(distribute,mesh_partition)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  n = 50
  order = 1
  _model = CartesianDiscreteModel(ranks,mesh_partition,(0,1,0,1),(n,n))
  base_model = UnstructuredDiscreteModel(_model)
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)
  el_Δ = get_el_Δ(_model)
  h = maximum(el_Δ)

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V_φ = TestFESpace(model,reffe_scalar)

  φh = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25,V_φ)

  Ωs = EmbeddedCollection(model,φh) do cutgeo,_,_
    Γ = EmbeddedBoundary(cutgeo)
    (;
      :Γ  => Γ,
      :dΓ => Measure(Γ,2*order),
    )
  end

  ls_evo = CutFEMEvolve(V_φ,Ωs,dΩ,h)
  ls_reinit = StabilisedReinit(V_φ,Ωs,dΩ,h)
  evo = UnfittedFEEvolution(ls_evo,ls_reinit)

  φ0 = copy(get_free_dof_values(φh))
  φh0 = FEFunction(V_φ,φ0)

  velh = interpolate(x->-1,V_φ)
  evolve!(evo,φh,velh,0.1)
  Δt = 0.1*h
  φh_expected_lsf = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25+evo.evolver.params.max_steps*Δt,V_φ)

  # Test advected LSF mataches expected LSF
  L2error(u) = sqrt(sum(∫(u ⋅ u)dΩ))
  @test L2error(φh_expected_lsf-φh) < 1e-3

  # # Test advected LSF mataches original LSF when going backwards
  velh = interpolate(x->1,V_φ)
  evolve!(evo,φh,velh,0.1)
  @test L2error(φh0-φh) < 1e-4
end

with_mpi() do distribute
  main(distribute,(2,2))
end

end