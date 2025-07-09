module MultiStageStabilisedReinitTestMPI
using Test

using GridapTopOpt
using Gridap, Gridap.Geometry, Gridap.Adaptivity
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers

using GridapDistributed, PartitionedArrays

function main(distribute,mesh_partition)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  order = 1
  n = 101
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

  φh = interpolate(x->-(x[1]-0.5)^2-(x[2]-0.5)^2+0.25^2,V_φ)
  φh0 = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25,V_φ)

  Ωs = EmbeddedCollection(model,φh) do cutgeo,_,_
    Γ = EmbeddedBoundary(cutgeo)
    (;
      :Γ => Γ,
      :dΓ => Measure(Γ,2*order)
    )
  end

  ls_evo = CutFEMEvolver(V_φ,Ωs,dΩ,h)
  reinit1 = StabilisedReinitialiser(V_φ,Ωs,dΩ,h;
    stabilisation_method=ArtificialViscosity(1.5h),
    nls = GridapSolvers.NewtonSolver(LUSolver();maxiter=50,rtol=1.e-14,verbose=i_am_main(ranks)))
  reinit2 = StabilisedReinitialiser(V_φ,Ωs,dΩ,h;
    stabilisation_method=InteriorPenalty(V_φ),
    nls = GridapSolvers.NewtonSolver(LUSolver();maxiter=50,rtol=1.e-14,verbose=i_am_main(ranks)))
  ls_reinit = GridapTopOpt.MultiStageStabilisedReinitialiser([reinit1,reinit2])
  evo = LevelSetEvolution(ls_evo,ls_reinit)
  reinit!(evo,φh);

  L2error(u) = sqrt(sum(∫(u ⋅ u)dΩ))
  # Check |∇(φh)|
  @test abs(L2error(norm ∘ ∇(φh))-1) < 1e-4

  # Check φh error
  @test L2error(φh-φh0) < 1e-4

  # Check facet coords
  geo = DiscreteGeometry(φh,model)
  geo0 = DiscreteGeometry(φh0,model)
  cutgeo = cut(model,geo)
  cutgeo0 = cut(model,geo0)
  Γ = EmbeddedBoundary(cutgeo)
  Γ0 = EmbeddedBoundary(cutgeo0)

  map(local_views(Γ),local_views(Γ0)) do Γ,Γ0
    @test norm(Γ.parent.subfacets.point_to_coords - Γ0.parent.subfacets.point_to_coords,Inf) < 1e-4
  end
end

with_mpi() do distribute
  main(distribute,(2,2))
end

end