module HeatReintialiserTest

using Gridap, Gridap.Adaptivity, Gridap.Geometry, Gridap.Helpers
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt, GridapSolvers
using Test

function main(model)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2)

  order = 1
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V_φ = TestFESpace(model,reffe_scalar)

  ## Level-set function
  f((x,y)) = (x-0.5)^2+(y-0.5)^2-0.35^2
  f_sdf((x,y)) = sqrt((x-0.5)^2+(y-0.5)^2)-0.35
  φh = interpolate(f,V_φ)
  φh0 = interpolate(f_sdf,V_φ)
  φh_old = FEFunction(V_φ,copy(get_free_dof_values(φh)))

  reinit_method = HeatReinitialiser(V_φ,model)
  reinit!(reinit_method,φh);

  L2error(u) = sqrt(sum(∫(u ⋅ u)dΩ))
  # Check |∇(φh)|
  @test abs(L2error(norm ∘ ∇(φh))-1) < 1e-3

  # Check φh error
  @test L2error(φh-φh0) < 1e-3

  geo = DiscreteGeometry(φh,model)
  geo0 = DiscreteGeometry(φh0,model)
  cutgeo = cut(model,geo)
  cutgeo0 = cut(model,geo0)
  Γ = EmbeddedBoundary(cutgeo)
  Γ0 = EmbeddedBoundary(cutgeo0)
  @test norm(Γ.subfacets.point_to_coords - Γ0.subfacets.point_to_coords,Inf) < 1e-3
end

_model = CartesianDiscreteModel((0,1,0,1),(101,101))
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = get_model(ref_model)

main(_model) # QUAD
main(model)  # TRI

end