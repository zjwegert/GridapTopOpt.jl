module MultiStageStabilisedReinitTest
using Test

using GridapTopOpt
using Gridap, Gridap.Geometry, Gridap.Adaptivity
using GridapEmbedded, GridapEmbedded.LevelSetCutters

order = 1
n = 201
_model = CartesianDiscreteModel((0,1,0,1),(n,n))
cd = Gridap.Geometry.get_cartesian_descriptor(_model)
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = ref_model.model
h = maximum(cd.sizes)

Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe_scalar)

φh = interpolate(x->-(x[1]-0.5)^2-(x[2]-0.5)^2+0.25^2,V_φ)
φh0 = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25,V_φ)

Ωs = EmbeddedCollection(model,φh) do cutgeo,_
  Γ = EmbeddedBoundary(cutgeo)
  (;
    :Γ => Γ,
    :dΓ => Measure(Γ,2*order)
  )
end

ls_evo = CutFEMEvolve(V_φ,Ωs,dΩ,h)
reinit1 = StabilisedReinit(V_φ,Ω,dΩ_act,hₕ;stabilisation_method=ArtificialViscosity(2.0))
reinit2 = StabilisedReinit(V_φ,Ω,dΩ_act,hₕ;stabilisation_method=GridapTopOpt.InteriorPenalty(V_φ,γg=1.0))
ls_reinit = GridapTopOpt.MultiStageStabilisedReinit([reinit1,reinit2])
evo = UnfittedFEEvolution(ls_evo,ls_reinit)
reinit!(evo,φh);

L2error(u) = sqrt(sum(∫(u ⋅ u)dΩ))
# Check |∇(φh)|
abs(L2error(norm ∘ ∇(φh))-1) < 1e-4

# Check φh error
@test L2error(φh-φh0) < 1e-4

# Check facet coords
geo = DiscreteGeometry(φh,model)
geo0 = DiscreteGeometry(φh0,model)
cutgeo = cut(model,geo)
cutgeo0 = cut(model,geo0)
Γ = EmbeddedBoundary(cutgeo)
Γ0 = EmbeddedBoundary(cutgeo0)
@test norm(Γ.subfacets.point_to_coords - Γ0.subfacets.point_to_coords,Inf) < 1e-4


# writevtk(
#   Ω,"results/test_evolve",
#   cellfields=[
#   "φh0"=>φh0,"|∇φh0|"=>norm ∘ ∇(φh0),
#   "φh_reinit"=>φh_reinit,"|∇φh_reinit|"=>norm ∘ ∇(φh_reinit)
#   ]
# )

end