module CutFEMEvolveTest
using Test

using GridapTopOpt
using Gridap, Gridap.Geometry, Gridap.Adaptivity
using GridapEmbedded

order = 1
n = 50
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

φh = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25,V_φ)

Ωs = EmbeddedCollection(model,φh) do cutgeo,_,_
  Γ = EmbeddedBoundary(cutgeo)
  Γg = GhostSkeleton(cutgeo)
  (;
    :Γ  => Γ,
    :dΓ => Measure(Γ,2*order),
    :n_Γ  => get_normal_vector(Γ), # Note, need to recompute inside obj/constraints to compute derivs
    :Γg => Γg,
    :dΓg => Measure(Γg,2*order),
    :n_Γg => get_normal_vector(Γg)
  )
end

ls_evo = CutFEMEvolver(V_φ,Ωs,dΩ,h)
ls_reinit = StabilisedReinitialiser(V_φ,Ωs,dΩ,h)
evo = LevelSetEvolution(ls_evo,ls_reinit)

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

# writevtk(
#   Ω,"results/test_evolve",
#   cellfields=[
#   "φh0"=>φh0,"|∇φh0|"=>norm ∘ ∇(φh0),
#   "φh_advect"=>φh,"|∇φh_advect|"=>norm ∘ ∇(φh),
#   "φh_expected_lsf"=>φh_expected_lsf
#   ]
# )

end