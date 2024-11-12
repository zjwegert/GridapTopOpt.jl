using GridapTopOpt
using GridapSolvers.NonlinearSolvers

using Gridap
using Gridap.Adaptivity

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData
import Gridap.Geometry: get_node_coordinates, collect1d

include("../../differentiable_trians.jl")

using GridapTopOpt: LevelSetEvolution, instantiate_caches
using Gridap.Algebra: NonlinearSolver
using Gridap.ODEs: ODESolver
using GridapDistributed: DistributedFESpace
import Gridap: solve!

include("../../../../src/LevelSetEvolution/UnfittedEvolution/UnfittedEvolution.jl")

order = 1
n = 101
_model = CartesianDiscreteModel((0,1,0,1),(n,n))
cd = Gridap.Geometry.get_cartesian_descriptor(_model)
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = ref_model.model
# model = simplexify(_model)
h = maximum(cd.sizes)

Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe_scalar)

α = 2.0*h
a_hilb(p,q) = ∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
vel_ext = VelocityExtension(a_hilb,V_φ,V_φ)

φh = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25,V_φ) # <- Already SDF
# φh = interpolate(x->-(x[1]-0.5)^2-(x[2]-0.5)^2+0.25^2,V_φ)
# φh = interpolate(x->cos(4π*x[1])*cos(4π*x[2])-0.11,V_φ)

function compute_test_velh(φh)
  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)
  Ωin = Triangulation(cutgeo,PHYSICAL_IN).a
  diff_Ωin = DifferentiableTriangulation(Ωin)
  dΩin = Measure(diff_Ωin,2*order)
  j_in(φ) = ∫(1)dΩin
  dj_in = gradient(j_in,φh)
  dj_vec_in = assemble_vector(dj_in,V_φ)
  project!(vel_ext,dj_vec_in)
  vel = dj_vec_in/sqrt(dj_vec_in'*vel_ext.K*dj_vec_in)
  return FEFunction(V_φ,vel)
end

ls_evo = CutFEMEvolve(model,V_φ,dΩ,h)
ls_reinit = StabilisedReinit(model,V_φ,dΩ,h)
evo = UnfittedFEEvolution(ls_evo,ls_reinit)
# reinit!(evo,φh);

φ0 = copy(get_free_dof_values(φh))
φh0 = FEFunction(V_φ,φ0)

# velh = compute_test_velh(φh)
velh = interpolate(x->-1,V_φ)
evolve!(evo,φh,velh,0.1)

ls_evo = CutFEMEvolve(model,V_φ,dΩ,h)
ls_reinit = StabilisedReinit(model,V_φ,dΩ,h)
evo = UnfittedFEEvolution(ls_evo,ls_reinit)

velh = interpolate(x->1,V_φ)
evolve!(evo,φh,velh,0.1)

Δt = _compute_Δt(h,0.1,get_free_dof_values(velh))
φh_expected_contour = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25+evo.evolver.params.max_steps*Δt,V_φ)

writevtk(
  Ω,"results/test_evolve",
  cellfields=[
  "φh0"=>φh0,"|∇φh0|"=>norm ∘ ∇(φh0),  
  "φh_advect"=>φh,"|∇φh_advect|"=>norm ∘ ∇(φh),
  # "φh_expected_contour"=>φh_expected_contour
  ]
)