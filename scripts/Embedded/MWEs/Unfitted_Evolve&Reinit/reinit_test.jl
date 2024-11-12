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
h = maximum(cd.sizes)

Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe_scalar)

φh = interpolate(x->-(x[1]-0.5)^2-(x[2]-0.5)^2+0.25^2,V_φ)

ls_evo = CutFEMEvolve(model,V_φ,dΩ,h)
ls_reinit = StabilisedReinit(model,V_φ,dΩ,h)

φ0 = copy(get_free_dof_values(φh))
φh0 = FEFunction(V_φ,φ0)

evo = UnfittedFEEvolution(ls_evo,ls_reinit)
reinit!(evo,φh);

L2error(u) = ∫(u*u)dΩ
@assert sum(L2error((norm ∘ ∇(φh_reinit)))) - 1 < 10^-4

writevtk(
  Ω,"results/test_evolve",
  cellfields=[
  "φh0"=>φh0,"|∇φh0|"=>norm ∘ ∇(φh0),  
  "φh_reinit"=>φh_reinit,"|∇φh_reinit|"=>norm ∘ ∇(φh_reinit)
  ]
)