using GridapTopOpt

using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.ODEs
import Gridap.Geometry: get_node_coordinates, collect1d

include("unfitted_evolution.jl")
include("../../differentiable_trians.jl")

order = 1
n = 101

_model = CartesianDiscreteModel((0,1,0,1),(n,n))
cd = Gridap.Geometry.get_cartesian_descriptor(_model)
h = maximum(cd.sizes)

model = simplexify(_model)
Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe_scalar)
Ut_φ = TransientTrialFESpace(V_φ,t -> (x->-1))

φh = interpolate(x->cos(2π*x[1])*cos(2π*x[2])-0.11,V_φ)

ls_evo = UnfittedFEEvolution(Ut_φ, V_φ, dΩ, h; NT = 5, c = 0.1)

reinit!(ls_evo,φh)
velh = interpolate(x->-1,V_φ)
evolve!(ls_evo,φh,get_free_dof_values(velh),0.01)
reinit!(ls_evo,φh)

writevtk(
  Ω,"results/test_evolve",
  cellfields=["φh"=>φh,"|∇φh|"=>norm ∘ ∇(φh)]
)