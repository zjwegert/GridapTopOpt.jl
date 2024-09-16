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

α = 2.0*h
a_hilb(p,q) = ∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
vel_ext = VelocityExtension(a_hilb,V_φ,V_φ)

# φh = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25,V_φ) # <- Already SDF
φh = interpolate(x->cos(2π*x[1])*cos(2π*x[2])-0.11,V_φ)

function compute_test_velh(φh)
  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)
  Ωin = Triangulation(cutgeo,PHYSICAL_IN).a
  diff_Ωin = DifferentiableTriangulation(Ωin)
  dΩin = Measure(diff_Ωin,2*order)
  j_in(φ) = ∫(φ)dΩin
  dj_in = gradient(j_in,φh)
  dj_vec_in = assemble_vector(dj_in,V_φ)
  project!(vel_ext,dj_vec_in)
  vel = dj_vec_in/sqrt(dj_vec_in'*vel_ext.K*dj_vec_in)
  return FEFunction(V_φ,vel)
end

ls_evo = UnfittedFEEvolution(Ut_φ, V_φ, dΩ, h; NT = 5, c = 0.1,
  reinit_nls = NLSolver(ftol=1e-12, iterations = 50, show_trace=true))
reinit!(ls_evo,φh) # <- inital sdf

velh = compute_test_velh(φh)
# velh = interpolate(x->-1,V_φ)
evolve!(ls_evo,φh,get_free_dof_values(velh),0.01)
reinit!(ls_evo,φh)

# You can compare the zero-iso curve for advected circle via
# φh_expected_contour = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25+ls_evo.params.NT*0.01,V_φ)

writevtk(
  Ω,"results/test_evolve",
  cellfields=["φh"=>φh,"|∇φh|"=>norm ∘ ∇(φh),"velh"=>velh]#,"φh_expected_contour"=>φh_expected_contour]
)