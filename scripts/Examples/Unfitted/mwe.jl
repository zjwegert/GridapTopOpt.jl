using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt

n = 50
base_model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1),(n,n)))
ref_model = refine(base_model, refinement_method = "barycentric")
ref_model = refine(ref_model) ### <-------- This breaks cut_conforming
model = get_model(ref_model)
Ω = Triangulation(model)

reffe_scalar = ReferenceFE(lagrangian,Float64,1)
V_φ = TestFESpace(model,reffe_scalar)

f(x,y0) = abs(x[2]-y0) - 0.05
g(x,x0,y0,r) = sqrt((x[1]-x0)^2 + (x[2]-y0)^2) - r
q(x) = min(f(x,0.75),f(x,0.25),
  g(x,0.15,0.5,0.09),
  g(x,0.5,0.6,0.2),
  g(x,0.85,0.5,0.09),
  g(x,0.5,0.15,0.05))
φh = interpolate(q,V_φ)
GridapTopOpt.correct_ls!(φh)
writevtk(Ω,"results/Omega_act_φh",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh))])

# Check LS
GridapTopOpt.correct_ls!(φh)
φ_cell_values = get_cell_dof_values(φh)
get_isolated_volumes_mask_polytopal(model,φ_cell_values,[1])