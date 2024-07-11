
using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData
import Gridap.Geometry: get_node_coordinates, collect1d

include("../differentiable_trians.jl")

order = 1
model = CartesianDiscreteModel((0,1,0,1),(30,30))
Ω = Triangulation(model)
dΩ = Measure(Ω,2order)

reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe_scalar)
U = TrialFESpace(V)
V_φ = TestFESpace(model,reffe_scalar)

φh = interpolate(x->max(abs(x[1]-0.5),abs(x[2]-0.5))-0.1,V_φ)
# φh = interpolate(x->max(abs(x[1]-0.5),abs(x[2]-0.5))-0.11,V_φ)
# φh = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-0.3,V_φ)

Ωin = DifferentiableTriangulation(φh) do φh
  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)
  return Triangulation(cutgeo,PHYSICAL_IN)
end

dΩin = Measure(Ωin,2order)

j(φ) = ∫(1)dΩin
dj = gradient(j,φh)
dj_vec = assemble_vector(dj,V_φ)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)
Γ = EmbeddedBoundary(cutgeo)
dΓ = Measure(Γ,2order)
dj_expected(q) = ∫(-q)dΓ
dj_exp_vec = assemble_vector(dj_expected,V_φ)

writevtk(Ω,"results/test",cellfields=["φh"=>φh,"dj"=>FEFunction(V_φ,dj_vec),"dj_expected"=>FEFunction(V_φ,dj_exp_vec)])
# writevtk(Ωin.recipe(φh),"results/test_phys",cellfields=["φh"=>φh,"dj"=>FEFunction(V_φ,dj_vec)])

# To-Do:
#   1. Add updateability condition
#   2. Add caches so we don't have to recompute everything every time
#   3. Figure out a way to share the cache between different triangulations created from the
#      same cut geometry, i.e PHYS_IN/PHYS_OUT/Boundary
#   4. Anything else?

