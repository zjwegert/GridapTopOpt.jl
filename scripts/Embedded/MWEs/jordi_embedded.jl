
using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData
import Gridap.Geometry: get_node_coordinates, collect1d

include("../differentiable_trians.jl")

model = CartesianDiscreteModel((0,1,0,1),(100,100))
Ω = Triangulation(model)
dΩ = Measure(Ω,2)

reffe_scalar = ReferenceFE(lagrangian,Float64,1)
V = TestFESpace(model,reffe_scalar)
U = TrialFESpace(V)
V_φ = TestFESpace(model,reffe_scalar)

φh = interpolate(x->-cos(4π*x[1])*cos(4*pi*x[2])/4-0.2/4,V_φ)

Ωin = DifferentiableTriangulation(φh) do φh
  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)
  return Triangulation(cutgeo,PHYSICAL_IN)
end

dΩin = Measure(Ωin,2)

j(φ) = ∫(φ)dΩin
gradient(j,φh)

# To-Do: 
#   1. Add updateability condition
#   2. Add caches so we don't have to recompute everything every time
#   3. Figure out a way to share the cache between different triangulations created from the 
#      same cut geometry, i.e PHYS_IN/PHYS_OUT/Boundary
#   4. Anything else?

