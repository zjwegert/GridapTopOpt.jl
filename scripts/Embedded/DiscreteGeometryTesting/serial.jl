using Gridap
using GridapEmbedded,GridapEmbedded.LevelSetCutters
import Gridap.Geometry: get_node_coordinates,collect1d

model = CartesianDiscreteModel((0,1,0,1),(100,100));
V_φ = TestFESpace(model,ReferenceFE(lagrangian,Float64,2))

φh = interpolate(x->-cos(4π*x[1])*cos(4*pi*x[2])/4-0.2/4,V_φ)
point_to_coords = collect1d(get_node_coordinates(model))
geo = DiscreteGeometry(get_free_dof_values(φh),point_to_coords,name="")
cutgeo = cut(model,geo)

Ω = Triangulation(cutgeo,PHYSICAL)

writevtk(Ω,"./results/discrete_geo_serial")