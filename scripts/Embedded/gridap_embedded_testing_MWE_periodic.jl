using Pkg; Pkg.activate()

using Gridap
using GridapEmbedded,GridapEmbedded.LevelSetCutters
import Gridap.Geometry: get_node_coordinates,collect1d

model = CartesianDiscreteModel((0,1,0,1),(100,100),isperiodic=(true,true));
V_φ = TestFESpace(model,ReferenceFE(lagrangian,Float64,1))

φh = interpolate(x->-cos(4π*x[1])*cos(4*pi*x[2])/4-0.2/4,V_φ)
analytic_geo=GridapEmbedded.disk(0.6;x0=Point(0.5,0.5))
cutgeo = cut(model,analytic_geo)

# point_to_coords = model.grid_topology.vertex_coordinates
# geo = DiscreteGeometry(get_free_dof_values(φh),point_to_coords,name="")
# cutgeo = cut(model,geo)

Ω = Triangulation(cutgeo,PHYSICAL)

writevtk(Ω,"./results/discrete_geo_periodic",cellfields=["φh"=>φh])
writevtk(Triangulation(model),"./results/discrete_geo_periodic_full",cellfields=["φh"=>φh])