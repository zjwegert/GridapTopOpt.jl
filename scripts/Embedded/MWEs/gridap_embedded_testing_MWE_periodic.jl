using Pkg; Pkg.activate()

using Gridap
using GridapEmbedded,GridapEmbedded.LevelSetCutters
import Gridap.Geometry: get_node_coordinates,collect1d

model = CartesianDiscreteModel((0,1,0,1),(100,100),isperiodic=(true,true));
V_φ = TestFESpace(model,ReferenceFE(lagrangian,Float64,1))

## Discrete
φh = interpolate(x->-cos(4π*(x[1]))*cos(4*pi*x[2])/4-0.2/4,V_φ)
point_to_coords = collect1d(get_node_coordinates(model))
geo = DiscreteGeometry(φh(point_to_coords),point_to_coords,name="")
cutgeo = cut(model,geo)

Ω = Triangulation(cutgeo,PHYSICAL)
Ω_act = Triangulation(cutgeo,ACTIVE)

# writevtk(Ω,"./results/discrete_geo_periodic",cellfields=["φh"=>φh])

V_φ_test = TestFESpace(Ω_act,ReferenceFE(lagrangian,Float64,1))
φh_test = interpolate(x->sqrt(x[1]^2+x[2]^2)-0.5,V_φ_test)
writevtk(Ω,"./results/discrete_geo_periodic",cellfields=["φh"=>φh_test])

## Analytic
analytic_geo=AnalyticalGeometry(x->-cos(4π*(x[1]))*cos(4*pi*x[2])/4-0.2/4)
cutgeo_analytic = cut(model,analytic_geo)
Ω_analytic = Triangulation(cutgeo_analytic,PHYSICAL)
Ω_analytic_act = Triangulation(cutgeo_analytic,ACTIVE)
# writevtk(Ω_analytic,"./results/analytic_geo_periodic",cellfields=["φh"=>φh])

V_φ_test = TestFESpace(Ω_analytic_act,ReferenceFE(lagrangian,Float64,1))
φh_test = interpolate(x->sqrt(x[1]^2+x[2]^2)-0.5,V_φ_test)
writevtk(Ω_analytic,"./results/analytic_geo_periodic",cellfields=["φh"=>φh_test])