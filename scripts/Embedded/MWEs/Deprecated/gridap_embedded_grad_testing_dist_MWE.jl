using Pkg; Pkg.activate()

using Gridap

using GridapDistributed, GridapPETSc, PartitionedArrays

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData
import Gridap.Geometry: get_node_coordinates, collect1d

include("../embedded_measures.jl")

parts = (3,3);
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(parts),)))
end

model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),(100,100));
Ω = Triangulation(model)
dΩ = Measure(Ω,2)

reffe_scalar = ReferenceFE(lagrangian,Float64,1)
V = TestFESpace(model,reffe_scalar)
U = TrialFESpace(V)
V_φ = TestFESpace(model,reffe_scalar)

# φh = interpolate(x->-cos(4π*x[1])*cos(4*pi*x[2])/4-0.2/4,V_φ)
φh = interpolate(x->-sqrt((x[1]-1/2)^2+(x[2]-1/2)^2)+0.2,V_φ)
embedded_meas = EmbeddedMeasureCache(φh,V_φ)
update_meas(args...) = update_embedded_measures!(embedded_meas,args...)
get_meas(args...) = get_embedded_measures(embedded_meas,args...)
update_embedded_measures!(embedded_meas,φh)

_j(φ,dΩ,dΩ1,dΩ2,dΓ) = ∫(φ)dΩ1 + ∫(φ)dΩ2 + ∫(φ)dΓ

j_iem = IntegrandWithEmbeddedMeasure(_j,(dΩ,),update_meas)
j_iem(φh)
gradient(j_iem,φh)

# Possible fix - doesn't work though!
# function alt_get_geo_params(ϕh::CellField,Vbg,gids)
#   Ωbg = get_triangulation(Vbg)
#   bgmodel = get_background_model(Ωbg)
#   own_model = remove_ghost_cells(bgmodel,gids)
#   geo1 = DiscreteGeometry(ϕh,own_model)
#   geo2 = DiscreteGeometry(-ϕh,own_model,name="")
#   get_geo_params(geo1,geo2,own_model)
# end

# gids = get_cell_gids(model)
# contribs = map(local_views(dΩ),local_views(φh),local_views(gids)) do dΩ,φh,gids
#   _f = u -> j_iem.F(u,dΩ,alt_get_geo_params(u,get_fe_space(φh),gids)[2]...)
#   return Gridap.Fields.gradient(_f,φh)
# end;