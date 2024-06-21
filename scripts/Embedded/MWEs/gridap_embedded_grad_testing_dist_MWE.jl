using Pkg; Pkg.activate()

using Gridap

using GridapDistributed, GridapPETSc, PartitionedArrays

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData
import Gridap.Geometry: get_node_coordinates, collect1d

include("../embedded_measures.jl")

parts = (2,2);
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

φh = interpolate(x->-cos(4π*x[1])*cos(4*pi*x[2])/4-0.2/4,V_φ)
embedded_meas = EmbeddedMeasureCache(φh,V_φ)
update_meas(args...) = update_embedded_measures!(embedded_meas,args...)
get_meas(args...) = get_embedded_measures(embedded_meas,args...)

_j(φ,dΩ,dΩ1,dΩ2,dΓ) = ∫(φ)dΩ1 + ∫(φ)dΩ2 + ∫(φ)dΓ

j_iem = IntegrandWithEmbeddedMeasure(_j,(dΩ,),update_meas)
j_iem(φh)
gradient(j_iem,φh)