
using Gridap
using Gridap.Geometry, Gridap.Arrays, Gridap.TensorValues, Gridap.Helpers
using Gridap.ReferenceFEs
using Gridap.ReferenceFEs: get_graph, isactive
using Gridap.Adaptivity

using GridapEmbedded, GridapEmbedded.Interfaces, GridapEmbedded.LevelSetCutters
using GridapTopOpt

using GridapDistributed, PartitionedArrays


n = 20
ranks = DebugArray(collect(1:4))

#model = simplexify(CartesianDiscreteModel((0,1,0,1),(n,n)))
#model = simplexify(CartesianDiscreteModel((0,1,0,1,0,1),(n,n,n)))
model = simplexify(CartesianDiscreteModel(ranks,(2,2),(0,1,0,1),(n,n)))

reffe = ReferenceFE(lagrangian,Float64,1)
V = TestFESpace(model,reffe)
# φh = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-0.35,V) # Circle
#φh = interpolate(x->cos(2π*x[1])*cos(2π*x[2])*cos(2π*x[3])-0.11,V)
φh = interpolate(x->cos(2π*x[1])*cos(2π*x[2])-0.11,V)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

cell_values = map(get_cell_dof_values,local_views(φh))
pmodel, subcell_to_inout, subcell_to_cell = GridapTopOpt.cut_conforming(model,cell_values);

cf_IN, cf_OUT = GridapTopOpt.get_isolated_volumes_mask_polytopal(
  model,cell_values,["boundary"]
)

writevtk(
  Triangulation(model),"results/polymodel",
  cellfields=[
    "φh"=>φh,
    "cf_IN"=>cf_IN,
    "cf_OUT"=>cf_OUT,
  ],
  append=false
)

writevtk(pmodel,"results/in_model";append=false)

in_model = Geometry.restrict(pmodel,findall(subcell_to_inout .== IN))
writevtk(in_model,"results/in_model";append=false)
