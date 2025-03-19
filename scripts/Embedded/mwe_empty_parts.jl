
using Gridap
using Gridap.Geometry, Gridap.FESpaces

using GridapEmbedded
using GridapEmbedded.Interfaces, GridapEmbedded.LevelSetCutters

using GridapDistributed, PartitionedArrays

using GridapTopOpt

np = (2,1)
ranks = DebugArray([1,2])

#model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1),(4,4)))
model = UnstructuredDiscreteModel(CartesianDiscreteModel(ranks,np,(0,1,0,1),(4,4)))

reffe_m = ReferenceFE(lagrangian,Float64,1)
M = FESpace(model,reffe_m)

ls(x) = ifelse(x[1] > 0.8,-1.0,1.0)
mh = interpolate(ls,M)

geo = DiscreteGeometry(mh,model)
cutgeo = cut(model,geo)

Ωin = Triangulation(cutgeo,PHYSICAL)
Ωac = Triangulation(cutgeo,ACTIVE)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},1)
U = FESpace(Ωac,reffe_u)
