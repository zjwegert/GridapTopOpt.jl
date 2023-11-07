using GridapDistributed
using Gridap
using PartitionedArrays
using Gridap.FESpaces

np = (2,1)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end
model = CartesianDiscreteModel(ranks,np,(0,1,0,1),(5,1);isperiodic=(true,false))

reffe = ReferenceFE(lagrangian,Float64,1)
V = FESpace(model,reffe)

x_vals = map(local_to_global,partition(V.gids))
x = PVector(x_vals,partition(V.gids))

local_sizes = DebugArray([(5,2),(6,2)])
y_vals = map(local_values(x),local_sizes) do x,sz
  reshape(circshift(reshape(x,sz),(-1,0)),prod(sz))
end
y = PVector(y_vals,partition(V.gids))

partition(y)
consistent!(y) |> fetch
partition(y)



