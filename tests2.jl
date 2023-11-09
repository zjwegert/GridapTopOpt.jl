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

dim = 2
trian = Triangulation(model)
coords = map(local_views(trian)) do trian
  node_coords = Gridap.Geometry.get_node_coordinates(trian)
  coords = Vector{Float64}(undef,dim*length(node_coords))
  k = 1
  for p in node_coords
    for d in 1:dim
      coords[k] = p[d]
      k += 1
    end
  end
  return coords
end

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

