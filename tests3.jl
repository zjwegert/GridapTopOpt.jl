
using Gridap, GridapDistributed, PartitionedArrays
using Gridap.Geometry, Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.Adaptivity

function get_terms(poly::Polytope, orders)
	_nodes, facenodes = Gridap.ReferenceFEs._compute_nodes(poly, orders)
	terms = Gridap.ReferenceFEs._coords_to_terms(_nodes, orders)
	return terms
end

function get_dof_map(poly::Polytope, orders)
	maps1D = map(ord -> map(t -> t[1], get_terms(SEGMENT, [ord])), orders)
	terms  = get_terms(poly, orders)
	dofmap = map(t -> CartesianIndex(map((x, m) -> findfirst(mi -> mi == x, m), Tuple(t), maps1D)), terms)
	return dofmap
end

model = CartesianDiscreteModel((0,1,0,1),(4,4))

order = 2
reffe = ReferenceFE(lagrangian,Float64,order)
V = FESpace(model,reffe)

reffe2 = LagrangianRefFE(Float64,QUAD,(2,1))

cell_dof_ids = get_cell_dof_ids(V)
n2c_map = Geometry.get_faces(get_grid_topology(model),0,2)

rr = RefinementRule(QUAD,(2,2))
rr_grid = Adaptivity.get_ref_grid(rr)
node_ids = get_cell_node_ids(rr_grid)


terms = get_terms(QUAD,(2*order,2*order))
lindices = LinearIndices((2*order+1,2*order+1))

dof_to_node_layout = map(t -> lindices[t],terms)
node_to_dof_layout = Vector{Int}(undef,length(dof_to_node_layout))
for (dof,node) in enumerate(dof_to_node_layout)
  node_to_dof_layout[node] = dof
end

cell_dofs = map(nodes -> node_to_dof_layout[nodes],node_ids)
