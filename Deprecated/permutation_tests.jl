using Gridap
using Gridap.Geometry, Gridap.FESpaces, Gridap.ReferenceFEs

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

D  = 2
nc = (D==2) ? (2,2) : (2,2,2)
domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
model  = CartesianDiscreteModel(domain,nc)

order = 2
poly  = (D==2) ? QUAD : HEX
reffe = LagrangianRefFE(Float64,poly,order)
V = FESpace(model,reffe)

ndofs =  order .* nc .+ 1
new_dof_ids  = LinearIndices(ndofs)

o2n_dof_map = fill(-1,num_free_dofs(V))
n2o_dof_map = fill(-1,num_free_dofs(V))

terms = get_terms(poly, fill(order,D))
cell_dof_ids = get_cell_dof_ids(V)
cache_cell_dof_ids = array_cache(cell_dof_ids)
for (iC,cell) in enumerate(CartesianIndices(nc))
  first_new_dof  = order .* (Tuple(cell) .- 1) .+ 1
  new_dofs_range = map(i -> i:i+order,first_new_dof)
  new_dofs = view(new_dof_ids,new_dofs_range...)

  cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,iC)
  for (iDof, dof) in enumerate(cell_dofs)
    t = terms[iDof]
    o2n_dof_map[dof] = new_dofs[t]
    n2o_dof_map[new_dofs[t]] = dof
  end
end

