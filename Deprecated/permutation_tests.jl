using Gridap
using Gridap.Geometry, Gridap.FESpaces, Gridap.ReferenceFEs, Gridap.Helpers
using GridapDistributed, PartitionedArrays
using CircularArrays

function create_dof_permutation(model::CartesianDiscreteModel{Dc},
                                space::UnconstrainedFESpace,
                                order::Integer) where Dc
  function get_terms(poly::Polytope, orders)
    _nodes, facenodes = Gridap.ReferenceFEs._compute_nodes(poly, orders)
    terms = Gridap.ReferenceFEs._coords_to_terms(_nodes, orders)
    return terms
  end
  desc = get_cartesian_descriptor(model)
  
  periodic = desc.isperiodic
  ncells   = desc.partition
  ndofs    = order .* ncells .+ 1 .- periodic
  @check prod(ndofs) == num_free_dofs(space)

  new_dof_ids  = CircularArray(LinearIndices(ndofs))
  n2o_dof_map = fill(-1,num_free_dofs(space))

  terms = get_terms(first(get_polytopes(model)), fill(order,Dc))
  cell_dof_ids = get_cell_dof_ids(space)
  cache_cell_dof_ids = array_cache(cell_dof_ids)
  for (iC,cell) in enumerate(CartesianIndices(ncells))
    first_new_dof  = order .* (Tuple(cell) .- 1) .+ 1
    new_dofs_range = map(i -> i:i+order,first_new_dof)
    new_dofs = view(new_dof_ids,new_dofs_range...)

    cell_dofs = getindex!(cache_cell_dof_ids,cell_dof_ids,iC)
    for (iDof, dof) in enumerate(cell_dofs)
      t = terms[iDof]
      n2o_dof_map[new_dofs[t]] = dof
    end
  end

  return n2o_dof_map
end

function create_dof_permutation(model::GridapDistributed.DistributedDiscreteModel,
                                space::GridapDistributed.DistributedFESpace,
                                order::Integer) where Dc
  local_perms = map(local_views(model),local_views(space)) do model, space
    create_dof_permutation(model,space,order)
  end
  return local_perms
end

D  = 2
np = Tuple(fill(2,D))

ranks = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end

nc = (D==2) ? (2,2) : (2,2,2)
domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
#model  = CartesianDiscreteModel(ranks,np,domain,nc)
model  = CartesianDiscreteModel(domain,(4,3),isperiodic=(true,false))

order = 2
poly  = (D==2) ? QUAD : HEX
reffe = LagrangianRefFE(Float64,poly,order)
space = FESpace(model,reffe)

perm_gids = create_dof_permutation(model,space,order)

uh = interpolate(x->x[1]-x[2],space)
x = get_free_dof_values(uh)
x_perm = pfill(0.0,partition(perm_gids));
