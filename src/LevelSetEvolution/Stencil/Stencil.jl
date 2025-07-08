"""
    abstract type Stencil

Finite difference stencil for a single step of the Hamilton-Jacobi
evolution equation and reinitialisation equation.

Your own spatial stencil can be implemented by extending the methods below.
"""
abstract type Stencil end

"""
    allocate_caches(::Stencil,φ,vel)

Allocate caches for a given `Stencil`.
"""
function allocate_caches(::Stencil,φ,vel)
  nothing # By default, no caches are required.
end

"""
    check_order(::Stencil,order)

Throw error if insufficient reference element order
to implement stencil in parallel.
"""
function check_order(::Stencil,order)
  @abstractmethod
end

"""
    reinit!(::Stencil,φ_new,φ,vel,Δt,Δx,isperiodic,caches) -> φ

Single finite difference step of the reinitialisation equation for a given `Stencil`.
"""
function reinit!(::Stencil,φ_new,φ,vel,Δt,Δx,isperiodic,caches)
  @abstractmethod
end

"""
    evolve!(::Stencil,φ,vel,Δt,Δx,isperiodic,caches) -> φ

Single finite difference step of the Hamilton-Jacobi evoluation equation for a given
`Stencil`.
"""
function evolve!(::Stencil,φ,vel,Δt,Δx,isperiodic,caches)
  @abstractmethod
end

include("FirstOrderStencil.jl")

## Utilities
## Stencil utilities
function get_stencil_params(model::CartesianDiscreteModel,space::FESpace)
  order = get_order(first(Gridap.CellData.get_data(get_fe_basis(space))))
  desc = get_cartesian_descriptor(model)
  isperiodic = desc.isperiodic
  ndof = order .* desc.partition .+ 1 .- isperiodic
  Δ = desc.sizes ./ order
  return order, isperiodic, Δ, ndof
end

function get_stencil_params(model::DistributedDiscreteModel,space::DistributedFESpace)
  order, isperiodic, Δ, ndof = map(local_views(model),local_views(space)) do model, space
    get_stencil_params(model,space)
  end |> PartitionedArrays.tuple_of_arrays

  isperiodic = getany(isperiodic)
  order = getany(order)
  Δ = getany(Δ)
  return order, isperiodic, Δ, ndof
end

Gridap.ReferenceFEs.get_order(f::Gridap.Fields.LinearCombinationFieldVector) = get_order(f.fields)

# Create dof permutation vector to enable finite differences on
#  higher order Lagrangian finite elements on a Cartesian mesh.
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

  new_dof_ids = CircularArray(LinearIndices(ndofs))
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
      #o2n_dof_map[dof] = new_dofs[t]
      n2o_dof_map[new_dofs[t]] = dof
    end
  end

  return n2o_dof_map
end

function create_dof_permutation(model::GridapDistributed.DistributedDiscreteModel,
                                space::GridapDistributed.DistributedFESpace,
                                order::Integer)
  local_perms = map(local_views(model),local_views(space)) do model, space
    create_dof_permutation(model,space,order)
  end
  return local_perms
end

function PartitionedArrays.permute_indices(indices::LocalIndices,perm)
  id = part_id(indices)
  n_glob = global_length(indices)
  l2g = view(local_to_global(indices),perm)
  l2o = view(local_to_owner(indices),perm)
  return LocalIndices(n_glob,id,l2g,l2o)
end

function allocate_caches(s::Stencil,φ::Vector,vel::Vector,perm,order,ndofs)
  stencil_caches = allocate_caches(s,reshape(φ,ndofs),reshape(vel,ndofs))
  φ_tmp   = similar(φ)
  vel_tmp = similar(vel)
  perm_caches = (order >= 2) ? (similar(φ), similar(vel)) : nothing
  return φ_tmp, vel_tmp, perm_caches, stencil_caches
end

function allocate_caches(s::Stencil,φ::PVector,vel::PVector,perm,order,local_ndofs)
  local_stencil_caches = map(local_views(φ),local_views(vel),local_views(local_ndofs)) do φ,vel,ndofs
    allocate_caches(s,reshape(φ,ndofs),reshape(vel,ndofs))
  end

  perm_indices = map(permute_indices,partition(axes(φ,1)),perm)
  perm_caches = (order >= 2) ? (pfill(0.0,perm_indices),pfill(0.0,perm_indices)) : nothing

  φ_tmp   = (order >= 2) ? pfill(0.0,perm_indices) : similar(φ)
  vel_tmp = (order >= 2) ? pfill(0.0,perm_indices) : similar(vel)
  return φ_tmp, vel_tmp, perm_caches, local_stencil_caches
end

function permute!(x_out,x_in,perm)
  for (i_new,i_old) in enumerate(perm)
    x_out[i_new] = x_in[i_old]
  end
  return x_out
end

function permute!(x_out::PVector,x_in::PVector,perm)
  map(permute!,partition(x_out),partition(x_in),perm)
  return x_out
end

function permute_inv!(x_out,x_in,perm)
  for (i_new,i_old) in enumerate(perm)
    x_out[i_old] = x_in[i_new]
  end
  return x_out
end
function permute_inv!(x_out::PVector,x_in::PVector,perm)
  map(permute_inv!,partition(x_out),partition(x_in),perm)
  return x_out
end