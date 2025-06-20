"""
    struct HamiltonJacobiEvolution{O}

A standard forward Euler in time finite difference method for solving the
Hamilton-Jacobi evolution equation and reinitialisation equation
on order `O` finite elements in serial or parallel.

Based on the scheme by Osher and Fedkiw ([link](https://doi.org/10.1007/b98879)).

# Parameters

- `stencil::Stencil`: Spatial finite difference stencil for a single step HJ
  equation and reinitialisation equation.
- `model`: A `CartesianDiscreteModel`.
- `space`: FE space for level-set function
- `perm`: A permutation vector
- `params`: Tuple of additional params
- `cache`: Stencil cache
"""
struct HamiltonJacobiEvolution{O} <: LevelSetEvolution
  stencil :: Stencil
  model
  space
  perm
  params
  cache
end

"""
    HamiltonJacobiEvolution(stencil::Stencil,model,space,tol,max_steps,max_steps_reinit)

Create an instance of `HamiltonJacobiEvolution` given a stencil, model, FE space, and
additional optional arguments. This automatically creates the DoF permutation
to handle high-order finite elements.
"""
function HamiltonJacobiEvolution(
  stencil::Stencil,
  model,
  space,
  tol=1.e-3,
  max_steps=100,
  max_steps_reinit=2000
)
  @check isa(model,CartesianDiscreteModel) || isa(model,DistributedDiscreteModel) """
    We expect `model` to be a `CartesianDiscreteModel` or `DistributedDiscreteModel`
    for the current implementation of finite differencing on arbitrary order Lagrangian
    finite elements.
  """

  # Parameters
  order, isperiodic, Δ, ndof = get_stencil_params(model,space)
  params = (;isperiodic,Δ,ndof,max_steps,max_steps_reinit,tol)

  # Check that we have sufficient order
  check_order(stencil,order)

  # Dof permutation
  perm = create_dof_permutation(model,space,order)

  # Caches
  φ, vel = zero_free_values(space), zero_free_values(space)
  cache  = allocate_caches(stencil,φ,vel,perm,order,ndof)

  return HamiltonJacobiEvolution{order}(stencil,model,space,perm,params,cache)
end

"""
    get_dof_Δ(m::HamiltonJacobiEvolution)

Return the distance betweem degree of freedom
"""
function get_dof_Δ(m::HamiltonJacobiEvolution)
  return m.params.Δ
end

# Compute the time step for the `HamiltonJacobiEvolution`.
function compute_Δt(::HamiltonJacobiEvolution,Δ,γ,φ,vel)
  T = eltype(γ)
  v_norm = maximum(abs,vel)
  return γ * min(Δ...) / (eps(T)^2 + v_norm)
end

"""
    evolve!(s::HamiltonJacobiEvolution{O},φ,vel,γ) where O

Solve the Hamilton-Jacobi evolution equation using the `HamiltonJacobiEvolution`.

# Hamilton-Jacobi evolution equation
``\\frac{\\partial\\phi}{\\partial t} + V(\\boldsymbol{x})\\lVert\\boldsymbol{\\nabla}\\phi\\rVert = 0,``

with ``\\phi(0,\\boldsymbol{x})=\\phi_0(\\boldsymbol{x})`` and ``\\boldsymbol{x}\\in D,~t\\in(0,T)``.

# Arguments

- `s::HamiltonJacobiEvolution{O}`: Method
- `φ`: level set function as a vector of degrees of freedom
- `vel`: the normal velocity as a vector of degrees of freedom
- `γ`: coeffient on the time step size.
"""
function evolve!(s::HamiltonJacobiEvolution{O},φ::PVector,vel::PVector,γ) where O
  _, _, perm_caches, stencil_cache = s.cache
  Δ, isperiodic,  = s.params.Δ, s.params.isperiodic
  ndof, max_steps = s.params.ndof, s.params.max_steps

  _φ   = (O >= 2) ? permute!(perm_caches[1],φ,s.perm) : φ
  _vel = (O >= 2) ? permute!(perm_caches[2],vel,s.perm) : vel

  ## CFL Condition (requires γ≤1.0)
  Δt = compute_Δt(s,Δ,γ,φ,vel)
  for _ in 1:max_steps
    # Apply operations across partitions
    map(local_views(_φ),local_views(_vel),stencil_cache,ndof) do _φ,_vel,stencil_cache,S
      φ_mat   = reshape(_φ,S)
      vel_mat = reshape(_vel,S)
      evolve!(s.stencil,φ_mat,vel_mat,Δt,Δ,isperiodic,stencil_cache)
    end
    # Update ghost nodes
    consistent!(_φ) |> fetch
  end
  φ = (O >= 2) ? permute_inv!(φ,_φ,s.perm) : _φ
  return φ
end

function evolve!(s::HamiltonJacobiEvolution{O},φ::Vector,vel::Vector,γ) where O
  _, _, perm_caches, stencil_cache = s.cache
  Δ, isperiodic,  = s.params.Δ, s.params.isperiodic
  ndof, max_steps = s.params.ndof, s.params.max_steps

  _φ   = (O >= 2) ? permute!(perm_caches[1],φ,s.perm) : φ
  _vel = (O >= 2) ? permute!(perm_caches[2],vel,s.perm) : vel

  ## CFL Condition (requires γ≤1.0)
  Δt = compute_Δt(s,Δ,γ,φ,vel)
  for _ in 1:max_steps
    φ_mat   = reshape(_φ,ndof)
    vel_mat = reshape(_vel,ndof)
    evolve!(s.stencil,φ_mat,vel_mat,Δt,Δ,isperiodic,stencil_cache)
  end
  φ = (O >= 2) ? permute_inv!(φ,_φ,s.perm) : _φ
  return φ
end

function evolve!(s::HamiltonJacobiEvolution,φh,args...)
  evolve!(s,get_free_dof_values(φh),args...)
end

"""
    reinit!(s::HamiltonJacobiEvolution{O},φ,γ) where O

Solve the reinitialisation equation using the `HamiltonJacobiEvolution`.

# Reinitialisation equation
``\\frac{\\partial\\phi}{\\partial t} + \\mathrm{sign}(\\phi_0)(\\lVert\\boldsymbol{\\nabla}\\phi\\rVert-1) = 0,``

with ``\\phi(0,\\boldsymbol{x})=\\phi_0(\\boldsymbol{x})`` and ``\\boldsymbol{x}\\in D,~t\\in(0,T)``.

# Arguments

- `s::HamiltonJacobiEvolution{O}`: Method
- `φ`: level set function as a vector of degrees of freedom
- `γ`: coeffient on the time step size.
"""
function reinit!(s::HamiltonJacobiEvolution{O},φ::PVector,γ) where O
  φ_tmp, vel_tmp, perm_caches, stencil_cache = s.cache
  Δ, isperiodic, ndof  = s.params.Δ, s.params.isperiodic, s.params.ndof
  tol, max_steps = s.params.tol, s.params.max_steps_reinit

  _φ = (O >= 2) ? permute!(perm_caches[1],φ,s.perm) : φ

  ## CFL Condition (requires γ≤0.5). Note infnorm(S) = 1.0
  Δt = compute_Δt(s,Δ,γ,_φ,1.0)

  # Apply operations across partitions
  step = 1; err = maximum(abs,φ); fill!(φ_tmp,0.0)
  while (err > tol) && (step <= max_steps)
    # Step of 1st order upwind reinitialisation equation
    map(local_views(φ_tmp),local_views(_φ),local_views(vel_tmp),stencil_cache,ndof) do φ_tmp,_φ,vel_tmp,stencil_cache,S
      φ_tmp_mat   = reshape(φ_tmp,S)
      φ_mat       = reshape(_φ,S)
      vel_tmp_mat = reshape(vel_tmp,S)
      reinit!(s.stencil,φ_tmp_mat,φ_mat,vel_tmp_mat,Δt,Δ,isperiodic,stencil_cache)
    end

    # Compute error
    _φ .-= φ_tmp # φ - φ_tmp
    err = maximum(abs,_φ)
    step += 1

    # Update φ
    copy!(_φ,φ_tmp)
    consistent!(_φ) |> fetch # We exchange ghosts here!
  end
  φ = (O >= 2) ? permute_inv!(φ,_φ,s.perm) : _φ
  return φ
end

function reinit!(s::HamiltonJacobiEvolution{O},φ::Vector,γ) where O
  φ_tmp, vel_tmp, perm_caches, stencil_cache = s.cache
  Δ, isperiodic, ndof = s.params.Δ, s.params.isperiodic, s.params.ndof
  tol, max_steps = s.params.tol, s.params.max_steps_reinit

  _φ = (O >= 2) ? permute!(perm_caches[1],φ,s.perm) : φ

  ## CFL Condition (requires γ≤0.5)
  Δt = compute_Δt(s,Δ,γ,_φ,1.0)

  # Apply operations across partitions
  step = 1; err = maximum(abs,φ); fill!(φ_tmp,0.0)
  while (err > tol) && (step <= max_steps)
    # Step of 1st order upwind reinitialisation equation
    φ_tmp_mat   = reshape(φ_tmp,ndof)
    φ_mat       = reshape(_φ,ndof)
    vel_tmp_mat = reshape(vel_tmp,ndof)
    reinit!(s.stencil,φ_tmp_mat,φ_mat,vel_tmp_mat,Δt,Δ,isperiodic,stencil_cache)

    # Compute error
    _φ .-= φ_tmp
    err = maximum(abs,_φ)
    step += 1

    # Update φ
    copy!(_φ,φ_tmp)
  end
  φ = (O >= 2) ? permute_inv!(φ,_φ,s.perm) : _φ
  return φ
end

function reinit!(s::HamiltonJacobiEvolution,φh,args...)
  reinit!(s,get_free_dof_values(φh),args...)
end

## Utilities
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