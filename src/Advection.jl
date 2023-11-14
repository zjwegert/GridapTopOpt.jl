
using Gridap.Helpers
using GridapDistributed: DistributedDiscreteModel
using PartitionedArrays: getany, tuple_of_arrays

# API definition for Stencil

abstract type Stencil end

function allocate_caches(::Stencil,φ,vel)
  nothing # By default, no caches are required.
end

function reinit!(::Stencil,φ_new,φ_old,vel,Δt,Δx,caches)
  @abstractmethod
end

function advect!(::Stencil,φ,vel,Δt,Δx,caches)
  @abstractmethod
end

function compute_Δt(::Stencil,φ,vel)
  @abstractmethod
end

# First order stencil

struct FirstOrderStencil{D,T} <: Stencil
  function FirstOrderStencil(D::Integer,::Type{T}) where T<:Real
    new{D,T}()
  end
end

function allocate_caches(::FirstOrderStencil{2},φ,vel)
  D⁺ʸ = similar(φ); D⁺ˣ = similar(φ)
  D⁻ʸ = similar(φ); D⁻ˣ = similar(φ)
  ∇⁺  = similar(φ); ∇⁻  = similar(φ)
  return D⁺ʸ, D⁺ˣ, D⁻ʸ, D⁻ˣ, ∇⁺, ∇⁻
end

function reinit!(::FirstOrderStencil{2,T},φ_new,φ,vel,Δt,Δx,caches) where T
  D⁺ʸ, D⁺ˣ, D⁻ʸ, D⁻ˣ, ∇⁺, ∇⁻ = caches
  Δx, Δy = Δx
  # Prepare shifted lsf
  circshift!(D⁺ʸ,φ,(0,-1)); circshift!(D⁻ʸ,φ,(0,1))
  circshift!(D⁺ˣ,φ,(-1,0)); circshift!(D⁻ˣ,φ,(1,0))
  # Forward (+) & Backward (-)
  D⁺ʸ .= @. (D⁺ʸ - φ)/Δy; D⁺ʸ[:,end] .= zero(T)
  D⁺ˣ .= @. (D⁺ˣ - φ)/Δx; D⁺ˣ[end,:] .= zero(T)
  D⁻ʸ .= @. (φ - D⁻ʸ)/Δy; D⁻ʸ[:,1]   .= zero(T)
  D⁻ˣ .= @. (φ - D⁻ˣ)/Δx; D⁻ˣ[1,:]   .= zero(T)
  # Operators
  ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2);
  ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2);
  # Update
  φ_new .= @. φ - Δt*(max(vel,0)*∇⁺ + min(vel,0)*∇⁻ - vel)
  return φ_new
end

function advect!(::FirstOrderStencil{2,T},φ,vel,Δt,Δx,caches) where T
  D⁺ʸ, D⁺ˣ, D⁻ʸ, D⁻ˣ, ∇⁺, ∇⁻ = caches
  Δx, Δy = Δx
  # Prepare shifted lsf
  circshift!(D⁺ʸ,φ,(0,-1)); circshift!(D⁻ʸ,φ,(0,1))
  circshift!(D⁺ˣ,φ,(-1,0)); circshift!(D⁻ˣ,φ,(1,0))
  # Forward (+) & Backward (-)
  D⁺ʸ .= @. (D⁺ʸ - φ)/Δy; D⁺ʸ[:,end] .= zero(T)
  D⁺ˣ .= @. (D⁺ˣ - φ)/Δx; D⁺ˣ[end,:] .= zero(T)
  D⁻ʸ .= @. (φ - D⁻ʸ)/Δy; D⁻ʸ[:,1]   .= zero(T)
  D⁻ˣ .= @. (φ - D⁻ˣ)/Δx; D⁻ˣ[1,:]   .= zero(T)
  # Operators
  ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2)
  ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2)
  # Update
  φ .= @. φ - Δt*(max(vel,0)*∇⁺ + min(vel,0)*∇⁻)
  return φ
end

function allocate_caches(::FirstOrderStencil{3},φ,vel)
  D⁺ᶻ = similar(φ); D⁺ʸ = similar(φ); D⁺ˣ = similar(φ)
  D⁻ᶻ = similar(φ); D⁻ʸ = similar(φ); D⁻ˣ = similar(φ)
  ∇⁺  = similar(φ); ∇⁻  = similar(φ)
  return D⁺ᶻ, D⁺ʸ, D⁺ˣ, D⁻ᶻ, D⁻ʸ, D⁻ˣ, ∇⁺, ∇⁻
end

function reinit!(::FirstOrderStencil{3,T},φ_new,φ,vel,Δt,Δx,caches) where T
  D⁺ᶻ, D⁺ʸ, D⁺ˣ, D⁻ᶻ, D⁻ʸ, D⁻ˣ, ∇⁺, ∇⁻=caches
  Δx, Δy, Δz = Δx
  # Prepare shifted lsf
  circshift!(D⁺ʸ,φ,(0,-1,0)); circshift!(D⁻ʸ,φ,(0,1,0))
  circshift!(D⁺ˣ,φ,(-1,0,0)); circshift!(D⁻ˣ,φ,(1,0,0))
  circshift!(D⁺ᶻ,φ,(0,0,-1)); circshift!(D⁻ᶻ,φ,(0,0,1))
  # Forward (+) & Backward (-)
  D⁺ʸ .= (D⁺ʸ - φ)/Δy; D⁺ʸ[:,end,:] .= zero(T)
  D⁺ˣ .= (D⁺ˣ - φ)/Δx; D⁺ˣ[end,:,:] .= zero(T)
  D⁺ᶻ .= (D⁺ᶻ - φ)/Δz; D⁺ᶻ[:,:,end] .= zero(T)
  D⁻ʸ .= (φ - D⁻ʸ)/Δy; D⁻ʸ[:,1,:]   .= zero(T)
  D⁻ˣ .= (φ - D⁻ˣ)/Δx; D⁻ˣ[1,:,:]   .= zero(T)
  D⁻ᶻ .= (φ - D⁻ᶻ)/Δz; D⁻ᶻ[:,:,1]   .= zero(T)
  # Operators
  ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2 + max(D⁻ᶻ,0)^2 + min(D⁺ᶻ,0)^2)
  ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2 + max(D⁺ᶻ,0)^2 + min(D⁻ᶻ,0)^2)
  # Update
  φ_new .= @. φ - Δt*(max(vel,0)*∇⁺ + min(S,0)*∇⁻ - vel)
  return φ_new
end

function advect!(::FirstOrderStencil{3,T},φ,vel,Δt,Δx,caches) where T
  D⁺ᶻ, D⁺ʸ, D⁺ˣ, D⁻ᶻ, D⁻ʸ, D⁻ˣ, ∇⁺, ∇⁻=caches
  Δx, Δy, Δz = Δx
  # Prepare shifted lsf
  circshift!(D⁺ʸ,φ,(0,-1,0)); circshift!(D⁻ʸ,φ,(0,1,0))
  circshift!(D⁺ˣ,φ,(-1,0,0)); circshift!(D⁻ˣ,φ,(1,0,0))
  circshift!(D⁺ᶻ,φ,(0,0,-1)); circshift!(D⁻ᶻ,φ,(0,0,1))
  # Forward (+) & Backward (-)
  D⁺ʸ .= (D⁺ʸ - φ)/Δy; D⁺ʸ[:,end,:] .= zero(T)
  D⁺ˣ .= (D⁺ˣ - φ)/Δx; D⁺ˣ[end,:,:] .= zero(T)
  D⁺ᶻ .= (D⁺ᶻ - φ)/Δz; D⁺ᶻ[:,:,end] .= zero(T)
  D⁻ʸ .= (φ - D⁻ʸ)/Δy; D⁻ʸ[:,1,:]   .= zero(T)
  D⁻ˣ .= (φ - D⁻ˣ)/Δx; D⁻ˣ[1,:,:]   .= zero(T)
  D⁻ᶻ .= (φ - D⁻ᶻ)/Δz; D⁻ᶻ[:,:,1]   .= zero(T)
  # Operators
  ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2 + max(D⁻ᶻ,0)^2 + min(D⁺ᶻ,0)^2)
  ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2 + max(D⁺ᶻ,0)^2 + min(D⁻ᶻ,0)^2)
  # Update
  φ .= @. φ - Δt*(max(vel,0)*∇⁺ + min(vel,0)*∇⁻)
  return φ
end

function compute_Δt(s::FirstOrderStencil{D,T},γ,φ,vel) where {D,T}
  v_norm = maximum(abs,vel)
  return γ * min(Δ...) / (eps(T)^2 + v_norm)
end

# Distributed advection stencil

struct DistributedAdvectionStencil{O}
  stencil   :: Stencil
  model     :: DistributedDiscreteModel
  space     :: DistributedFESpace
  perm      :: Vector
  max_steps
  max_steps_reinit 
  tol
  Δ
  local_sizes
end

function AdvectionStencil(stencil::Stencil,
                          model::DistributedDiscreteModel,
                          space::DistributedFESpace,
                          max_steps::Int,max_steps_reinit::Int,tol::T) where T
  order = get_order(first(Gridap.CellData.get_data(get_fe_basis(V))))
  local_sizes, local_Δ, perm = map(local_views(model),local_views(space)) do model, space
    desc = get_cartesian_descriptor(model)
    dof_permutation = create_dof_permutation(model,space,order)
    return desc.partition .+ 1, desc.sizes, dof_permutation
  end |> PartitionedArrays.tuple_of_arrays
  Δ = PartitionedArrays.getany(local_Δ)
  return DistributedAdvectionStencil{order}(
    stencil,model,space,perm,max_steps,max_steps_reinit,tol,Δ,local_sizes)
end

Gridap.ReferenceFEs.get_order(f::Gridap.Fields.LinearCombinationFieldVector) = get_order(f.fields)

function create_dof_permutation(model::CartesianDiscreteModel{Dc},
                                space::UnconstrainedFESpace,
                                order::Integer) where Dc
  function get_terms(poly::Polytope, orders)
    _nodes, facenodes = Gridap.ReferenceFEs._compute_nodes(poly, orders)
    terms = Gridap.ReferenceFEs._coords_to_terms(_nodes, orders)
    return terms
  end
  desc = get_cartesian_descriptor(model)
  
  ncells = desc.partition
  ndofs  = order .* ncells .+ 1
  @check prod(ndofs) == num_free_dofs(space)

  new_dof_ids  = LinearIndices(ndofs)
  #o2n_dof_map = fill(-1,num_free_dofs(V))
  n2o_dof_map = fill(-1,num_free_dofs(V))

  terms = get_terms(poly, fill(order,Dc))
  cell_dof_ids = get_cell_dof_ids(V)
  cache_cell_dof_ids = array_cache(cell_dof_ids)
  for (iC,cell) in enumerate(CartesianIndices(nc))
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

function allocate_caches(s::DistributedAdvectionStencil{O},φ::PVector,vel::PVector) where O
  local_caches = map(local_views(φ),local_views(vel)) do φ,vel
    allocate_caches(s.stencil,φ,vel)
  end
  φ_tmp   = similar(φ)
  vel_tmp = similar(vel)
  perm_caches = (O >= 2) ? (similar(φ), similar(vel)) : nothing
  return φ_tmp, vel_tmp, perm_caches, local_caches
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

function advect!(s::DistributedAdvectionStencil{O},φ::PVector,vel::PVector,γ,caches) where O
  _, _, perm_caches, local_caches = caches
  
  _φ   = (O >= 2) ? permute!(perm_caches[1],φ,s.perm) : φ
  _vel = (O >= 2) ? permute!(perm_caches[2],vel,s.perm) : vel

  ## CFL Condition (requires γ≤1.0)
  Δt = compute_Δt(s.stencil,γ,φ,vel)
  for _ ∈ Base.OneTo(s.max_steps)
    # Apply operations across partitions
    map(local_views(_φ),local_views(_vel),local_caches,s.local_sizes) do _φ,_vel,caches,S
      φ_mat   = reshape(_φ,S)
      vel_mat = reshape(_vel,S)
      advect!(s.stencil,φ_mat,vel_mat,s.Δ,Δt,caches)
    end
    # Update ghost nodes
    consistent!(_φ) |> fetch
  end
  φ = (O >= 2) ? permute_inv!(φ,_φ,s.perm) : _φ
  return φ
end

function reinit!(s::DistributedAdvectionStencil,φ::PVector,γ,caches)
  φ_tmp, vel_tmp, perm_caches, local_caches = caches
  _φ = (O >= 2) ? permute!(perm_caches[1],φ,s.perm) : φ

  # Compute approx sign function S
  vel_tmp .= @. _φ / sqrt(_φ*_φ + prod(Δ))

  ## CFL Condition (requires γ≤0.5)
  Δt = compute_Δt(s.stencil,γ,_φ,1.0) # As inform(vel_tmp) = 1.0

  # Apply operations across partitions
  step = 1; err = maximum(abs,φ); fill!(φ_tmp,0.0)
  while (err > tol) && (step <= max_steps) 
    # Step of 1st order upwind reinitialisation equation
    map(local_views(φ_tmp),local_views(_φ),local_views(vel_tmp),local_caches,s.local_sizes) do φ_tmp,φ,vel_tmp,local_caches,S
      φ_tmp_mat   = reshape(φ,S)
      φ_mat       = reshape(φ,S)
      vel_tmp_mat = reshape(vel_tmp,S)
      reinit!(s.stencil,φ_tmp_mat,φ_mat,vel_tmp_mat,s.Δ,Δt,local_caches)
    end

    # Compute error
    _φ .-= φ_tmp # φ - φ_tmp
    err = maximum(abs,_φ) # Ghosts not needed yet: partial maximums computed using owned values only. 
    step += 1

    # Update φ
    copy!(_φ,φ_tmp)
    consistent!(_φ) |> fetch # We exchange ghosts here!
  end
  φ = (O >= 2) ? permute_inv!(φ,_φ,s.perm) : _φ
  return φ
end
