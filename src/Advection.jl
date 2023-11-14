
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
  function FirstOrderStencil(D::Integer,γ::T) where T
    new{D,T}(γ)
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

struct DistributedStencil
  stencil         :: Stencil
  model
  max_steps
  reinit_max_steps
  tol
  Δ
  local_sizes
end

function Stencil(stencil::Stencil,
                 model::DistributedDiscreteModel,
                 max_steps::Int,
                 reinit_max_steps::Int,
                 tol::T) where T
  local_sizes, local_Δ = map(local_views(model)) do model
    desc = get_cartesian_descriptor(model)
    return desc.partition .+ 1, desc.sizes
  end
  Δ = PartitionedArrays.getany(local_Δ)
  return DistributedStencil(stencil,model,max_steps,reinit_max_steps,tol,Δ,local_sizes)
end

function allocate_caches(s::DistributedStencil,φ::PVector,vel::PVector)
  local_caches = map(local_views(φ),local_views(vel)) do φ,vel
    allocate_caches(s.stencil,φ,vel)
  end
  φ_tmp   = similar(φ)
  vel_tmp = similar(vel)
  return φ_tmp, vel_tmp, local_caches
end

function advect!(s::DistributedStencil,φ::PVector,vel::PVector,γ,caches)
  _, _, local_caches = caches

  ## CFL Condition (requires γ≤1.0)
  Δt = compute_Δt(s.stencil,γ,φ,vel)
  for _ ∈ Base.OneTo(s.max_steps)
    # Apply operations across partitions
    map(local_views(φ),local_views(),local_views(vel),local_caches,s.local_sizes) do φ,vel,caches,S
      _φ   = reshape(φ,S)
      _vel = reshape(vel,S)
      advect!(s.stencil,_φ,_vel,s.Δ,Δt,caches)
    end
    # Update ghost nodes
    consistent!(φ) |> fetch
  end
  return φ
end

function reinit!(s::DistributedStencil,φ::PVector,vel::PVector,γ,caches)
  φ_tmp, vel_tmp, local_caches = caches

  # Compute approx sign function S
  vel_tmp .= @. φ / sqrt(φ*φ + prod(Δ))

  ## CFL Condition (requires γ≤0.5)
  Δt = compute_Δt(s.stencil,γ,φ,1.0) # As inform(vel_tmp) = 1.0

  # Apply operations across partitions
  step = 1; err = maximum(abs,φ); fill!(φ_tmp,0.0)
  while (err > tol) && (step <= max_steps) 
    # Step of 1st order upwind reinitialisation equation
    map(local_views(φ_tmp),local_views(φ),local_views(vel_tmp),local_caches,s.local_sizes) do φ_tmp,φ,vel_tmp,local_caches,S
      _φ_tmp   = reshape(φ,S)
      _φ       = reshape(φ,S)
      _vel_tmp = reshape(vel,S)
      reinit!(s.stencil,_φ_tmp,_φ,_vel_tmp,s.Δ,Δt,local_caches)
    end

    # Compute error
    φ .-= φ_tmp # φ - φ_tmp
    err = maximum(abs,φ) # Ghosts not needed yet: partial maximums computed using owned values only. 
    step += 1

    # Update φ
    copy!(φ,φ_tmp)
    consistent!(φ) |> fetch # We exchange ghosts here!
  end
  return φ # Should this be φ_new for consistency? 
end
