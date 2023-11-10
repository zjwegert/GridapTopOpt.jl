
using Gridap.Helpers
using GridapDistributed: DistributedDiscreteModel
using PartitionedArrays: getany

# API definition for AdvectionStencil

abstract type AdvectionStencil end

function allocate_caches(::AdvectionStencil,φ,vel)
  nothing # By default, no caches are required.
end

function reinit!(::AdvectionStencil,φ_new,φ_old,vel,Δt,Δx,caches)
  @abstractmethod
end

function advect!(::AdvectionStencil,φ,vel,Δt,Δx,caches)
  @abstractmethod
end

function compute_Δt(::AdvectionStencil,φ,vel)
  @abstractmethod
end

# First order stencil

struct FirstOrderStencil{D,T} <: AdvectionStencil
  γ :: T
  function FirstOrderStencil(D::Integer,γ::T) where T
    new{D,T}(γ)
  end
end

function allocate_caches(::FirstOrderStencil,φ,vel)
  # Copy code here...
end

function reinit!(::FirstOrderStencil,φ_new,φ_old,vel,Δt,Δx,caches)
  # Copy code here...
end

function advect!(::FirstOrderStencil,φ,vel,Δt,Δx,caches)
  # Copy code here...
end

function compute_Δt(s::FirstOrderStencil{D,T},φ,vel) where {D,T}
  γ = s.γ
  v_norm = maximum(abs,vel)
  return γ * min(Δ...) / (eps(T)^2 + v_norm)
end

# Distributed advection stencil

struct DistributedAdvectionStencil
  stencil :: AdvectionStencil
  model
  max_steps
  tol
  Δ
  local_sizes
end

function AdvectionStencil(stencil::AdvectionStencil,
                          model::DistributedDiscreteModel,
                          max_steps::Int,
                          tol::T) where T
  local_sizes, local_Δ = map(local_views(model)) do model
    desc = get_cartesian_descriptor(model)
    return desc.partition .+ 1, desc.sizes
  end
  Δ = PartitionedArrays.getany(local_Δ)
  return DistributedAdvectionStencil(stencil,model,max_steps,tol,Δ,local_sizes)
end

function allocate_caches(s::DistributedAdvectionStencil,φ::PVector,vel::PVector)
  local_caches = map(local_views(φ),local_views(vel)) do φ,vel
    allocate_caches(s.stencil,φ,vel)
  end
  φ_tmp   = similar(φ)
  vel_tmp = similar(vel)
  return φ_tmp, vel_tmp, local_caches
end

function advect!(s::DistributedAdvectionStencil,φ::PVector,vel::PVector,caches)
  _, _, local_caches = caches

  ## CFL Condition (requires γ≤1.0)
  Δt = compute_Δt(s.stencil,φ,vel)
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

function reinit!(s::DistributedAdvectionStencil,φ::PVector,vel::PVector,caches)
  φ_tmp, vel_tmp, local_caches = caches

  # Compute approx sign function S
  vel_tmp .= @. φ / sqrt(φ*φ + prod(Δ))

  ## CFL Condition (requires γ≤0.5)
  Δt = compute_Δt(s.stencil,φ,1.0) # As inform(vel_tmp) = 1.0

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
