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

"""
    struct FirstOrderStencil{D,T} <: Stencil end

A first order upwind difference scheme based on Osher and Fedkiw 
([link](https://doi.org/10.1007/b98879)).
"""
struct FirstOrderStencil{D,T} <: Stencil
  function FirstOrderStencil(D::Integer,::Type{T}) where T<:Real
    new{D,T}()
  end
end

function check_order(::FirstOrderStencil,order)
  @check order >= 1 "FirstOrderStencil requires reference element to have order >= 1" 
end

function allocate_caches(::FirstOrderStencil{2},φ,vel)
  D⁺ʸ = similar(φ); D⁺ˣ = similar(φ)
  D⁻ʸ = similar(φ); D⁻ˣ = similar(φ)
  ∇⁺  = similar(φ); ∇⁻  = similar(φ)
  return D⁺ʸ, D⁺ˣ, D⁻ʸ, D⁻ˣ, ∇⁺, ∇⁻
end

function reinit!(::FirstOrderStencil{2,T},φ_new,φ,vel,Δt,Δx,isperiodic,caches) where T
  D⁺ʸ, D⁺ˣ, D⁻ʸ, D⁻ˣ, ∇⁺, ∇⁻ = caches
  Δx, Δy = Δx
  xperiodic,yperiodic = isperiodic
  # Prepare shifted lsf
  circshift!(D⁺ʸ,φ,(0,-1)); circshift!(D⁻ʸ,φ,(0,1))
  circshift!(D⁺ˣ,φ,(-1,0)); circshift!(D⁻ˣ,φ,(1,0))
  # Sign approximation
  ∇⁺ .= @. (D⁺ʸ - D⁻ʸ)/(2Δy); ~yperiodic ? ∇⁺[:,[1,end]] .= zero(T) : 0;
  ∇⁻ .= @. (D⁺ˣ - D⁻ˣ)/(2Δx); ~xperiodic ? ∇⁻[[1,end],:] .= zero(T) : 0;
  ϵₛ = minimum((Δx,Δy))
  vel .= @. φ/sqrt(φ^2 + ϵₛ^2*(∇⁺^2+∇⁻^2))
  # Forward (+) & Backward (-)
  D⁺ʸ .= @. (D⁺ʸ - φ)/Δy; ~yperiodic ? D⁺ʸ[:,end] .= zero(T) : 0;
  D⁺ˣ .= @. (D⁺ˣ - φ)/Δx; ~xperiodic ? D⁺ˣ[end,:] .= zero(T) : 0;
  D⁻ʸ .= @. (φ - D⁻ʸ)/Δy; ~yperiodic ? D⁻ʸ[:,1]   .= zero(T) : 0;
  D⁻ˣ .= @. (φ - D⁻ˣ)/Δx; ~xperiodic ? D⁻ˣ[1,:]   .= zero(T) : 0;
  # Operators
  ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2);
  ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2);
  # Update
  φ_new .= @. φ - Δt*(max(vel,0)*∇⁺ + min(vel,0)*∇⁻ - vel)
  return φ_new
end

function evolve!(::FirstOrderStencil{2,T},φ,vel,Δt,Δx,isperiodic,caches) where T
  D⁺ʸ, D⁺ˣ, D⁻ʸ, D⁻ˣ, ∇⁺, ∇⁻ = caches
  Δx, Δy = Δx
  xperiodic,yperiodic = isperiodic
  # Prepare shifted lsf
  circshift!(D⁺ʸ,φ,(0,-1)); circshift!(D⁻ʸ,φ,(0,1))
  circshift!(D⁺ˣ,φ,(-1,0)); circshift!(D⁻ˣ,φ,(1,0))
  # Forward (+) & Backward (-)
  D⁺ʸ .= @. (D⁺ʸ - φ)/Δy; ~yperiodic ? D⁺ʸ[:,end] .= zero(T) : 0;
  D⁺ˣ .= @. (D⁺ˣ - φ)/Δx; ~xperiodic ? D⁺ˣ[end,:] .= zero(T) : 0;
  D⁻ʸ .= @. (φ - D⁻ʸ)/Δy; ~yperiodic ? D⁻ʸ[:,1]   .= zero(T) : 0;
  D⁻ˣ .= @. (φ - D⁻ˣ)/Δx; ~xperiodic ? D⁻ˣ[1,:]   .= zero(T) : 0;
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

function reinit!(::FirstOrderStencil{3,T},φ_new,φ,vel,Δt,Δx,isperiodic,caches) where T
  D⁺ᶻ, D⁺ʸ, D⁺ˣ, D⁻ᶻ, D⁻ʸ, D⁻ˣ, ∇⁺, ∇⁻=caches
  Δx, Δy, Δz = Δx
  xperiodic,yperiodic,zperiodic = isperiodic
  # Prepare shifted lsf
  circshift!(D⁺ʸ,φ,(0,-1,0)); circshift!(D⁻ʸ,φ,(0,1,0))
  circshift!(D⁺ˣ,φ,(-1,0,0)); circshift!(D⁻ˣ,φ,(1,0,0))
  circshift!(D⁺ᶻ,φ,(0,0,-1)); circshift!(D⁻ᶻ,φ,(0,0,1))
  # Sign approximation
  ∇⁺ .= @. (D⁺ʸ - D⁻ʸ)/(2Δy); ~yperiodic ? ∇⁺[:,[1,end],:] .= zero(T) : 0;
  ∇⁻ .= @. (D⁺ˣ - D⁻ˣ)/(2Δx); ~xperiodic ? ∇⁻[[1,end],:,:] .= zero(T) : 0;
  ∇⁺ .= @. ∇⁺^2+∇⁻^2 # |∇φ|² (partially computed)
  ∇⁻ .= @. (D⁺ᶻ - D⁻ᶻ)/(2Δz); ~xperiodic ? ∇⁻[:,:,[1,end]] .= zero(T) : 0;
  ϵₛ = minimum((Δx,Δy,Δz))
  vel .= @. φ/sqrt(φ^2 + ϵₛ^2*(∇⁺+∇⁻^2))
  # Forward (+) & Backward (-)
  D⁺ʸ .= (D⁺ʸ - φ)/Δy; ~yperiodic ? D⁺ʸ[:,end,:] .= zero(T) : 0;
  D⁺ˣ .= (D⁺ˣ - φ)/Δx; ~xperiodic ? D⁺ˣ[end,:,:] .= zero(T) : 0;
  D⁺ᶻ .= (D⁺ᶻ - φ)/Δz; ~zperiodic ? D⁺ᶻ[:,:,end] .= zero(T) : 0;
  D⁻ʸ .= (φ - D⁻ʸ)/Δy; ~yperiodic ? D⁻ʸ[:,1,:]   .= zero(T) : 0;
  D⁻ˣ .= (φ - D⁻ˣ)/Δx; ~xperiodic ? D⁻ˣ[1,:,:]   .= zero(T) : 0;
  D⁻ᶻ .= (φ - D⁻ᶻ)/Δz; ~zperiodic ? D⁻ᶻ[:,:,1]   .= zero(T) : 0;
  # Operators
  ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2 + max(D⁻ᶻ,0)^2 + min(D⁺ᶻ,0)^2)
  ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2 + max(D⁺ᶻ,0)^2 + min(D⁻ᶻ,0)^2)
  # Update
  φ_new .= @. φ - Δt*(max(vel,0)*∇⁺ + min(vel,0)*∇⁻ - vel)
  return φ_new
end

function evolve!(::FirstOrderStencil{3,T},φ,vel,Δt,Δx,isperiodic,caches) where T
  D⁺ᶻ, D⁺ʸ, D⁺ˣ, D⁻ᶻ, D⁻ʸ, D⁻ˣ, ∇⁺, ∇⁻=caches
  Δx, Δy, Δz = Δx
  xperiodic,yperiodic,zperiodic = isperiodic
  # Prepare shifted lsf
  circshift!(D⁺ʸ,φ,(0,-1,0)); circshift!(D⁻ʸ,φ,(0,1,0))
  circshift!(D⁺ˣ,φ,(-1,0,0)); circshift!(D⁻ˣ,φ,(1,0,0))
  circshift!(D⁺ᶻ,φ,(0,0,-1)); circshift!(D⁻ᶻ,φ,(0,0,1))
  # Forward (+) & Backward (-)
  D⁺ʸ .= (D⁺ʸ - φ)/Δy; ~yperiodic ? D⁺ʸ[:,end,:] .= zero(T) : 0;
  D⁺ˣ .= (D⁺ˣ - φ)/Δx; ~xperiodic ? D⁺ˣ[end,:,:] .= zero(T) : 0;
  D⁺ᶻ .= (D⁺ᶻ - φ)/Δz; ~zperiodic ? D⁺ᶻ[:,:,end] .= zero(T) : 0;
  D⁻ʸ .= (φ - D⁻ʸ)/Δy; ~yperiodic ? D⁻ʸ[:,1,:]   .= zero(T) : 0;
  D⁻ˣ .= (φ - D⁻ˣ)/Δx; ~xperiodic ? D⁻ˣ[1,:,:]   .= zero(T) : 0;
  D⁻ᶻ .= (φ - D⁻ᶻ)/Δz; ~zperiodic ? D⁻ᶻ[:,:,1]   .= zero(T) : 0;
  # Operators
  ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2 + max(D⁻ᶻ,0)^2 + min(D⁺ᶻ,0)^2)
  ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2 + max(D⁺ᶻ,0)^2 + min(D⁻ᶻ,0)^2)
  # Update
  φ .= @. φ - Δt*(max(vel,0)*∇⁺ + min(vel,0)*∇⁻)
  return φ
end

"""
    struct SecondOrderStencil{D,T} <: Stencil end

A second order upwind difference scheme based on Sethian's
Level Set Methods and Fast Marching Methods (2nd edition).

Experimental feature: `SecondOrderStencil` is not working as expected for periodic problems.
"""
struct SecondOrderStencil{D,T} <: Stencil
  function SecondOrderStencil(D::Integer,::Type{T}) where T<:Real
    @warn "Experimental feature: SecondOrderStencil is not working as expected for periodic problems."
    new{D,T}()
  end
end

function check_order(::SecondOrderStencil,order)
  @check order >= 2 "FirstOrderStencil requires reference element to have order >= 2" 
end

function allocate_caches(::SecondOrderStencil{2},φ,vel)
  D⁺ʸ = similar(φ); D⁺ˣ = similar(φ)
  D⁻ʸ = similar(φ); D⁻ˣ = similar(φ)
  D⁺ʸ⁺ʸ = similar(φ); D⁺ˣ⁺ˣ = similar(φ)
  D⁺ʸ⁻ʸ = similar(φ); D⁺ˣ⁻ˣ = similar(φ)
  D⁻ʸ⁻ʸ = similar(φ); D⁻ˣ⁻ˣ = similar(φ)
  ∇⁺  = similar(φ); ∇⁻  = similar(φ)
  return D⁺ʸ, D⁺ˣ, D⁻ʸ, D⁻ˣ, D⁺ʸ⁺ʸ, D⁺ˣ⁺ˣ, 
    D⁺ʸ⁻ʸ, D⁺ˣ⁻ˣ, D⁻ʸ⁻ʸ, D⁻ˣ⁻ˣ, ∇⁺, ∇⁻
end

function switch(x::M,y::M) where M <: AbstractFloat
  if x*y >= 0
    return (abs(x) <= abs(y)) ? x : y;     
  else 
    return 0
  end
end

function reinit!(::SecondOrderStencil{2,T},φ_new,φ,vel,Δt,Δx,isperiodic,caches) where T
  D⁺ʸ, D⁺ˣ, D⁻ʸ, D⁻ˣ, D⁺ʸ⁺ʸ, D⁺ˣ⁺ˣ, 
    D⁺ʸ⁻ʸ, D⁺ˣ⁻ˣ, D⁻ʸ⁻ʸ, D⁻ˣ⁻ˣ, ∇⁺, ∇⁻ = caches
  Δx, Δy = Δx
  xperiodic,yperiodic = isperiodic
  # Prepare shifted lsf
  circshift!(D⁺ʸ,φ,(0,-1)); circshift!(D⁻ʸ,φ,(0,1))
  circshift!(D⁺ˣ,φ,(-1,0)); circshift!(D⁻ˣ,φ,(1,0))
  circshift!(D⁺ʸ⁺ʸ,D⁺ʸ,(0,-1)); circshift!(D⁻ʸ⁻ʸ,D⁻ʸ,(0,1))
  circshift!(D⁺ˣ⁺ˣ,D⁺ˣ,(-1,0)); circshift!(D⁻ˣ⁻ˣ,D⁻ˣ,(1,0))
  # Sign approximation
  ∇⁺ .= @. (D⁺ʸ - D⁻ʸ)/(2Δy); ~yperiodic ? ∇⁺[:,[1,end]] .= zero(T) : 0;
  ∇⁻ .= @. (D⁺ˣ - D⁻ˣ)/(2Δx); ~xperiodic ? ∇⁻[[1,end],:] .= zero(T) : 0;
  ϵₛ = minimum((Δx,Δy))
  vel .= @. φ/sqrt(φ^2 + ϵₛ^2*(∇⁺^2+∇⁻^2))
  # Forward (+) & Backward (-)
  D⁺ʸ⁺ʸ .= @. (D⁺ʸ⁺ʸ - 2D⁺ʸ + φ)/Δy^2; ~yperiodic ? D⁺ʸ⁺ʸ[:,end-1:end] .= zero(T) : 0;
  D⁺ˣ⁺ˣ .= @. (D⁺ˣ⁺ˣ - 2D⁺ˣ - φ)/Δx^2; ~xperiodic ? D⁺ˣ⁺ˣ[end-1:end,:] .= zero(T) : 0;
  D⁺ʸ⁻ʸ .= @. (D⁺ʸ - 2φ + D⁻ʸ)/Δy^2; ~yperiodic ? D⁺ʸ⁻ʸ[:,[1,end]]     .= zero(T) : 0;
  D⁺ˣ⁻ˣ .= @. (D⁺ˣ - 2φ + D⁻ˣ)/Δx^2; ~xperiodic ? D⁺ˣ⁻ˣ[[1,end],:]     .= zero(T) : 0;
  D⁻ʸ⁻ʸ .= @. (φ - 2D⁻ʸ + D⁻ʸ⁻ʸ)/Δy^2; ~yperiodic ? D⁻ʸ⁻ʸ[:,1:2]       .= zero(T) : 0;
  D⁻ˣ⁻ˣ .= @. (φ - 2D⁻ˣ + D⁻ˣ⁻ˣ)/Δx^2; ~xperiodic ? D⁻ˣ⁻ˣ[1:2,:]       .= zero(T) : 0;
  D⁺ʸ .= @. (D⁺ʸ - φ)/Δy; ~yperiodic ? D⁺ʸ[:,end] .= zero(T) : 0;
  D⁺ˣ .= @. (D⁺ˣ - φ)/Δx; ~xperiodic ? D⁺ˣ[end,:] .= zero(T) : 0;
  D⁻ʸ .= @. (φ - D⁻ʸ)/Δy; ~yperiodic ? D⁻ʸ[:,1]   .= zero(T) : 0;
  D⁻ˣ .= @. (φ - D⁻ˣ)/Δx; ~xperiodic ? D⁻ˣ[1,:]   .= zero(T) : 0;
  # Operators
  D⁻ˣ .= @. D⁻ˣ + Δx/2*switch(D⁻ˣ⁻ˣ,D⁺ˣ⁻ˣ)
  D⁺ˣ .= @. D⁺ˣ - Δx/2*switch(D⁺ˣ⁺ˣ,D⁺ˣ⁻ˣ)
  D⁻ʸ .= @. D⁻ʸ + Δy/2*switch(D⁻ʸ⁻ʸ,D⁺ʸ⁻ʸ)
  D⁺ʸ .= @. D⁺ʸ - Δy/2*switch(D⁺ʸ⁺ʸ,D⁺ʸ⁻ʸ)
  ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2)
  ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2)
  # Update
  φ_new .= @. φ - Δt*(max(vel,0)*∇⁺ + min(vel,0)*∇⁻ - vel)
  return φ_new
end

function evolve!(::SecondOrderStencil{2,T},φ,vel,Δt,Δx,isperiodic,caches) where T
  D⁺ʸ, D⁺ˣ, D⁻ʸ, D⁻ˣ, D⁺ʸ⁺ʸ, D⁺ˣ⁺ˣ, 
    D⁺ʸ⁻ʸ, D⁺ˣ⁻ˣ, D⁻ʸ⁻ʸ, D⁻ˣ⁻ˣ, ∇⁺, ∇⁻ = caches
  Δx, Δy = Δx
  xperiodic,yperiodic = isperiodic
  # Prepare shifted lsf
  circshift!(D⁺ʸ,φ,(0,-1)); circshift!(D⁻ʸ,φ,(0,1))
  circshift!(D⁺ˣ,φ,(-1,0)); circshift!(D⁻ˣ,φ,(1,0))
  circshift!(D⁺ʸ⁺ʸ,D⁺ʸ,(0,-1)); circshift!(D⁻ʸ⁻ʸ,D⁻ʸ,(0,1))
  circshift!(D⁺ˣ⁺ˣ,D⁺ˣ,(-1,0)); circshift!(D⁻ˣ⁻ˣ,D⁻ˣ,(1,0))
  # Forward (+) & Backward (-)
  D⁺ʸ⁺ʸ .= @. (D⁺ʸ⁺ʸ - 2D⁺ʸ + φ)/Δy^2; ~yperiodic ? D⁺ʸ⁺ʸ[:,end-1:end] .= zero(T) : 0;
  D⁺ˣ⁺ˣ .= @. (D⁺ˣ⁺ˣ - 2D⁺ˣ - φ)/Δx^2; ~xperiodic ? D⁺ˣ⁺ˣ[end-1:end,:] .= zero(T) : 0;
  D⁺ʸ⁻ʸ .= @. (D⁺ʸ - 2φ + D⁻ʸ)/Δy^2; ~yperiodic ? D⁺ʸ⁻ʸ[:,[1,end]]     .= zero(T) : 0;
  D⁺ˣ⁻ˣ .= @. (D⁺ˣ - 2φ + D⁻ˣ)/Δx^2; ~xperiodic ? D⁺ˣ⁻ˣ[[1,end],:]     .= zero(T) : 0;
  D⁻ʸ⁻ʸ .= @. (φ - 2D⁻ʸ + D⁻ʸ⁻ʸ)/Δy^2; ~yperiodic ? D⁻ʸ⁻ʸ[:,1:2]       .= zero(T) : 0;
  D⁻ˣ⁻ˣ .= @. (φ - 2D⁻ˣ + D⁻ˣ⁻ˣ)/Δx^2; ~xperiodic ? D⁻ˣ⁻ˣ[1:2,:]       .= zero(T) : 0;
  D⁺ʸ .= @. (D⁺ʸ - φ)/Δy; ~yperiodic ? D⁺ʸ[:,end] .= zero(T) : 0;
  D⁺ˣ .= @. (D⁺ˣ - φ)/Δx; ~xperiodic ? D⁺ˣ[end,:] .= zero(T) : 0;
  D⁻ʸ .= @. (φ - D⁻ʸ)/Δy; ~yperiodic ? D⁻ʸ[:,1]   .= zero(T) : 0;
  D⁻ˣ .= @. (φ - D⁻ˣ)/Δx; ~xperiodic ? D⁻ˣ[1,:]   .= zero(T) : 0;
  # Operators
  D⁻ˣ .= @. D⁻ˣ + Δx/2*switch(D⁻ˣ⁻ˣ,D⁺ˣ⁻ˣ)
  D⁺ˣ .= @. D⁺ˣ - Δx/2*switch(D⁺ˣ⁺ˣ,D⁺ˣ⁻ˣ)
  D⁻ʸ .= @. D⁻ʸ + Δy/2*switch(D⁻ʸ⁻ʸ,D⁺ʸ⁻ʸ)
  D⁺ʸ .= @. D⁺ʸ - Δy/2*switch(D⁺ʸ⁺ʸ,D⁺ʸ⁻ʸ)
  ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2)
  ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2)
  # Update
  φ .= @. φ - Δt*(max(vel,0)*∇⁺ + min(vel,0)*∇⁻)
  return φ_new
end

function allocate_caches(::SecondOrderStencil{3},φ,vel)
  D⁺ᶻ = similar(φ); D⁺ʸ = similar(φ); D⁺ˣ = similar(φ)
  D⁻ᶻ = similar(φ); D⁻ʸ = similar(φ); D⁻ˣ = similar(φ)
  D⁺ᶻ⁺ᶻ = similar(φ); D⁺ʸ⁺ʸ = similar(φ); D⁺ˣ⁺ˣ = similar(φ)
  D⁺ᶻ⁻ᶻ = similar(φ); D⁺ʸ⁻ʸ = similar(φ); D⁺ˣ⁻ˣ = similar(φ)
  D⁻ᶻ⁻ᶻ = similar(φ); D⁻ʸ⁻ʸ = similar(φ); D⁻ˣ⁻ˣ = similar(φ)
  ∇⁺  = similar(φ); ∇⁻  = similar(φ)
  return D⁺ᶻ, D⁺ʸ, D⁺ˣ, D⁻ᶻ, D⁻ʸ, D⁻ˣ, D⁺ᶻ⁺ᶻ, D⁺ʸ⁺ʸ, D⁺ˣ⁺ˣ, 
    D⁺ᶻ⁻ᶻ, D⁺ʸ⁻ʸ, D⁺ˣ⁻ˣ, D⁻ᶻ⁻ᶻ, D⁻ʸ⁻ʸ, D⁻ˣ⁻ˣ, ∇⁺, ∇⁻
end

function reinit!(::SecondOrderStencil{3,T},φ_new,φ,vel,Δt,Δx,isperiodic,caches) where T
  D⁺ᶻ, D⁺ʸ, D⁺ˣ, D⁻ᶻ, D⁻ʸ, D⁻ˣ, D⁺ᶻ⁺ᶻ, D⁺ʸ⁺ʸ, D⁺ˣ⁺ˣ, 
    D⁺ᶻ⁻ᶻ, D⁺ʸ⁻ʸ, D⁺ˣ⁻ˣ, D⁻ᶻ⁻ᶻ, D⁻ʸ⁻ʸ, D⁻ˣ⁻ˣ, ∇⁺, ∇⁻ = caches
  Δx, Δy, Δz = Δx
  xperiodic,yperiodic,zperiodic = isperiodic
  # Prepare shifted lsf
  circshift!(D⁺ʸ,φ,(0,-1,0)); circshift!(D⁻ʸ,φ,(0,1,0))
  circshift!(D⁺ˣ,φ,(-1,0,0)); circshift!(D⁻ˣ,φ,(1,0,0))
  circshift!(D⁺ᶻ,φ,(0,0,-1)); circshift!(D⁻ᶻ,φ,(0,0,1))
  circshift!(D⁺ʸ⁺ʸ,D⁺ʸ,(0,-1,0)); circshift!(D⁻ʸ⁻ʸ,D⁻ʸ,(0,1,0))
  circshift!(D⁺ˣ⁺ˣ,D⁺ˣ,(-1,0,0)); circshift!(D⁻ˣ⁻ˣ,D⁻ˣ,(1,0,0))
  circshift!(D⁺ᶻ⁺ᶻ,D⁺ˣ,(0,0,-1)); circshift!(D⁻ᶻ⁻ᶻ,D⁻ˣ,(0,0,1))
  # Sign approximation
  ∇⁺ .= @. (D⁺ʸ - D⁻ʸ)/(2Δy); ~yperiodic ? ∇⁺[:,[1,end],:] .= zero(T) : 0;
  ∇⁻ .= @. (D⁺ˣ - D⁻ˣ)/(2Δx); ~xperiodic ? ∇⁻[[1,end],:,:] .= zero(T) : 0;
  ∇⁺ .= @. ∇⁺^2+∇⁻^2 # |∇φ|² (partially computed)
  ∇⁻ .= @. (D⁺ᶻ - D⁻ᶻ)/(2Δz); ~xperiodic ? ∇⁻[:,:,[1,end]] .= zero(T) : 0;
  ϵₛ = minimum((Δx,Δy,Δz))
  vel .= @. φ/sqrt(φ^2 + ϵₛ^2*(∇⁺+∇⁻^2))
  # Forward (+) & Backward (-)
  D⁺ᶻ⁺ᶻ .= @. (D⁺ᶻ⁺ᶻ - 2D⁺ᶻ + φ)/Δz^2; ~yperiodic ? D⁺ᶻ⁺ᶻ[:,:,end-1:end] .= zero(T) : 0;
  D⁺ʸ⁺ʸ .= @. (D⁺ʸ⁺ʸ - 2D⁺ʸ + φ)/Δy^2; ~yperiodic ? D⁺ʸ⁺ʸ[:,end-1:end,:] .= zero(T) : 0;
  D⁺ˣ⁺ˣ .= @. (D⁺ˣ⁺ˣ - 2D⁺ˣ - φ)/Δx^2; ~xperiodic ? D⁺ˣ⁺ˣ[end-1:end,:,:] .= zero(T) : 0;
  D⁺ᶻ⁻ᶻ .= @. (D⁺ᶻ - 2φ + D⁻ᶻ)/Δz^2; ~yperiodic ? D⁺ᶻ⁻ᶻ[:,:,[1,end]]     .= zero(T) : 0;
  D⁺ʸ⁻ʸ .= @. (D⁺ʸ - 2φ + D⁻ʸ)/Δy^2; ~yperiodic ? D⁺ʸ⁻ʸ[:,[1,end],:]     .= zero(T) : 0;
  D⁺ˣ⁻ˣ .= @. (D⁺ˣ - 2φ + D⁻ˣ)/Δx^2; ~xperiodic ? D⁺ˣ⁻ˣ[[1,end],:,:]     .= zero(T) : 0;
  D⁻ᶻ⁻ᶻ .= @. (φ - 2D⁻ᶻ + D⁻ᶻ⁻ᶻ)/Δz^2; ~yperiodic ? D⁻ᶻ⁻ᶻ[:,:,1:2]       .= zero(T) : 0;
  D⁻ʸ⁻ʸ .= @. (φ - 2D⁻ʸ + D⁻ʸ⁻ʸ)/Δy^2; ~yperiodic ? D⁻ʸ⁻ʸ[:,1:2,:]       .= zero(T) : 0;
  D⁻ˣ⁻ˣ .= @. (φ - 2D⁻ˣ + D⁻ˣ⁻ˣ)/Δx^2; ~xperiodic ? D⁻ˣ⁻ˣ[1:2,:,:]       .= zero(T) : 0;
  D⁺ʸ .= (D⁺ʸ - φ)/Δy; ~yperiodic ? D⁺ʸ[:,end,:] .= zero(T) : 0;
  D⁺ˣ .= (D⁺ˣ - φ)/Δx; ~xperiodic ? D⁺ˣ[end,:,:] .= zero(T) : 0;
  D⁺ᶻ .= (D⁺ᶻ - φ)/Δz; ~zperiodic ? D⁺ᶻ[:,:,end] .= zero(T) : 0;
  D⁻ʸ .= (φ - D⁻ʸ)/Δy; ~yperiodic ? D⁻ʸ[:,1,:]   .= zero(T) : 0;
  D⁻ˣ .= (φ - D⁻ˣ)/Δx; ~xperiodic ? D⁻ˣ[1,:,:]   .= zero(T) : 0;
  D⁻ᶻ .= (φ - D⁻ᶻ)/Δz; ~zperiodic ? D⁻ᶻ[:,:,1]   .= zero(T) : 0;
  # Operators
  D⁻ˣ .= @. D⁻ˣ + Δx/2*switch(D⁻ˣ⁻ˣ,D⁺ˣ⁻ˣ)
  D⁺ˣ .= @. D⁺ˣ - Δx/2*switch(D⁺ˣ⁺ˣ,D⁺ˣ⁻ˣ)
  D⁻ʸ .= @. D⁻ʸ + Δy/2*switch(D⁻ʸ⁻ʸ,D⁺ʸ⁻ʸ)
  D⁺ʸ .= @. D⁺ʸ - Δy/2*switch(D⁺ʸ⁺ʸ,D⁺ʸ⁻ʸ)
  D⁻ᶻ .= @. D⁻ᶻ + Δz/2*switch(D⁻ᶻ⁻ᶻ,D⁺ᶻ⁻ᶻ)
  D⁺ᶻ .= @. D⁺ᶻ - Δz/2*switch(D⁺ᶻ⁺ᶻ,D⁺ᶻ⁻ᶻ)
  ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2 + max(D⁻ᶻ,0)^2 + min(D⁺ᶻ,0)^2)
  ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2 + max(D⁺ᶻ,0)^2 + min(D⁻ᶻ,0)^2)
  # Update
  φ_new .= @. φ - Δt*(max(vel,0)*∇⁺ + min(vel,0)*∇⁻ - vel)
  return φ_new
end

function evolve!(::SecondOrderStencil{3,T},φ_new,φ,vel,Δt,Δx,isperiodic,caches) where T
  D⁺ᶻ, D⁺ʸ, D⁺ˣ, D⁻ᶻ, D⁻ʸ, D⁻ˣ, D⁺ᶻ⁺ᶻ, D⁺ʸ⁺ʸ, D⁺ˣ⁺ˣ, 
    D⁺ᶻ⁻ᶻ, D⁺ʸ⁻ʸ, D⁺ˣ⁻ˣ, D⁻ᶻ⁻ᶻ, D⁻ʸ⁻ʸ, D⁻ˣ⁻ˣ, ∇⁺, ∇⁻ = caches
  Δx, Δy, Δz = Δx
  xperiodic,yperiodic,zperiodic = isperiodic
  # Prepare shifted lsf
  circshift!(D⁺ʸ,φ,(0,-1,0)); circshift!(D⁻ʸ,φ,(0,1,0))
  circshift!(D⁺ˣ,φ,(-1,0,0)); circshift!(D⁻ˣ,φ,(1,0,0))
  circshift!(D⁺ᶻ,φ,(0,0,-1)); circshift!(D⁻ᶻ,φ,(0,0,1))
  circshift!(D⁺ʸ⁺ʸ,D⁺ʸ,(0,-1,0)); circshift!(D⁻ʸ⁻ʸ,D⁻ʸ,(0,1,0))
  circshift!(D⁺ˣ⁺ˣ,D⁺ˣ,(-1,0,0)); circshift!(D⁻ˣ⁻ˣ,D⁻ˣ,(1,0,0))
  circshift!(D⁺ᶻ⁺ᶻ,D⁺ˣ,(0,0,-1)); circshift!(D⁻ᶻ⁻ᶻ,D⁻ˣ,(0,0,1))
  # Forward (+) & Backward (-)
  D⁺ᶻ⁺ᶻ .= @. (D⁺ᶻ⁺ᶻ - 2D⁺ᶻ + φ)/Δz^2; ~yperiodic ? D⁺ᶻ⁺ᶻ[:,:,end-1:end] .= zero(T) : 0;
  D⁺ʸ⁺ʸ .= @. (D⁺ʸ⁺ʸ - 2D⁺ʸ + φ)/Δy^2; ~yperiodic ? D⁺ʸ⁺ʸ[:,end-1:end,:] .= zero(T) : 0;
  D⁺ˣ⁺ˣ .= @. (D⁺ˣ⁺ˣ - 2D⁺ˣ - φ)/Δx^2; ~xperiodic ? D⁺ˣ⁺ˣ[end-1:end,:,:] .= zero(T) : 0;
  D⁺ᶻ⁻ᶻ .= @. (D⁺ᶻ - 2φ + D⁻ᶻ)/Δz^2; ~yperiodic ? D⁺ᶻ⁻ᶻ[:,:,[1,end]]     .= zero(T) : 0;
  D⁺ʸ⁻ʸ .= @. (D⁺ʸ - 2φ + D⁻ʸ)/Δy^2; ~yperiodic ? D⁺ʸ⁻ʸ[:,[1,end],:]     .= zero(T) : 0;
  D⁺ˣ⁻ˣ .= @. (D⁺ˣ - 2φ + D⁻ˣ)/Δx^2; ~xperiodic ? D⁺ˣ⁻ˣ[[1,end],:,:]     .= zero(T) : 0;
  D⁻ᶻ⁻ᶻ .= @. (φ - 2D⁻ᶻ + D⁻ᶻ⁻ᶻ)/Δz^2; ~yperiodic ? D⁻ᶻ⁻ᶻ[:,:,1:2]       .= zero(T) : 0;
  D⁻ʸ⁻ʸ .= @. (φ - 2D⁻ʸ + D⁻ʸ⁻ʸ)/Δy^2; ~yperiodic ? D⁻ʸ⁻ʸ[:,1:2,:]       .= zero(T) : 0;
  D⁻ˣ⁻ˣ .= @. (φ - 2D⁻ˣ + D⁻ˣ⁻ˣ)/Δx^2; ~xperiodic ? D⁻ˣ⁻ˣ[1:2,:,:]       .= zero(T) : 0;
  D⁺ʸ .= (D⁺ʸ - φ)/Δy; ~yperiodic ? D⁺ʸ[:,end,:] .= zero(T) : 0;
  D⁺ˣ .= (D⁺ˣ - φ)/Δx; ~xperiodic ? D⁺ˣ[end,:,:] .= zero(T) : 0;
  D⁺ᶻ .= (D⁺ᶻ - φ)/Δz; ~zperiodic ? D⁺ᶻ[:,:,end] .= zero(T) : 0;
  D⁻ʸ .= (φ - D⁻ʸ)/Δy; ~yperiodic ? D⁻ʸ[:,1,:]   .= zero(T) : 0;
  D⁻ˣ .= (φ - D⁻ˣ)/Δx; ~xperiodic ? D⁻ˣ[1,:,:]   .= zero(T) : 0;
  D⁻ᶻ .= (φ - D⁻ᶻ)/Δz; ~zperiodic ? D⁻ᶻ[:,:,1]   .= zero(T) : 0;
  # Operators
  D⁻ˣ .= @. D⁻ˣ + Δx/2*switch(D⁻ˣ⁻ˣ,D⁺ˣ⁻ˣ)
  D⁺ˣ .= @. D⁺ˣ - Δx/2*switch(D⁺ˣ⁺ˣ,D⁺ˣ⁻ˣ)
  D⁻ʸ .= @. D⁻ʸ + Δy/2*switch(D⁻ʸ⁻ʸ,D⁺ʸ⁻ʸ)
  D⁺ʸ .= @. D⁺ʸ - Δy/2*switch(D⁺ʸ⁺ʸ,D⁺ʸ⁻ʸ)
  D⁻ᶻ .= @. D⁻ᶻ + Δz/2*switch(D⁻ᶻ⁻ᶻ,D⁺ᶻ⁻ᶻ)
  D⁺ᶻ .= @. D⁺ᶻ - Δz/2*switch(D⁺ᶻ⁺ᶻ,D⁺ᶻ⁻ᶻ)
  ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2 + max(D⁻ᶻ,0)^2 + min(D⁺ᶻ,0)^2)
  ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2 + max(D⁺ᶻ,0)^2 + min(D⁻ᶻ,0)^2)
  # Update
  φ .= @. φ - Δt*(max(vel,0)*∇⁺ + min(vel,0)*∇⁻)
  return φ_new
end