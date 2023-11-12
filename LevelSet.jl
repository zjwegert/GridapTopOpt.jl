# Generate initial LSF
gen_lsf(ξ,a) = x::VectorValue -> -1/4*prod(cos.(get_array(ξ*pi*x))) - a/4

# struct LevelSetUpdate{N,M}
#     model
#     fe_space
#     γ
#     max_steps
#     cache

#     function LevelSetUpdate{M,N}() where {M,N}
#         new{M,N}()
#     end

# end

# struct VelocityExtension
#     U
#     V
#     assem
#     dΩ
#     K
#     cache
# end

# struct LevelSetCache{N,M} 

#     φ_tmp

# end

## H-J and reinit
"""
    Single step for first order upwind method for H-J equation (2D)
"""
function advect!(φ::T,V::T,Δ::NTuple{2,M},Δt::M,caches) where {M,T<:Array{M,2}}
    D⁺ʸ,D⁺ˣ,D⁻ʸ,D⁻ˣ,∇⁺,∇⁻=caches
    Δx,Δy = Δ
    # Prepare shifted lsf
    circshift!(D⁺ʸ,φ,(0,-1)); circshift!(D⁻ʸ,φ,(0,1)); 
    circshift!(D⁺ˣ,φ,(-1,0)); circshift!(D⁻ˣ,φ,(1,0)); 
    # Forward (+) & Backward (-)
    D⁺ʸ .= @. (D⁺ʸ - φ)/Δy;
    D⁺ˣ .= @. (D⁺ˣ - φ)/Δx;
    D⁻ʸ .= @. (φ - D⁻ʸ)/Δy;
    D⁻ˣ .= @. (φ - D⁻ˣ)/Δx;
    # Check for boundaries with ghost nodes
    D⁺ʸ[:,end] .= zero(M);
    D⁺ˣ[end,:] .= zero(M);
    D⁻ʸ[:,1] .= zero(M);
    D⁻ˣ[1,:] .= zero(M);
    # Operators
    ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2);
    ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2);
    # Update
    φ .= @. φ - Δt*(max(V,0)*∇⁺ + min(V,0)*∇⁻); 
    return nothing
end

"""
    Single step for first order upwind method for reinitialisation equation (2D)
"""
function reinit!(φ::T,φ_tmp::T,S::T,Δ::NTuple{2,M},Δt::M,caches) where {M,T<:Array{M,2}}
    D⁺ʸ,D⁺ˣ,D⁻ʸ,D⁻ˣ,∇⁺,∇⁻=caches
    Δx,Δy = Δ
    # Prepare shifted lsf
    circshift!(D⁺ʸ,φ,(0,-1)); circshift!(D⁻ʸ,φ,(0,1)); 
    circshift!(D⁺ˣ,φ,(-1,0)); circshift!(D⁻ˣ,φ,(1,0)); 
    # Forward (+) & Backward (-)
    D⁺ʸ .= @. (D⁺ʸ - φ)/Δy;
    D⁺ˣ .= @. (D⁺ˣ - φ)/Δx;
    D⁻ʸ .= @. (φ - D⁻ʸ)/Δy;
    D⁻ˣ .= @. (φ - D⁻ˣ)/Δx;
    # Check for boundaries with ghost nodes
    D⁺ʸ[:,end] .= zero(M);
    D⁺ˣ[end,:] .= zero(M);
    D⁻ʸ[:,1] .= zero(M);
    D⁻ˣ[1,:] .= zero(M);
    # Operators
    ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2);
    ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2);
    # Update
    φ_tmp .= @. φ - Δt*(max(S,0)*∇⁺ + min(S,0)*∇⁻ - S); 
    return nothing
end

"""
    Single step for first order upwind method for H-J equation (3D)
"""
function advect!(φ::T,V::T,Δ::NTuple{3,M},Δt::M,caches) where {M,T<:Array{M,3}}
    D⁺ʸ,D⁺ˣ,D⁻ʸ,D⁻ˣ,∇⁺,∇⁻=caches
    Δx,Δy,Δz = Δ 
    # Prepare shifted lsf
    circshift!(D⁺ʸ,φ,(0,-1,0)); circshift!(D⁻ʸ,φ,(0,1,0)); 
    circshift!(D⁺ˣ,φ,(-1,0,0)); circshift!(D⁻ˣ,φ,(1,0,0));
    circshift!(D⁺ᶻ,φ,(0,0,-1)); circshift!(D⁻ᶻ,φ,(0,0,1));
    # Forward (+) & Backward (-)
    D⁺ʸ .= (D⁺ʸ - φ)/Δy;
    D⁺ˣ .= (D⁺ˣ - φ)/Δx;
    D⁺ᶻ .= (D⁺ᶻ - φ)/Δz;
    D⁻ʸ .= (φ - D⁻ʸ)/Δy;
    D⁻ˣ .= (φ - D⁻ˣ)/Δx;
    D⁻ᶻ .= (φ - D⁻ᶻ)/Δz;
    # Check for boundaries with ghost nodes
    D⁺ʸ[:,end,:] .= zero(M);
    D⁺ˣ[end,:,:] .= zero(M);
    D⁺ᶻ[:,:,end] .= zero(M);
    D⁻ʸ[:,1,:] .= zero(M);
    D⁻ˣ[1,:,:] .= zero(M);
    D⁻ᶻ[:,:,1] .= zero(M);
    # Operators
    ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2 + max(D⁻ᶻ,0)^2 + min(D⁺ᶻ,0)^2);
    ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2 + max(D⁺ᶻ,0)^2 + min(D⁻ᶻ,0)^2);
    # Update
    φ .= @. φ - Δt*(max(V,0)*∇⁺ + min(V,0)*∇⁻); 
    return nothing
end

"""
    Single step for first order upwind method for reinitialisation equation (3D)
"""
function reinit!(φ::T,φ_tmp::T,S::T,Δ::NTuple{3,M},Δt::M) where {M,T<:Array{M,3}}
    D⁺ʸ,D⁺ˣ,D⁻ʸ,D⁻ˣ,∇⁺,∇⁻=caches
    Δx,Δy,Δz = Δ 
    # Prepare shifted lsf
    circshift!(D⁺ʸ,φ,(0,-1,0)); circshift!(D⁻ʸ,φ,(0,1,0)); 
    circshift!(D⁺ˣ,φ,(-1,0,0)); circshift!(D⁻ˣ,φ,(1,0,0));
    circshift!(D⁺ᶻ,φ,(0,0,-1)); circshift!(D⁻ᶻ,φ,(0,0,1));
    # Forward (+) & Backward (-)
    D⁺ʸ .= (D⁺ʸ - φ)/Δy;
    D⁺ˣ .= (D⁺ˣ - φ)/Δx;
    D⁺ᶻ .= (D⁺ᶻ - φ)/Δz;
    D⁻ʸ .= (φ - D⁻ʸ)/Δy;
    D⁻ˣ .= (φ - D⁻ˣ)/Δx;
    D⁻ᶻ .= (φ - D⁻ᶻ)/Δz;
    # Check for boundaries with ghost nodes
    D⁺ʸ[:,end,:] .= zero(M);
    D⁺ˣ[end,:,:] .= zero(M);
    D⁺ᶻ[:,:,end] .= zero(M);
    D⁻ʸ[:,1,:] .= zero(M);
    D⁻ˣ[1,:,:] .= zero(M);
    D⁻ᶻ[:,:,1] .= zero(M);
    # Operators
    ∇⁺ .= @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2 + max(D⁻ᶻ,0)^2 + min(D⁺ᶻ,0)^2);
    ∇⁻ .= @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2 + max(D⁺ᶻ,0)^2 + min(D⁻ᶻ,0)^2);
    # Update
    φ_tmp .= @. φ - Δt*(max(S,0)*∇⁺ + min(S,0)*∇⁻ - S); 
    return nothing
end

function advect!(φ::T,V::T,model::D,Δ::NTuple{N,M},γ::M,max_steps::Int) where {N,M,
        T<:PVector{Vector{M}},D<:DistributedDiscreteModel}
    ## CFL Condition (requires γ≤1.0)
    Δt = γ*min(Δ...)/(eps(M)^2+infnorm(V))
    # Find part size and location of boundaries with ghost nodes
    part_size,g_loc = map(ghost_values(φ),local_views(model)) do φ_ghost,model
        part_size = size(model.grid.node_coords)
        g_loc = find_part_ghost_boundaries(φ_ghost,part_size)
        part_size,g_loc
    end |> tuple_of_arrays
    for _ ∈ Base.OneTo(max_steps)
        # Apply operations across partitions
        map(local_views(φ),local_views(V),part_size,g_loc) do φ,V,part_size,g_loc
            # Step of 1st order upwind H-J evolution equation
            advect_step!(reshape(φ,part_size),reshape(V,part_size),g_loc,Δ,Δt)
        end
        # Update ghost nodes
        consistent!(φ) |> fetch
    end
end

function reinit!(φ::T,model::D,Δ::NTuple{N,M},γ::M,max_steps::Int,tol::M,caches) where {N,M,
        T<:PVector{Vector{M}},D<:DistributedDiscreteModel}
    # Initalise φ_tmp
    φ_tmp = zero(φ) # <- cache me
    # Compute approx sign function S
    ∏Δ = prod(Δ);
    S = zero(φ) # <- cache me
    map(local_views(S),local_views(φ)) do S,φ
        S .= @. φ/sqrt(φ^2 + ∏Δ)
    end
    ## CFL Condition (requires γ≤0.5)
    Δt = γ*min(Δ...)
    part_size = map(local_views(model)) do model # <- in setup
        get_cartesian_descriptor(model).partition .+ 1
    end
    # Apply operations across partitions
    for _ ∈ Base.OneTo(max_steps)
        # Step of 1st order upwind reinitialisation equation
        map(local_views(φ),local_views(φ_tmp),local_views(S),part_size) do φ,φ_tmp,S,part_size
            reinit_step!(reshape(φ,part_size),reshape(φ_tmp,part_size),reshape(S,part_size),Δ,Δt,caches)
        end
        # Update ghost nodes
        consistent!(φ_tmp) |> fetch
        # Check convergence |φ-φ_tmp|<ε
        if infnorm(φ-φ_tmp)<tol
            break
        end
        # Update φ
        map(local_views(φ),local_views(φ_tmp)) do φ,φ_tmp
            φ .= φ_tmp
        end
        consistent!(φ) |> fetch
    end
end