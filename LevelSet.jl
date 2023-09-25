# Generate initial LSF
gen_lsf(ξ,a) = x::VectorValue -> -1/4*prod(cos.(get_array(ξ*pi*x))) - a/4

## H-J and reinit
"""
    Single step for first order upwind method for H-J equation (2D)
"""
function advect_step!(φ::T,V::T,g_loc::NTuple{4,Bool},Δ::NTuple{2,M},Δt::M) where {M,T<:Array{M,2}}
    (X⁻,X⁺,Y⁻,Y⁺) = g_loc
    Δx,Δy = Δ 
    # Prepare shifted lsf
    φ⁺ʸ = circshift(φ,(0,-1)); φ⁻ʸ = circshift(φ,(0,1)); 
    φ⁺ˣ = circshift(φ,(-1,0)); φ⁻ˣ = circshift(φ,(1,0)); 
    # Forward (+) & Backward (-)
    D⁺ʸ = @. (φ⁺ʸ - φ)/Δy;
    D⁺ˣ = @. (φ⁺ˣ - φ)/Δx;
    D⁻ʸ = @. (φ - φ⁻ʸ)/Δy;
    D⁻ˣ = @. (φ - φ⁻ˣ)/Δx;
    # Check for boundaries with ghost nodes
    (~Y⁺) ? D⁺ʸ[:,end] .= zero(M) : 0;
    (~X⁺) ? D⁺ˣ[end,:] .= zero(M) : 0;
    (~Y⁻) ? D⁻ʸ[:,1] .= zero(M) : 0;
    (~X⁻) ? D⁻ˣ[1,:] .= zero(M) : 0;
    # Operators
    ∇⁺ = @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2);
    ∇⁻ = @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2);
    # Update
    φ .= @. φ - Δt*(max(V,0)*∇⁺ + min(V,0)*∇⁻); 
    return nothing
end

"""
    Single step for first order upwind method for reinitialisation equation (2D)
"""
function reinit_step!(φ::T,φ_tmp::T,S::T,g_loc::NTuple{4,Bool},Δ::NTuple{2,M},Δt::M) where {M,T<:Array{M,2}}
    (X⁻,X⁺,Y⁻,Y⁺) = g_loc
    Δx,Δy = Δ 
    # Prepare shifted lsf
    φ⁺ʸ = circshift(φ,(0,-1)); φ⁻ʸ = circshift(φ,(0,1)); 
    φ⁺ˣ = circshift(φ,(-1,0)); φ⁻ˣ = circshift(φ,(1,0)); 
    # Forward (+) & Backward (-)
    D⁺ʸ = @. (φ⁺ʸ - φ)/Δy;
    D⁺ˣ = @. (φ⁺ˣ - φ)/Δx;
    D⁻ʸ = @. (φ - φ⁻ʸ)/Δy;
    D⁻ˣ = @. (φ - φ⁻ˣ)/Δx;
    # Check for boundaries with ghost nodes
    (~Y⁺) ? D⁺ʸ[:,end] .= zero(M) : 0;
    (~X⁺) ? D⁺ˣ[end,:] .= zero(M) : 0;
    (~Y⁻) ? D⁻ʸ[:,1] .= zero(M) : 0;
    (~X⁻) ? D⁻ˣ[1,:] .= zero(M) : 0;
    # Operators
    ∇⁺ = @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2);
    ∇⁻ = @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2);
    # Update
    φ_tmp .= @. φ - Δt*(max(S,0)*∇⁺ + min(S,0)*∇⁻ - S); 
    return nothing
end

"""
    Single step for first order upwind method for H-J equation (3D)
"""
function advect_step!(φ::T,V::T,g_loc::NTuple{6,Bool},Δ::NTuple{3,M},Δt::M) where {M,T<:Array{M,3}}
    (X⁻,X⁺,Y⁻,Y⁺,Z⁻,Z⁺) = g_loc
    Δx,Δy,Δz = Δ 
    # Prepare shifted lsf
    φ⁺ʸ = circshift(φ,(0,-1,0)); φ⁻ʸ = circshift(φ,(0,1,0)); 
    φ⁺ˣ = circshift(φ,(-1,0,0)); φ⁻ˣ = circshift(φ,(1,0,0));
    φ⁺ᶻ = circshift(φ,(0,0,-1)); φ⁻ᶻ = circshift(φ,(0,0,1));
    # Forward (+) & Backward (-)
    D⁺ʸ = (φ⁺ʸ - φ)/Δy;
    D⁺ˣ = (φ⁺ˣ - φ)/Δx;
    D⁺ᶻ = (φ⁺ᶻ - φ)/Δz;
    D⁻ʸ = (φ - φ⁻ʸ)/Δy;
    D⁻ˣ = (φ - φ⁻ˣ)/Δx;
    D⁻ᶻ = (φ - φ⁻ᶻ)/Δz;
    # Check for boundaries with ghost nodes
    (~Y⁺) ? D⁺ʸ[:,end,:] .= zero(M) : 0;
    (~X⁺) ? D⁺ˣ[end,:,:] .= zero(M) : 0;
    (~Z⁺) ? D⁺ᶻ[:,:,end] .= zero(M) : 0;
    (~Y⁻) ? D⁻ʸ[:,1,:] .= zero(M) : 0;
    (~X⁻) ? D⁻ˣ[1,:,:] .= zero(M) : 0;
    (~Z⁻) ? D⁻ᶻ[:,:,1] .= zero(M) : 0;
    # Operators
    ∇⁺ = @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2 + max(D⁻ᶻ,0)^2 + min(D⁺ᶻ,0)^2);
    ∇⁻ = @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2 + max(D⁺ᶻ,0)^2 + min(D⁻ᶻ,0)^2);
    # Update
    φ .= @. φ - Δt*(max(V,0)*∇⁺ + min(V,0)*∇⁻); 
    return nothing
end

"""
    Single step for first order upwind method for reinitialisation equation (3D)
"""
function reinit_step!(φ::T,φ_tmp::T,S::T,g_loc::NTuple{6,Bool},Δ::NTuple{3,M},Δt::M) where {M,T<:Array{M,3}}
    (X⁻,X⁺,Y⁻,Y⁺,Z⁻,Z⁺) = g_loc
    Δx,Δy,Δz = Δ 
    # Prepare shifted lsf
    φ⁺ʸ = circshift(φ,(0,-1,0)); φ⁻ʸ = circshift(φ,(0,1,0)); 
    φ⁺ˣ = circshift(φ,(-1,0,0)); φ⁻ˣ = circshift(φ,(1,0,0));
    φ⁺ᶻ = circshift(φ,(0,0,-1)); φ⁻ᶻ = circshift(φ,(0,0,1));
    # Forward (+) & Backward (-)
    D⁺ʸ = (φ⁺ʸ - φ)/Δy;
    D⁺ˣ = (φ⁺ˣ - φ)/Δx;
    D⁺ᶻ = (φ⁺ᶻ - φ)/Δz;
    D⁻ʸ = (φ - φ⁻ʸ)/Δy;
    D⁻ˣ = (φ - φ⁻ˣ)/Δx;
    D⁻ᶻ = (φ - φ⁻ᶻ)/Δz;
    # Check for boundaries with ghost nodes
    (~Y⁺) ? D⁺ʸ[:,end,:] .= zero(M) : 0;
    (~X⁺) ? D⁺ˣ[end,:,:] .= zero(M) : 0;
    (~Z⁺) ? D⁺ᶻ[:,:,end] .= zero(M) : 0;
    (~Y⁻) ? D⁻ʸ[:,1,:] .= zero(M) : 0;
    (~X⁻) ? D⁻ˣ[1,:,:] .= zero(M) : 0;
    (~Z⁻) ? D⁻ᶻ[:,:,1] .= zero(M) : 0;
    # Operators
    ∇⁺ = @. sqrt(max(D⁻ʸ,0)^2 + min(D⁺ʸ,0)^2 + max(D⁻ˣ,0)^2 + min(D⁺ˣ,0)^2 + max(D⁻ᶻ,0)^2 + min(D⁺ᶻ,0)^2);
    ∇⁻ = @. sqrt(max(D⁺ʸ,0)^2 + min(D⁻ʸ,0)^2 + max(D⁺ˣ,0)^2 + min(D⁻ˣ,0)^2 + max(D⁺ᶻ,0)^2 + min(D⁻ᶻ,0)^2);
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

function reinit!(φ::T,model::D,Δ::NTuple{N,M},γ::M,max_steps::Int,tol::M) where {N,M,
        T<:PVector{Vector{M}},D<:DistributedDiscreteModel}
    # Initalise φ_tmp
    φ_tmp = zero(φ)
    # Compute approx sign function S
    ∏Δ = prod(Δ);
    S = zero(φ)
    map(local_views(S),local_views(φ)) do S,φ
        S .= @. φ/sqrt(φ^2 + ∏Δ)
    end
    consistent!(S) |> fetch
    ## CFL Condition (requires γ≤0.5)
    Δt = γ*min(Δ...)#/(eps(M)^2+infnorm(S)) # As inform(S)=1
    # Find part size and location of boundaries with ghost nodes
    part_size,g_loc = map(ghost_values(φ),local_views(model)) do φ_ghost,model
            part_size = size(model.grid.node_coords)
            g_loc = find_part_ghost_boundaries(φ_ghost,part_size)
            part_size,g_loc
    end |> tuple_of_arrays
    # Apply operations across partitions
    for _ ∈ Base.OneTo(max_steps)
        # Step of 1st order upwind reinitialisation equation
        map(local_views(φ),local_views(φ_tmp),local_views(S),part_size,g_loc) do φ,φ_tmp,S,part_size,g_loc
            reinit_step!(reshape(φ,part_size),reshape(φ_tmp,part_size),reshape(S,part_size),g_loc,Δ,Δt)
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

# Detect location of ghost cells (hacky but works...)
function find_part_ghost_boundaries(φ_ghost,part_size::NTuple{2,M}) where M<:Integer
    ghost_indxs = φ_ghost.indices[1]
    part_indxs = collect(reshape(Base.OneTo(prod(part_size)),part_size))
    part_indxs[2:end-1,2:end-1] .= -1
    # Check if ghost boundary lies on each side of array
    X⁻ = length(part_indxs[1,2:end-1] ∩ ghost_indxs) == part_size[2]-2
    X⁺ = length(part_indxs[end,2:end-1] ∩ ghost_indxs) == part_size[2]-2
    Y⁻ = length(part_indxs[2:end-1,1] ∩ ghost_indxs) == part_size[1]-2
    Y⁺ = length(part_indxs[2:end-1,end] ∩ ghost_indxs) == part_size[1]-2
    return (X⁻,X⁺,Y⁻,Y⁺)
end

function find_part_ghost_boundaries(φ_ghost,part_size::NTuple{3,M}) where M<:Integer
    ghost_indxs = φ_ghost.indices[1]
    part_indxs = collect(reshape(Base.OneTo(prod(part_size)),part_size))
    part_indxs[2:end-1,2:end-1,2:end-1] .= -1
    # Check if ghost boundary lies on each side of array
    X⁻ = length(part_indxs[1,2:end-1,2:end-1] ∩ ghost_indxs) == part_size[2]*part_size[3]-2part_size[2]-2part_size[3]+4
    X⁺ = length(part_indxs[end,2:end-1,2:end-1] ∩ ghost_indxs) == part_size[2]*part_size[3]-2part_size[2]-2part_size[3]+4
    Y⁻ = length(part_indxs[2:end-1,1,2:end-1] ∩ ghost_indxs) == part_size[1]*part_size[3]-2part_size[1]-2part_size[3]+4
    Y⁺ = length(part_indxs[2:end-1,end,2:end-1] ∩ ghost_indxs) == part_size[1]*part_size[3]-2part_size[1]-2part_size[3]+4
    Z⁻ = length(part_indxs[2:end-1,2:end-1,1] ∩ ghost_indxs) == part_size[1]*part_size[2]-2part_size[1]-2part_size[2]+4
    Z⁺ = length(part_indxs[2:end-1,2:end-1,end] ∩ ghost_indxs) == part_size[1]*part_size[2]-2part_size[1]-2part_size[2]+4
    return (X⁻,X⁺,Y⁻,Y⁺,Z⁻,Z⁺)
end