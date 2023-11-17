abstract type AbstractOptimiser end

# Return tuple of first iteration state
function Base.iterate(::T) where T <: AbstractOptimiser
    @notimplemented
end

# Return tuple of next iteration state given current state
function Base.iterate(::T,state) where T <: AbstractOptimiser
    @notimplemented
end

## Augmented Lagrangian optimiser
struct AugmentedLagrangian{M} <: AbstractOptimiser
    λ::Vector{M}
    Λ::Vector{M}
    ζ::M
    vft::M
    update_mod::Int
    history::Array{M}
    last_iter_cache
    function AugmentedLagrangian(
            φ::AbstractVector,
            pcfs::PDEConstrainedFunctionals{N},
            stencil,vel_ext,interp,γ,γ_reinit;
            λ::Vector{M}=zeros(M,N),
            Λ::Vector{M}=zeros(M,N),
            ζ::M = 1.2,
            vft::M = 0.5,
            update_mod::Int = 5,
            max_iters::Int = 1000) where {N,M<:AbstractFloat}
        history = zeros(M,max_iters,2+N);
        vel = get_free_dof_values(interpolate(0,V_φ))
        cache = (φ,pcfs,stencil,vel_ext,interp,vel,γ,γ_reinit)
        new{M}(λ,Λ,ζ,vft,update_mod,history,cache)
    end
end

# Initialise AGM parameters
function initialise!(m::AugmentedLagrangian{M},J_init::M) where M
    vft = m.vft;
    m.λ .= 0.1*J_init/vft;
    m.Λ .= @. abs(0.01*m.λ/vft)
    return λ,Λ
end

# Update AGM parameters
function update!(m::AugmentedLagrangian{M},iter,C_new::Vector{M}) where M
    λ = m.λ; Λ = m.Λ;
    λ .-= Λ*C_new;
    iszero(iter % m.update_mod) ? Λ .*= m.ζ : 0;
    return λ,Λ
end

# Stopping criterion
function conv_cond(m::AugmentedLagrangian,state)
    return it > 10 && (all(@.(abs(Li-history[it-5:it-1,1])) .< 1/5/maximum(el_size)*L_new) &&
        all(abs(Ci) < 0.0001)) ? nothing : state; 
end

# 0th iteration
function Base.iterate(m::AugmentedLagrangian)
    φ,pcfs,_,vel_ext,_,_,_,_ = m.cache
    history = m.history
    ## Compute FE problem and shape derivatives
    J_init,C_init,dJ,dC = Gridap.evaluate!(pcfs,φ)

    ## Compute initial values
    λ,Λ = initialise!(m,J_init)
    L_init = J_init + sum(@. -λ*C_init + Λ/2*C_init^2)
    history[1,:] = [J_init,C_init...,L_init];

    ## Compute dL and project
    dL = dJ + sum(-λ[i]*dC[i] + Λ[i]*C_init[i]*dC[i] for i ∈ eachindex(λ))
    # Because project! takes a linear form on the RHS this should
    #   be the same as projecting each shape derivative then computing dL
    project!(vel_ext,dL)

    return 0,J_init,C_init,L_init,dL
end

# ith iteration
function Base.iterate(m::AugmentedLagrangian,state)
    it,φ,_,_,_,dL = state
    _,pcfs,stencil,vel_ext,interp,vel,γ,γ_reinit = m.cache
    
    ## Advect & Reinitialise
    interpolate!(FEFunction(U_reg,dL),vel,V_φ)
    advect!(stencil,φ,vel,γ)
    reinit!(stencil,φ,γ_reinit)   
    #### Up to here
    ## Calculate objective and constraints
    J_smap(φh)
    J_new = evaluate_functional(J_smap)
    C_new = evaluate_functional.(C_smaps)
    L_new = J_new + sum(@. -λ*C_new + Λ/2*C_new^2)

    ## Compute shape derivatives
    U_reg, _ = vel_ext.spaces
    dJh = compute_shape_derivative!(φh,J_smap)
    dCh = compute_shape_derivative!(φh,C_smaps)
    
    ## Augmented Lagrangian method
    λ,Λ = update!(m,it,C_new)
    dJ = get_free_dof_values(dJh)
    dC = get_free_dof_values.(dCh)
    dL = dJ + sum(-λ[i]*dC[i] + Λ[i]*C_new[i]*dC[i] for i ∈ eachindex(λ))
    copy!(get_free_dof_values(dLh),dL)
    consistent!(get_free_dof_values(dLh)) |> fetch
    project!(vel_ext,dLh)

    return it+1,J_new,C_new,L_new,λ,Λ,uh,φh,dLh
end