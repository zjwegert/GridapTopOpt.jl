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
struct AugmentedLagrangian{M} <: AbstractOptimiser where M <: AbstractFloat
    λ::Vector{M}
    Λ::Vector{M}
    ζ::M
    vft::M
    update_mod::Int
    last_iter_cache
    function AugmentedLagrangian(
            pcfs,stencil,vel_ext,interp,γ,γ_reinit;
            λ=zeros(PetscScalar,length(C_smaps)+1),
            Λ=zeros(PetscScalar,length(C_smaps)+1),
            ζ = 1.2,
            vft = 0.5,
            update_mod::Int = 5)
        cache = (pcfs,stencil,vel_ext,interp,γ,γ_reinit)
        new(λ,Λ,ζ,vft,update_mod,cache)
    end
end

# Method parameters
function initialise!(m::AugmentedLagrangian{M},J_init::M) where M
    vft = m.vft;
    m.λ .= 0.1*J_init/vft;
    m.Λ .= @. abs(0.01*m.λ/vft)
    return λ,Λ
end

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
    J_smap,C_smaps,stencil,stencil_caches,vel_ext,V_φ,vel,γ,γ_reinit = m.cache
    ## Compute FE problems
    uh,φh = get_states(J_smap)
    J_smap(φh)

    ## Compute initial values
    J_init = evaluate_functional(J_smap)
    C_init = evaluate_functional.(C_smaps)
    λ,Λ = initialise!(m,J_init)
    L_init = J_init + sum(@. -λ*C_init + Λ/2*C_init^2)
    
    ## Compute shape derivatives
    U_reg, _ = vel_ext.spaces
    dJh = compute_shape_derivative!(φh,J_smap)
    dCh = compute_shape_derivative!(φh,C_smaps)
    # project!(vel_ext,dJh)
    # project!(vel_ext,dCh)

    dJ = get_free_dof_values(dJh)
    dC = get_free_dof_values.(dCh)
    dL = dJ + sum(-λ[i]*dC[i] + Λ[i]*C_init[i]*dC[i] for i ∈ eachindex(λ))
    dLh = FEFunction(U_reg,dL)
    # Because project! takes a linear form on the RHS this should
    #   be the same as projecting each shape derivative then computing dL
    project!(vel_ext,dLh)

    return 0,J_init,C_init,L_init,λ,Λ,uh,φh,dLh
end

# ith iteration
function Base.iterate(m::AugmentedLagrangian,state)
    it,_,_,_,_,_,uh,φh,dLh = state
    J_smap,C_smaps,stencil,stencil_caches,vel_ext,V_φ,vel,γ,γ_reinit = m.cache
    φ = get_free_dof_values(φh)
    ## Advect & Reinitialise
    interpolate!(dLh,vel,V_φ)
    advect!(stencil,φ,vel,γ,stencil_caches)
    reinit!(stencil,φ,γ_reinit,stencil_caches)   

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