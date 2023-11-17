abstract type AbstractOptimiser end

# Return tuple of first iteration state
function Base.iterate(::T) where T <: AbstractOptimiser
    @notimplemented
end

# Return tuple of next iteration state given current state
function Base.iterate(::T,state) where T <: AbstractOptimiser
    @notimplemented
end

# Getters
get_optimiser_history(::AbstractOptimiser) = @notimplemented
get_level_set(::AbstractOptimiser) = @notimplemented

## Augmented Lagrangian optimiser
mutable struct AugmentedLagrangianHistory{T}
    # I'm not the biggest fan of this but it solves the problem of not
    #  having the explicit iteration count stored in a useful way.
    #  It also means we return a single item in iterator loop.
    it      :: Int
    const J :: Vector{T}
    const C :: Matrix{T}
    const L :: Vector{T}
    function AugmentedLagrangianHistory(T::Type,max_iters::Int,nconsts::Int)
        J = zeros(T,max_iters);
        C = zeros(T,max_iters,nconsts)
        L = similar(J)
        new{T}(0,J,C,L)
    end
end

# Convienence 
Base.last(m::AugmentedLagrangianHistory) = (m.it,m.J[m.it],m.C[m.it,:],m.L[m.it])

function write_history(m::AugmentedLagrangianHistory,path)
    J = m.J; C = m.C; L = m.L
    data = zip(J,C,L)
    writedlm("$path/history.csv",data)
end

function update!(m::AugmentedLagrangianHistory,J,C,L)
    m.it += 1
    m.J[m.it] = J 
    m.C[m.it] = C
    m.L[m.it] = L
    return nothing
end

struct AugmentedLagrangian{M} <: AbstractOptimiser
    λ               :: Vector{M}
    Λ               :: Vector{M}
    ζ               :: M
    vft             :: M
    update_mod      :: Int
    history         :: AugmentedLagrangianHistory{M}
    conv_criterion  :: Function
    cache
    function AugmentedLagrangian(
            φ::AbstractVector,
            pcfs::PDEConstrainedFunctionals{N},
            stencil,vel_ext,interp,γ,γ_reinit;
            λ::Vector{M}=zeros(M,N),
            Λ::Vector{M}=zeros(M,N),
            ζ::M = 1.2,
            vft::M = 0.5,
            update_mod::Int = 5,
            max_iters::Int = 1000,
            conv_criterion::Function = conv_cond) where {N,M<:AbstractFloat}

        history = AugmentedLagrangianHistory(M,max_iters,N)
        vel = get_free_dof_values(interpolate(0,V_φ))
        cache = (φ,pcfs,stencil,vel_ext,interp,vel,γ,γ_reinit)
        new{M}(λ,Λ,ζ,vft,update_mod,history,conv_criterion,cache)
    end
end

get_optimiser_history(m::AugmentedLagrangian) = m.history
get_level_set(m::AugmentedLagrangian) = first(m.cache)

# Initialise AGM parameters
function initialise!(m::AugmentedLagrangian{M},J_init::M,C_init::Vector{M}) where M
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
    history = m.history
    it,_,Ci,Li,_,_ = state

    return it > 10 && (all(@.(abs(Li-history.L[it-5:it-1])) .< 1/5/maximum(el_size)*Li) &&
        all(@. abs(Ci) < 0.0001)) ? nothing : state; 
end

# 0th iteration
function Base.iterate(m::AugmentedLagrangian)
    φ,pcfs,_,vel_ext,_,_,_,_ = m.cache
    ## Compute FE problem and shape derivatives
    J_init,C_init,dJ,dC = Gridap.evaluate!(pcfs,φ)

    ## Compute initial values
    λ,Λ = initialise!(m,J_init,C_init)
    L_init = J_init + sum(@. -λ*C_init + Λ/2*C_init^2)
    update!(m.history,J_init,C_init,L_init)

    ## Compute dL and project
    dL = dJ + sum(-λ[i]*dC[i] + Λ[i]*C_init[i]*dC[i] for i ∈ eachindex(λ))
    # Because project! takes a linear form on the RHS this should
    #   be the same as projecting each shape derivative then computing dL
    project!(vel_ext,dL)

    return m.history,dL
end

# ith iteration
function Base.iterate(m::AugmentedLagrangian,dL)
    φ,pcfs,stencil,vel_ext,_,vel,γ,γ_reinit = m.cache
    conv_criterion = m.conv_criterion
    ## Advect & Reinitialise
    interpolate!(FEFunction(U_reg,dL),vel,V_φ)
    advect!(stencil,φ,vel,γ)
    reinit!(stencil,φ,γ_reinit)   

    ## Calculate objective, constraints, and shape derivatives
    J_new,C_new,dJ,dC = Gridap.evaluate!(pcfs,φ)
    L_new = J_new + sum(@. -λ*C_new + Λ/2*C_new^2)
    
    ## Augmented Lagrangian method
    λ,Λ = update!(m,it,C_new)
    dL = dJ + sum(-λ[i]*dC[i] + Λ[i]*C_new[i]*dC[i] for i ∈ eachindex(λ))
    project!(vel_ext,dL)

    ## History
    update!(m.history,J_new,C_new,L_new)

    if conv_criterion(m)
        return nothing
    else
        return m.history,dL
    end
end