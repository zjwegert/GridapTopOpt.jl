abstract type AbstractOptimiser end

function setup!(...) where T<:AbstractOptimiser
    @abstractmethod
end

function step!(...) where T<:AbstractOptimiser
    @abstractmethod
end

function update_params!(::AbstractOptimiser,args...)
    return nothing
end

struct AugmentedLagrangian <: AbstractOptimiser
    ζ
    update_mod::Int
    params
    function AugmentedLagrangian(;λ0 = -0.01,Λ0 = 100,ζ = 0.9,update_mod::Int = 5)
        new(ζ,update_mod,[λ0,Λ0])
    end
end

function update!(m::AugmentedLagrangian,iter,C_new,vf)
    Λ = m.params[2]
    m.params[1] -= 1/Λ*(C_new - vf); 
    iszero(iter % m.update_mod) ? m.params[2] *= m.ζ : 0;
    return m.params[1],m.params[2]
end

function setup!(m::AugmentedLagrangian)
    ...
end

function step!(m::AugmentedLagrangian,iter,J,C,caches)
    @assert isone(length(C)) 

    ## Extend shape sensitivities
    ext_v_J = hilbertian_ext(v_J,φh,hilb_data,interp) |> get_free_dof_values
    ext_v_C = hilbertian_ext(v_C,φh,hilb_data,interp) |> get_free_dof_values
    
    ## Augmented Lagrangian method
    λ,Λ = update!(m,iter,C_new,vf)
    v_Lh = FEFunction(hilb_data.U_reg,ext_v_J - λ_new*ext_v_C + 1/Λ_new*(C_new - vf)*ext_v_C)    
    v_Lh_full = interpolate(v_Lh,V_φ)
    g_Ω = get_free_dof_values(v_Lh_full);
    
    ## Advect  & Reinitialise
    advect!(φ,g_Ω,model,Δ,γ,steps)
    reinit!(φ,model,Δ,0.5,2000,reinit_tol)       

    ## Calculate objective and constraints
    J_new,v_J,uh = obj(φh,g,mat,solve_data,interp)
    C_new,v_C = vol(φh,interp,solve_data.dΩ,solve_data.V_L2,vol_D)
    L_new = J_new - λ*(C_new - vf) + 1/(2Λ)*(C_new - vf)^2
end