abstract type AbstractOptimiser end

function setup!(...) where T<:AbstractOptimiser
    @abstractmethod
end

function step!(...) where T<:AbstractOptimiser
    @abstractmethod
end

function check_stopping_criteria(...) where T<:AbstractOptimiser
    @abstractmethod
end

Base.@kwdef struct AugmentedLagrangian
    λ = -0.01
    Λ = 100
    ζ = 0.9
end

function step!(m::AugmentedLagrangian,J,C) where T<:AugmentedLagrangian
    # @assert isone(length(C)) 

    ## Calculate objective and constraints
    J_new,v_J,uh = obj(φh,g,mat,solve_data,interp)
    C_new,v_C = vol(φh,interp,solve_data.dΩ,solve_data.V_L2,vol_D)
    L_new = J_new - λ*(C_new - vf) + 1/(2Λ)*(C_new - vf)^2

    if it > 10 && all(@.(abs(L_new-history[it-5:it-1,3])) .< 1/5/maximum(el)*L_new) &&
            abs(C_new-vf) < 0.0001
        write_vtk(Ω,"$path/struc_$it",φh,uh,interp.H)
        return true
    else
        return false
    end
    ## Extend shape sensitivities
    ext_v_J = hilbertian_ext(v_J,φh,hilb_data,interp) |> get_free_dof_values
    ext_v_C = hilbertian_ext(v_C,φh,hilb_data,interp) |> get_free_dof_values
    # Augmented Lagrangian method
    λ -= 1/Λ*(C_new - vf); 
    iszero(it % 5) ? Λ *= ζ : 0;
    v_Lh = FEFunction(hilb_data.U_reg,ext_v_J - λ*ext_v_C + 1/Λ*(C_new - vf)*ext_v_C)    
    v_Lh_full = interpolate(v_Lh,V_φ)
    g_Ω = get_free_dof_values(v_Lh_full);
    ## Advect  & Reinitialise
    advect!(φ,g_Ω,model,Δ,γ,steps)
    reinit!(φ,model,Δ,0.5,2000,reinit_tol)       
end