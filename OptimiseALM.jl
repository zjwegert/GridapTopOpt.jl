function main(ranks,mesh_partition,setup::Function,path::String)
    ### Get parameters
    prob,obj,lsf,vf,mat,g,fe_order,coord_max,el,η_coeff,
        α_coeff,_,_,_,γ,steps,reinit_tol = setup()  
    vol_D = prod(coord_max);
    Δ = coord_max./el # Element size
    η = η_coeff*maximum(Δ); α = α_coeff*maximum(Δ) # Smoothing parameters
    interp = SmoothErsatzMaterialInterpolation(η=η) # Material interpolation
    λ = -0.01; Λ = 100; ζ = 0.9; # Augmented Lagrangian parameters
    ### Setup FE data
    model,Ω,V_φ,solve_data,hilb_data = prob(ranks,mesh_partition,fe_order,coord_max,el,α)
    ### Setup level set
    φh = interpolate(lsf,V_φ)
    φ = get_free_dof_values(φh)
    reinit!(φ,model,Δ,0.5,2000,reinit_tol)
    ### Main loop
    J_new,_,uh = obj(φh,g,mat,solve_data,interp)
    C_new,_ = vol(φh,interp,solve_data.dΩ,solve_data.V_L2,vol_D)
    L_new = J_new - λ*(C_new - vf) + 1/(2Λ)*(C_new - vf)^2
    history = NaN*zeros(1000,3); history[1,:] = [J_new,C_new,L_new]
    MPI.Comm_rank(comm) == root ? println("it: ",0," | J: ",
        round(J_new;digits=5)," | ","Vol: ",round(C_new;digits=5)) : 0
    write_vtk(Ω,"$path/struc_0",φh,uh,interp.H)
    for it in 1:1000
        ## Calculate objective and constraints
        J_new,v_J,uh = obj(φh,g,mat,solve_data,interp)
        C_new,v_C = vol(φh,interp,solve_data.dΩ,solve_data.V_L2,vol_D)
        L_new = J_new - λ*(C_new - vf) + 1/(2Λ)*(C_new - vf)^2
        ## Log
        history[it,:] = [J_new,C_new,L_new]
        if MPI.Comm_rank(comm) == root 
            println("it: ",it," | J: ",round(J_new;digits=5)," | ","Vol: ",round(C_new;digits=5))
            println("λ = ",λ," Λ = ",Λ)
            writedlm("$path/history.csv",history)
        end
        # iszero(it % 30) ? write_vtk(Ω,"$path/struc_$it",φh,uh,interp.H) : 0
        write_vtk(Ω,"$path/struc_$it",φh,uh,interp.H)
        ## Check stopping criteria
        if it > 10 && all(@.(abs(L_new-history[it-5:it-1,3])) .< 1/5/maximum(el)*L_new) &&
                abs(C_new-vf) < 0.0001
            write_vtk(Ω,"$path/struc_$it",φh,uh,interp.H)
            break
        end
        ## Extend shape sensitivities
        ext_v_J = hilbertian_ext(v_J,φh,hilb_data,interp) |> get_free_dof_values
        ext_v_C = hilbertian_ext(v_C,φh,hilb_data,interp) |> get_free_dof_values
        # Augmented Lagrangian method
        λ -= 1/Λ*(C_new - vf); 
        iszero(it % 5) ? Λ *= ζ : 0;
        v_Lh = FEFunction(hilb_data.U_reg,ext_v_J - λ*ext_v_C + 1/Λ*(C_new - vf)*ext_v_C)
        g_Ω = get_vel_at_lsf(v_Lh,V_φ);        
        ## Advect  & Reinitialise
        advect!(φ,g_Ω,model,Δ,γ,steps)
        reinit!(φ,model,Δ,0.5,2000,reinit_tol)        
    end
end

## With line search
# function main(ranks,mesh_partition,setup::Function,path::String)
#     ### Get parameters
#     prob,obj,lsf,vf,mat,g,fe_order,coord_max,el,η_coeff,
#         α_coeff,μ,γ_max,γ_min,γ,steps,reinit_tol = setup()  
#     vol_D = prod(coord_max);
#     Δ = coord_max./el # Element size
#     η = η_coeff*maximum(Δ); α = α_coeff*maximum(Δ) # Smoothing parameters
#     interp = SmoothErsatzMaterialInterpolation(η=η) # Material interpolation
#     λ = -0.01; Λ = 100; ζ = 0.9; # Augmented Lagrangian parameters
#     ### Setup FE data
#     model,Ω,V_φ,solve_data,hilb_data = prob(ranks,mesh_partition,fe_order,coord_max,el,α)
#     ### Setup level set
#     φh = interpolate(lsf,V_φ)
#     φ = get_free_dof_values(φh)
#     reinit!(φ,model,Δ,0.1,2000,reinit_tol)
#     φ0 = zero(φ); φ0 .= φ;
#     ### Main loop
#     J_new,_,uh = obj(φh,g,mat,solve_data,interp)
#     C_new,_ = vol(φh,interp,solve_data.dΩ,solve_data.V_L2,vol_D)
#     L_new = J_new - λ*(C_new - vf) + 1/(2Λ)*(C_new - vf)^2
#     history = NaN*zeros(1000,3);
#     history[1,:] = [J_new,C_new,L_new]
#     MPI.Comm_rank(comm) == root ? println("it: ",0," | J: ",
#         round(J_new;digits=5)," | ","Vol: ",round(C_new;digits=5)) : 0
#     write_vtk(Ω,"$path/struc_0",φh,uh,interp.H)
#     for it in 1:1000
#         ## Create temp level set structure and other objects
#         φ0 .= φ;
#         consistent!(φ0) |> fetch
#         J,C,L = [copy(J_new),copy(C_new),copy(L_new)];
#         ## Calculate shape sensitivities
#         _,v_J,uh = obj(φh,g,mat,solve_data,interp)
#         _,v_C = vol(φh,interp,solve_data.dΩ,solve_data.V_L2,vol_D)
#         ext_v_J = hilbertian_ext(v_J,φh,hilb_data,interp) |> get_free_dof_values
#         ext_v_C = hilbertian_ext(v_C,φh,hilb_data,interp) |> get_free_dof_values
#         # Augmented Lagrangian method
#         λ -= 1/Λ*(C - vf); 
#         iszero(it % 5) ? Λ *= ζ : 0;
#         L = J - λ*(C - vf) + 1/(2Λ)*(C - vf)^2
#         v_Lh = FEFunction(hilb_data.U_reg,ext_v_J - λ*ext_v_C + 1/Λ*(C - vf)*ext_v_C)
#         g_Ω = get_vel_at_lsf(v_Lh,V_φ);
#         ## Line search
#         for k ∈ 1:10
#             ## Advect  & Reinitialise
#             advect!(φ,g_Ω,model,Δ,γ,steps)
#             reinit!(φ,model,Δ,0.1,2000,reinit_tol)
#             ## Calcuate new objective and constraints
#             J_new,_,uh = obj(φh,g,mat,solve_data,interp)
#             C_new,_ = vol(φh,interp,solve_data.dΩ,solve_data.V_L2,vol_D)
#             L_new = J_new - λ*(C_new - vf) + 1/(2Λ)*(C_new - vf)^2
#             _μ = abs(C_new-vf) < 0.01 ? μ/10 : μ
#             if L_new < L + _μ*abs(L) || γ <= γ_min ## Accept
#                 γ = min(1.1*γ,γ_max)
#                 break
#             else ## Reject
#                 γ = max(0.7*γ,γ_min)
#                 φ .= φ0
#                 consistent!(φ) |> fetch
#             end
#         end
#         ## Log
#         history[it,:] = [J_new,C_new,L_new]
#         if MPI.Comm_rank(comm) == root 
#             println("it: ",it," | J: ",round(J_new;digits=5)," | ","Vol: ",round(C_new;digits=5))
#             println("λ = ",λ," Λ = ",Λ)
#             writedlm("$path/history.csv",history)
#         end
#         # iszero(it % 30) ? write_vtk(Ω,"$path/struc_$it",φh,uh,interp.H) : 0
#         write_vtk(Ω,"$path/struc_$it",φh,uh,interp.H)
#         ## Check stopping criteria
#         if it > 10 && all(@.(abs(L_new-history[it-5:it-1,3])) .< 10/prod(el)*L_new) &&
#                 abs(C_new-vf) < 0.0001
#             write_vtk(Ω,"$path/struc_$it",φh,uh,interp.H)
#             break
#         end
#     end
# end

# Material interpolation
Base.@kwdef struct SmoothErsatzMaterialInterpolation{M<:AbstractFloat}
    η::M # Smoothing radius
    ϵₘ::M = 10^-3 # Void material multiplier
    H = x -> H_η(x,η=η)
    DH = x -> DH_η(x,η=η)
    I = φ -> (1 - H(φ)) + ϵₘ*H(φ)
    ρ = φ -> 1 - H(φ)
end

# Objective and constraint
function thermal_compliance(φh::DistributedCellField,g,D::M,solve_data::NT,interp::T) where {
        M<:AbstractFloat,T<:SmoothErsatzMaterialInterpolation,NT<:NamedTuple}
    I = interp.I; dΩ=solve_data.dΩ; dΓ_N=solve_data.dΓ_N;
    ## Weak form
    a(u,v) = ∫((I ∘ φh)*D*∇(u)⋅∇(v))dΩ
    l(v) = ∫(v*g)dΓ_N
    ## Assembly
    op = AffineFEOperator(a,l,solve_data.U,solve_data.V,solve_data.assem)
    K = op.op.matrix;
    ## Solve
    ls = PETScLinearSolver()
    uh = solve(ls,op)
    u = correct_ghost_layout(uh,K.col_partition)
    ## Compute J and v_J
    J = dot(u,(K*u))
    v_J = interpolate(-D*∇(uh)⋅∇(uh),solve_data.V_L2)
    return J,v_J,uh
end
function vol(φh::DistributedCellField,interp::T,dΩ::ME,V_L2::V,vol_D::M) where {
        M<:AbstractFloat,T<:SmoothErsatzMaterialInterpolation,
        V<:DistributedFESpace,ME<:DistributedMeasure}
    ρ = interp.ρ;
    vol = sum(∫(ρ ∘ φh)dΩ)/vol_D;
    v_vol = interpolate(x->one(M)/vol_D,V_L2)
    return vol,v_vol
end
# Hilbertian extension-regulariation
function hilbertian_ext(vh::DistributedCellField,φh::DistributedCellField,hilb_data::NT,
        interp::T) where {T<:SmoothErsatzMaterialInterpolation,NT<:NamedTuple}
    ## Unpack
    dΩ=hilb_data.dΩ; K=hilb_data.K; DH = interp.DH
    ## Linear Form
    J′(v) = ∫(-vh*v*(DH ∘ φh)*(norm ∘ ∇(φh)))dΩ;
    ## Assembly
    b = assemble_vector(J′,hilb_data.assem,hilb_data.V_reg)
    op = AffineFEOperator(hilb_data.U_reg,hilb_data.V_reg,K,b)
    ## Solve
    ls = PETScLinearSolver()
    xh = solve(ls,op)
    x = correct_ghost_layout(xh,K.col_partition)
    return FEFunction(hilb_data.U_reg,x)
end


