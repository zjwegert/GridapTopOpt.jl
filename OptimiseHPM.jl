function main(ranks,mesh_partition,setup::Function,path::String)
    ### Get parameters
    prob,obj,lsf,vf,mat,g,fe_order,coord_max,el,η_coeff,
        α_coeff,μ,γ_max,γ_min,γ,steps,reinit_tol = setup()
    vol_D = prod(coord_max);
    Δ = coord_max./el # Element size
    η,α = [η_coeff*maximum(Δ),α_coeff*maximum(Δ)] # Smoothing parameters
    interp = SmoothErsatzMaterialInterpolation(η=η) # Material interpolation
    HPM = HilbertianProjectionMethod{Float64}() # Projection method
    ### Setup FE data
    model,Ω,V_φ,solve_data,hilb_data = prob(ranks,mesh_partition,fe_order,coord_max,el,α)
    ### Setup level set
    φh = interpolate(lsf,V_φ)
    φ = get_free_dof_values(φh)
    reinit!(φ,model,Δ,0.1,2000,reinit_tol)
    φ0 = zero(φ); φ0 .= φ;
    ### Main loop
    J_new,_,uh = obj(φh,g,mat,solve_data,interp)
    C_new,_ = vol(φh,interp,solve_data.dΩ,solve_data.V_L2,vol_D)
    history = NaN*zeros(1000,2);
    history[1,:] = [J_new,C_new]
    MPI.Comm_rank(comm) == root ? println("it: ",0," | J: ",
        round(J_new;digits=5)," | ","Vol: ",round(C_new;digits=5)) : 0
    write_vtk(Ω,"$path/struc_0",φh,uh,interp.H)
    for it in 1:1000
        ## Create temp level set structure and other objects
        φ0 .= φ;
        consistent!(φ0) |> fetch
        J,C = [copy(J_new),copy(C_new)];
        ## Calculate shape sensitivities and apply constraint method
        _,v_J,uh = obj(φh,g,mat,solve_data,interp)
        _,v_C = vol(φh,interp,solve_data.dΩ,solve_data.V_L2,vol_D)
        ext_v_J = hilbertian_ext(v_J,φh,hilb_data,interp)
        ext_v_C = hilbertian_ext(v_C,φh,hilb_data,interp)
        θn_h = project(get_free_dof_values(ext_v_J),[C-vf],[get_free_dof_values(ext_v_C)],
            hilb_data.K,hilb_data.U_reg,HPM)
        θn_h_full = interpolate(θn_h,V_φ)
        g_Ω = get_free_dof_values(θn_h_full);
        ## Line search
        for k ∈ 1:10
            ## Advect  & Reinitialise
            advect!(φ,g_Ω,model,Δ,γ,steps)
            reinit!(φ,model,Δ,0.1,2000,reinit_tol)
            ## Calcuate new objective and constraints
            J_new,_,uh = obj(φh,g,mat,solve_data,interp)
            C_new,_ = vol(φh,interp,solve_data.dΩ,solve_data.V_L2,vol_D)
            ## Reduce line search parameter if constraints close to saturation
            _μ = abs(C_new-vf) < 0.01 ? μ/10 : μ
            if J_new < J + _μ*abs(J) || γ <= γ_min ## Accept
                γ = min(1.1*γ,γ_max)
                break
            else ## Reject
                γ = max(0.7*γ,γ_min)
                φ .= φ0
                consistent!(φ) |> fetch
            end
        end
        ## Log
        history[it,:] = [J_new,C_new]
        if MPI.Comm_rank(comm) == root 
            println("it: ",it," | J: ",round(J_new;digits=5)," | ","Vol: ",round(C_new;digits=5))
            writedlm("$path/history.csv",history)
        end
        iszero(it % 30) ? write_vtk(Ω,"$path/struc_$it",φh,uh,interp.H) : 0
        ## Check stopping criteria
        if it > 10 && all(@.(abs(J_new-history[it-5:it-1,1])) .< 1/5/maximum(el)*J_new) &&
                abs(C_new-vf) < 0.0001
            write_vtk(Ω,"$path/struc_$it",φh,uh,interp.H)
            break
        end
    end
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