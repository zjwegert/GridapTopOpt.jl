function main(mesh_partition,distribute)
    ## Parameters
    order = 1;
    dom = (0,1,0,1,0,1);
    el_size = (50,50,50);
    vf = 0.5;
    γ = 0.1;
    γ_reinit = 0.5;
    max_steps = 10
    tol = 10^-2
    max_iters = 1000;
    D = 1;
    η_coeff = 2;
    output_path = "./Results/main_testing"

    ## FE Setup
    model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
    Δ = get_Δ(model)
    f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= ymax*prop_Γ_D + eps() || 
        x[2] >= ymax-ymax*prop_Γ_D - eps()) && (x[3] <= zmax*prop_Γ_D + eps() ||
        x[3] >= zmax-zmax*prop_Γ_D - eps())) ? true : false;
    f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= 
        ymax/2+ymax*prop_Γ_N/4 + eps() && zmax/2-zmax*prop_Γ_N/4 - eps() <= x[3]
        <= zmax/2+zmax*prop_Γ_N/4 + eps()) ? true : false;
    update_labels!(1,model,f_Γ_D,"Gamma_D")
    update_labels!(2,model,f_Γ_N,"Gamma_N")

    ## Triangulations and measures
    Ω = Triangulation(model)
    Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
    dΩ = Measure(Ω,2order)
    dΓ_N = Measure(Γ_N,2order)

    ## Spaces
    V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"],dirichlet_masks=[true])
    U = TrialFESpace(V,[0.0])
    V_φ = TestFESpace(model,reffe)
    V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
    U_reg = TrialFESpace(V_reg,0)

    ## Create FE functions
    φh = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25,V_φ);
    φ = get_free_dof_values(φh)

    ## Interpolation and weak form
    interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ))
    H,I,ρ = interp.H,interp.I,interp.ρ

    a(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*D*∇(u)⋅∇(v))dΩ
    l(v,φ,dΩ,dΓ_N) = ∫(v*g)dΓ_N
    res(u,v,φ,dΩ,dΓ_N) = a(u,v,φ,dΩ,dΓ_N) - l(v,φ,dΩ,dΓ_N)

    ## Optimisation functionals
    J = (u,φ,dΩ,dΓ_N) -> ∫((I ∘ φ)*D*∇(u)⋅∇(u))dΩ
    dJ = (q,u,φ,dΩ,dΓ_N) -> ∫(-D*∇(u)⋅∇(u)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;
    Vol = (u,φ,dΩ,dΓ_N) -> ∫(((ρ ∘ φ) - vf)/vol_D)dΩ;
    dVol = (q,u,φ,dΩ,dΓ_N) -> ∫(1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

    ## Finite difference solver and level set function
    stencil = AdvectionStencil(FirstOrderStencil(D,Float64),model,V_φ,max_steps,tol)
    reinit!(stencil,φ,γ_reinit)

    ## Setup solver and FE operators
    state_map = AffineFEStateMap(a,l,res,U,V,V_φ,U_reg,φh,dΩ,dΓ_N)
    pcfs = PDEConstrainedFunctionals(J,[Vol],state_map,analytic_dC=[dVol])

    ## Hilbertian extension-regularisation problems
    a_hilb = (p,q,dΩ)->∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
    vel_ext = VelocityExtension(a_hilb,U_reg,V_reg,dΩ)
    
    ## Optimiser
    optimiser = AugmentedLagrangian(φ,pcfs,stencil,vel_ext,interp,γ,γ_reinit,max_iters);

                    # Write getters and remove from λ
    for (it,Ji,Ci,Li) in optimiser
        if i_am_main(ranks) 
            println("it: ",it," | J: ",round(Ji;digits=5),
                              " | ","C: ",round(Ci;digits=5),
                              " | ","L: ",round(Li;digits=5))
            println("λ = ",λ," Λ = ",Λ)
            writedlm("$path/history.csv",history[1:it,:])
        end
        # if isone(it) || iszero(it % 30) 
        writevtk(Ω,output_path*"/struc_$it",cellfields=["phi"=>φhi,"H(phi)"=>(H ∘ φhi),
            "|nabla(phi))|"=>(norm ∘ ∇(φhi)),"uh"=>uhi,"dLh"=>dLhi])
        # end
    end
end