function main(mesh_partition,distribute)
    ## Parameters
    order = 1;
    dom = (0,1,0,1,0,1);
    el_size = (50,50,50);
    γ = 0.1;
    γ_reinit = 0.5;
    max_steps = 10
    tol = 10^-2
    max_iters = 1000;
    D = 1;
    interp = SmoothErsatzMaterialInterpolation()
    H,I,ρ = interp.H,interp.I,interp.ρ
    output_path = "./Results/main_testing"

    ## FE Setup
    model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
    f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= ymax*prop_Γ_D + eps() || 
        x[2] >= ymax-ymax*prop_Γ_D - eps()) && (x[3] <= zmax*prop_Γ_D + eps() ||
        x[3] >= zmax-zmax*prop_Γ_D - eps())) ? true : false;
    f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= 
        ymax/2+ymax*prop_Γ_N/4 + eps() && zmax/2-zmax*prop_Γ_N/4 - eps() <= x[3]
        <= zmax/2+zmax*prop_Γ_N/4 + eps()) ? true : false;
    update_labels!(1,model,f_Γ_D,coord_max,"Gamma_D")
    update_labels!(2,model,f_Γ_N,coord_max,"Gamma_N")

    ## Triangulations and measures
    Ω = Triangulation(model)
    Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
    dΩ = Measure(Ω,2order)
    dΓ_N = Measure(Γ_N,2order)

    ## Spaces
    V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"],dirichlet_masks=[true])
    U = TrialFESpace(V,[0.0])
    V_φ = TestFESpace(model,reffe)

    ## Create FE functions
    uh = zero(U);
    φh = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25,V_φ);
    φ = get_free_dof_values(φh)

    ## Weak form
    a(u,v,ϕ,dΩ,dΓ_N) = ∫((I ∘ _ϕh)*D*∇(u)⋅∇(v))dΩ
    l(v,ϕ,dΩ,dΓ_N) = ∫(v*g)dΓ_N
    res(u,v,ϕ,dΩ,dΓ_N) = a(u,v,ϕ,dΩ,dΓ_N) - l(v,ϕ,dΩ,dΓ_N)

    ## Optimisation functionals
    J = (u,ϕ,dΩ,dΓ_N) -> ∫((I ∘ ϕ)*D*∇(u)⋅∇(u))dΩ
    dJ = (q,u,ϕ,dΩ,dΓ_N) -> ∫(-D*∇(u)⋅∇(u)*q*(DH ∘ ϕ)*(norm ∘ ∇(ϕ)))dΩ;
    Vol = (u,ϕ,dΩ,dΓ_N) -> ∫(((ρ ∘ ϕ) - 0.5)/vol_D)dΩ;
    dVol = (q,u,ϕ,dΩ,dΓ_N) -> ∫(1/vol_D*q*(DH ∘ ϕ)*(norm ∘ ∇(ϕ)))dΩ
    J_func = Functional(J,dΩ,uh,φh;dF = dJ)
    Vol_func = Functional(Vol,dΩ,uh,φh;dF = dVol)

    ## Hilbertian extension-regularisation problems
    a_hilb = (p,q,dΩ)->∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
    hilb_solver = LUSolver()
    vel_ext = VelocityExtension(model,interp,a_hilb,order;dirichlet_tags=["Gamma_N"],ls=hilb_solver)
    U_reg,_ = vel_ext.spaces

    ## Finite difference solver and level set function
    stencil = DistributedStencil(FirstOrderStencil(D,Float64),model,max_steps,tol)
    vel = get_free_dof_values(zero(V_φ));
    stencil_caches = allocate_caches(stencil,φ,vel)
    reinit!(stencil,φ,γ_reinit,stencil_caches)

    ## Setup solver and FE operators
    solver = LUSolver()
    J_smap, C_smaps = AffineFEStateMap(J_func,[Vol_func],U,V,U_reg,a,l,res;ls = solver);
    
    ## Optimiser
    optimiser = AugmentedLagrangian(J_smap,C_smaps,
        stencil,stencil_caches,vel_ext,V_φ,vel,γ,γ_reinit);

    ## Stopping criterion
    history = zeros(max_iters,length(C_smaps)+2);
    function conv_cond(it,Ji,Ci,Li,λ,Λ,uhi,φhi,dLhi)
        return ~(it > 10 && 
            all(@.(abs(Li-history[it-5:it-1,1])) .< 1/5/maximum(el_size)*L_new) &&
            all(abs(Ci) < 0.0001))
    end

    for (it,Ji,Ci,Li,λ,Λ,uhi,φhi,dLhi) in takewhile(conv_cond,optimiser)
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