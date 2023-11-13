# function main()

#     # Problem setup, defining the blocks

#     φh
#     caches = setup!(optimizer,...) # <- must include first iteration
#     for k in 1:nsteps
#         J_new,C_new,φh_new = step!(optimizer,...)
#         # Logging using results
#         # Check convergence criteria, using results
#     end

# end

function main_3D(mesh_partition,distribute)
    ## Parameters
    order = 1;
    dom = (0,1,0,1,0,1);
    el_size = (50,50,50);
    # ...
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

    ## Weak form
    function a(u,v,ϕ,dΩ,dΓ_N)
        _ϕh = ϕ_to_ϕₕ(ϕ,V_ϕ)
        ∫((I ∘ _ϕh)*D*∇(u)⋅∇(v))dΩ
    end
    a(ϕ,dΩ,dΓ_N) = (u,v) -> a(u,v,ϕ,dΩ,dΓ_N)
    l(v,ϕ,dΩ,dΓ_N) = ∫(v*g)dΓ_N
    l(ϕ,dΩ,dΓ_N) = v -> l(v,ϕ,dΩ,dΓ_N)
    res(u,v,ϕ,dΩ,dΓ_N) = a(u,v,ϕ,dΩ,dΓ_N) - l(v,ϕ,dΩ,dΓ_N)
    # This large block is not very nice, we should clean this up.

    ## Finite difference solver and level set function
    # Note, γ (and therefore Δt) for advect and reinit will most likely be different.
    # Will need to adjust Advection.jl to account for this - γ_advect, γ_reinit
    stencil = FirstOrderStencil(D,γ);
    adv_stencil = DistributedAdvectionStencil(stencil,model,max_steps,tol)
    φh = interpolate(gen_lsf(4,0.2),V_φ)
    φ = get_free_dof_values(φh)
    reinit!(adv_stencil,φ,caches)

    ## Optimisation functionals
    using_ad = true;
    J = (u,ϕ,dΩ,dΓ_N) -> ∫((I ∘ ϕ)*D*∇(u)⋅∇(u))dΩ
    dJ = using_ad ? nothing : (q,u,ϕ,dΩ,dΓ_N) -> ∫(-D*∇(u)⋅∇(u)*v*(DH ∘ ϕ)*(norm ∘ ∇(ϕ)))dΩ;
    C1 = (u,ϕ,dΩ,dΓ_N) -> ∫(((ρ ∘ ϕ) - 0.5)/vol_D)dΩ;
    dC1 = using_ad ? nothing : (q,u,ϕ,dΩ,dΓ_N) -> ∫(1/vol_D*v*(DH ∘ ϕ)*(norm ∘ ∇(ϕ)))dΩ

    J_func = Functional(J,dΩ,uh,ϕh;dF = dJ)
    C1_func = Functional(C1,dΩ,uh,ϕh;dF = dC1)

    """
    For each of J_func, Ci_func (i=1:n) there is a 
        smap = AffineFEStateMap(ϕh,loss,a,l,res);
    for AD purposes. Can these be setup in the optimiser `setup!` phase?

    Some important properties of these:
    1. The caches of both should point to all the same objects (?)
    2. We may only want AD for particular functionals (e.g., only J_func and a subset of {Ci})
    3. J_func() or J_func(state...) returns value of functional J_func at current state
    4. smap(ϕ) should compute FE problem in place and return FEFunction
    5. We can call `compute_shape_derivative!` on a functional & cache to compute the shape derivative in place and return FEFunction
    """

    optimiser = AugmentedLagrangian();

    ## Hilbertian extension-regularisation problems
    a_hilb = (p,q,dΩ)->∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
    vel_ext = VelocityExtension(model,interp,a_hilb,1)
    ## Solver
    solver = PETScLinearSolver() # <- This will crash if outside GridapPETSc.with. Need to decide where this goes?
    caches = setup!(optimizer,...) # <- Needs to run first iteration and get sensitivities etc.

    for k in 1:nsteps
        J_new,C_new,φh_new = step!(optimizer,...) # Change to iterator like other julia packages and remove for loop
        # Logging using results
        # Check convergence criteria, using results
    end
end