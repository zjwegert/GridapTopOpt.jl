using Gridap, GridapDistributed, GridapPETSc, PartitionedArrays, LSTO_Distributed

"""
    (Serial) Maximum bulk modulus inverse homogenisation with Lagrangian method in 2D.

    Optimisation problem:
        Min J(Ω) = -κ(Ω) + ∫ ξ dΩ
         Ω
      s.t., ⎡For unique εᴹᵢ, find uᵢ∈V=H¹ₚₑᵣ(Ω)ᵈ, 
            ⎣∫ ∑ᵢ C ⊙ ε(uᵢ) ⊙ ε(vᵢ) dΩ = ∫ -∑ᵢ C ⊙ ε⁰ᵢ ⊙ ε(vᵢ) dΩ, ∀v∈V.
""" 
function main()
    ## Parameters
    order = 1;
    xmax,ymax=(1.0,1.0)
    dom = (0,xmax,0,ymax);
    el_size = (200,200);
    γ = 0.1;
    γ_reinit = 0.5;
    max_steps = floor(Int,minimum(el_size)/10)
    tol = 1/(order^2*10)*prod(inv,minimum(el_size))
    C = isotropic_2d(1.,0.3);
    η_coeff = 2;
    α_coeff = 4;
    path = "./Results/main_inverse_homogenisation"

    ## FE Setup
    model = CartesianDiscreteModel(dom,el_size,isperiodic=(true,true));
    Δ = get_Δ(model)
    f_Γ_D(x) = iszero(x)
    update_labels!(1,model,f_Γ_D,"origin")

    ## Triangulations and measures
    Ω = Triangulation(model)
    dΩ = Measure(Ω,2order)

    ## Spaces
    reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
    reffe_scalar = ReferenceFE(lagrangian,Float64,order)
    _V = TestFESpace(model,reffe;dirichlet_tags=["origin"])
    _U = TrialFESpace(_V,VectorValue(0.0,0.0))
    U = MultiFieldFESpace([_U,_U,_U]);
    V = MultiFieldFESpace([_V,_V,_V]);
    V_reg = V_φ = TestFESpace(model,reffe_scalar)
    U_reg = TrialFESpace(V_reg)

    ## Create FE functions
    lsf_fn = x->max(gen_lsf(2,0.4)(x),gen_lsf(2,0.4;b=VectorValue(0,0.5))(x));
    φh = interpolate(lsf_fn,V_φ);
    φ = get_free_dof_values(φh)

    ## Interpolation and weak form
    interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ))
    I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

    εᴹ = (TensorValue(1.,0.,0.,0.),     # ϵᵢⱼ⁽¹¹⁾≡ϵᵢⱼ⁽¹⁾
          TensorValue(0.,0.,0.,1.),     # ϵᵢⱼ⁽²²⁾≡ϵᵢⱼ⁽²⁾
          TensorValue(0.,1/2,1/2,0.))   # ϵᵢⱼ⁽¹²⁾≡ϵᵢⱼ⁽³⁾

    a(u,v,φ,dΩ) = ∫((I ∘ φ)*sum(C ⊙ ε(u[i]) ⊙ ε(v[i]) for i ∈ eachindex(εᴹ)))dΩ
    l(v,φ,dΩ) = ∫(-(I ∘ φ)*sum(C ⊙ εᴹ[i] ⊙ ε(v[i]) for i ∈ eachindex(εᴹ)))dΩ;
    res(u,v,φ,dΩ) = a(u,v,φ,dΩ) - l(v,φ,dΩ)

    ## Optimisation functionals
    _C(C,ε_p,ε_q) = C ⊙ ε_p ⊙ ε_q;
    _K(C,(u1,u2,u3),εᴹ) = (_C(C,ε(u1)+εᴹ[1],εᴹ[1]) + _C(C,ε(u2)+εᴹ[2],εᴹ[2]) + 2*_C(C,ε(u1)+εᴹ[1],εᴹ[2]))/4
    _v_K(C,(u1,u2,u3),εᴹ) = (_C(C,ε(u1)+εᴹ[1],ε(u1)+εᴹ[1]) + _C(C,ε(u2)+εᴹ[2],ε(u2)+εᴹ[2]) + 2*_C(C,ε(u1)+εᴹ[1],ε(u2)+εᴹ[2]))/4   

    ξ = 0.54983
    J = (u,φ,dΩ) -> ∫(-(I ∘ φ)*_K(C,u,εᴹ) + ξ*(ρ ∘ φ))dΩ
    dJ = (q,u,φ,dΩ) -> ∫((ξ - _v_K(C,u,εᴹ))*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;

    ## Finite difference solver and level set function
    stencil = AdvectionStencil(FirstOrderStencil(2,Float64),model,V_φ,Δ./order,max_steps,tol)
    reinit!(stencil,φ,γ_reinit)

    ## Setup solver and FE operators
    state_map = AffineFEStateMap(a,l,res,U,V,V_φ,U_reg,φh,dΩ)
    pcfs = PDEConstrainedFunctionals(J,state_map,analytic_dJ=dJ)

    ## Hilbertian extension-regularisation problems
    α = α_coeff*maximum(Δ)
    a_hilb = (p,q,dΩ)->∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
    vel_ext = VelocityExtension(a_hilb,U_reg,V_reg,dΩ)
    
    ## Optimiser
    make_dir(path)
    optimiser = AugmentedLagrangian(φ,pcfs,stencil,vel_ext,interp,el_size,γ,γ_reinit);
    for history in optimiser
        it,Ji,Ci,Li = last(history)
        print_history(it,["J"=>Ji])
        write_history(history,path*"/history.csv")
        write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh))])
    end
    it,Ji,Ci,Li = last(optimiser.history)
    print_history(it,["J"=>Ji])
    write_history(optimiser.history,path*"/history.csv")
    uhi = get_state(pcfs)
    write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),
        "|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh1"=>uhi[1],"uh2"=>uhi[2],"uh3"=>uhi[3]];iter_mod=1)
end

main();