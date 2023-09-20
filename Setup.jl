## User defined setups
function pipe_setup_2d()
    ## FE setup name and objective
    prob = pipe_2d
    obj = thermal_compliance
    vf = 0.5; # Required volume fraction
    ## Material and loading
    D = 1.0 # Thermal coefficient
    g = 1.0; # Heat flow
    ## FE Parameters
    fe_order = 1;
    coord_max=(1.0,1.0);
    el=(400,400); # Elements in axial directions
    ## LS Paramters
    lsf = gen_lsf(4,0.2); # Initial LSF
    η_coeff = 2 # Interpolation radius coefficent
    α_coeff = 4 # Hilbertian smoothing coefficent
    μ=0.005 # Line search parameter
    γ,γ_min,γ_max = [0.05,0.001,0.1] # CFL Coefficent
    steps = Int(floor(minimum(el)/10)) # Finite diff. steps
    reinit_tol = min(4*prod(inv,el),10^-4) # Reinit tol.

    return prob,obj,lsf,vf,D,g,fe_order,coord_max,el,
        η_coeff,α_coeff,μ,γ_max,γ_min,γ,steps,reinit_tol
end

function pipe_setup_3d()
    ## FE setup name
    prob = pipe_3d
    obj = thermal_compliance
    vf = 0.5; # Required volume fraction
    ## Material and loading
    D = 1.0 # Thermal coefficient
    g = 3.0 #6.0; # Heat flow
    ## FE Parameters
    fe_order = 1;
    coord_max=(1.0,1.0,1.0);
    el=(100,100,100); # Elements in axial directions
    ## LS Paramters
    lsf = gen_lsf(4,0.2); # Initial LSF
    η_coeff = 2 # Interpolation radius coefficent
    α_coeff = 4 # Hilbertian smoothing coefficent
    μ=0.005 # Line search parameter
    γ,γ_min,γ_max = [0.1,0.001,0.1] # CFL Coefficent
    steps = Int(floor(minimum(el)/3)); # Finite diff. steps
    reinit_tol = min(1000*prod(inv,el),10^-3) # Reinit tol.

    return prob,obj,lsf,vf,D,g,fe_order,coord_max,el,
        η_coeff,α_coeff,μ,γ_max,γ_min,γ,steps,reinit_tol
end

function cantilever_setup_2d()
    ## FE setup name
    prob = cantilever_2d
    obj = elastic_compliance
    vf = 0.5; # Required volume fraction
    ## Material and loading
    C = isotropic_2d(1.0,0.3) # Stiffness tensor
    g = VectorValue(0,-0.5); # Loading
    ## FE Parameters
    fe_order = 1;
    coord_max=(2.0,1.0);
    el=(800,400); # Elements in axial directions
    ## LS Paramters
    lsf = gen_lsf(4,0.2); # Initial LSF
    η_coeff = 2 # Interpolation radius coefficent
    α_coeff = 4 # Hilbertian smoothing coefficent
    μ=0.005 # Line search parameter
    γ,γ_min,γ_max = [0.1,0.001,0.1] # CFL Coefficent
    steps = Int(floor(minimum(el)/10)) # Finite diff. steps
    reinit_tol = min(4*prod(inv,el),10^-4) # Reinit tol.

    return prob,obj,lsf,vf,C,g,fe_order,coord_max,el,
        η_coeff,α_coeff,μ,γ_max,γ_min,γ,steps,reinit_tol
end

function cantilever_setup_3d()
    ## FE setup name
    prob = cantilever_3d
    obj = elastic_compliance
    vf = 0.5; # Required volume fraction
    ## Material and loading
    C = isotropic_3d(1.0,0.3) # Stiffness tensor
    g = VectorValue(0,0,-0.5); # Loading
    ## FE Parameters
    fe_order = 1;
    coord_max=(2.0,1.0,1.0);
    el=(100,50,50); # Elements in axial directions
    ## LS Paramters
    lsf = gen_lsf(4,0.2); # Initial LSF
    η_coeff = 2 # Interpolation radius coefficent
    α_coeff = 4 # Hilbertian smoothing coefficent
    μ=0.005 # Line search parameter
    γ,γ_min,γ_max = [0.1,0.001,0.1] # CFL Coefficent
    steps = Int(floor(minimum(el)/3)); # Finite diff. steps
    reinit_tol = min(1000*prod(inv,el),10^-3) # Reinit tol.

    return prob,obj,lsf,vf,C,g,fe_order,coord_max,el,
        η_coeff,α_coeff,μ,γ_max,γ_min,γ,steps,reinit_tol
end

## FE Setups
function pipe_2d(ranks,mesh_partition,order::T,coord_max::NTuple{2,M},
        el_size::NTuple{2,T},α::M) where {T<:Integer,M<:AbstractFloat}
    ## Model
    dom = (0,coord_max[1],0,coord_max[2]);
    model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
    ## Define Γ_N and Γ_D
    xmax,ymax = coord_max
    prop_Γ_N,prop_Γ_D = [0.4,0.2]
    f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= ymax*prop_Γ_D + eps() || 
        x[2] >= ymax-ymax*prop_Γ_D - eps())) ? true : false;
    f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= 
        ymax/2+ymax*prop_Γ_N/4 + eps()) ? true : false;
    update_labels!(1,model,f_Γ_D,coord_max,"Gamma_D")
    update_labels!(2,model,f_Γ_N,coord_max,"Gamma_N")
    ## Triangulations and measures
    Ω = Triangulation(model)
    Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
    dΩ = Measure(Ω,2order)
    dΓ_N = Measure(Γ_N,2order)
    ## Spaces
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,reffe;conformity=:H1,
        dirichlet_tags=["Gamma_D"],dirichlet_masks=[true])
    U = TrialFESpace(V,[0.0])
    ## Assembler
    Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv=Vector{PetscScalar}
    assem=SparseMatrixAssembler(Tm,Tv,U,V)
    # Space for shape sensitivities
    V_L2 = TestFESpace(model,reffe;conformity=:L2)
    # Space in Hilb. ext.-reg. endowed with H-regularity
    V_reg = TestFESpace(model,reffe;conformity=:H1,
        dirichlet_tags=["Gamma_N"],dirichlet_masks=[true])
    U_reg = TrialFESpace(V_reg,[0.0])
    # FE space for LSF
    V_φ = TestFESpace(model,reffe;conformity=:H1)
    ## Build stiffness matrix for Hilbertian ext-reg
    A(u,v) = α^2*∫(∇(u) ⊙ ∇(v))dΩ + ∫(u ⋅ v)dΩ;
    Hilb_assem=SparseMatrixAssembler(Tm,Tv,U_reg,V_reg)
    K = assemble_matrix(A,Hilb_assem,U_reg,V_reg)
    ## Collect data
    solve_data = (U=U,V=V,assem=assem,V_L2=V_L2,dΩ=dΩ,dΓ_N=dΓ_N)
    hilb_data = (U_reg=U_reg,V_reg=V_reg,assem=Hilb_assem,dΩ=dΩ,K=K)
    return model,Ω,V_φ,solve_data,hilb_data
end

function pipe_3d(ranks,mesh_partition,order::T,coord_max::NTuple{3,M},
        el_size::NTuple{3,T},α::M) where {T<:Integer,M<:AbstractFloat}
    ## Model
    dom = (0,coord_max[1],0,coord_max[2],0,coord_max[3]);
    model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
    ## Define Γ_N and Γ_D
    xmax,ymax,zmax = coord_max
    prop_Γ_N = 0.4
    prop_Γ_D = 0.2
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
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,reffe;conformity=:H1,
        dirichlet_tags=["Gamma_D"],dirichlet_masks=[true])
    U = TrialFESpace(V,[0.0])
    ## Assembler
    Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv=Vector{PetscScalar}
    assem=SparseMatrixAssembler(Tm,Tv,U,V)
    # Space for shape sensitivities
    V_L2 = TestFESpace(model,reffe;conformity=:L2)
    # Space in Hilb. ext.-reg. endowed with H-regularity
    V_reg = TestFESpace(model,reffe;conformity=:H1,
        dirichlet_tags=["Gamma_N"],dirichlet_masks=[true])
    U_reg = TrialFESpace(V_reg,[0.0])
    # FE space for LSF 
    V_φ = TestFESpace(model,reffe;conformity=:H1)
    ## Build stiffness matrix for Hilbertian ext-reg
    A(u,v) = α^2*∫(∇(u) ⊙ ∇(v))dΩ + ∫(u ⋅ v)dΩ;
    Hilb_assem=SparseMatrixAssembler(Tm,Tv,U_reg,V_reg)
    K = assemble_matrix(A,Hilb_assem,U_reg,V_reg)
    # K = assemble_matrix(A,U_reg,V_reg)
    ## Collect data
    solve_data = (U=U,V=V,assem=assem,V_L2=V_L2,dΩ=dΩ,dΓ_N=dΓ_N)
    hilb_data = (U_reg=U_reg,V_reg=V_reg,assem=Hilb_assem,dΩ=dΩ,K=K)
    return model,Ω,V_φ,solve_data,hilb_data
end

function cantilever_2d(ranks,mesh_partition,order::T,coord_max::NTuple{2,M},
        el_size::NTuple{2,T},α::M) where {T<:Integer,M<:AbstractFloat}
    ## Model
    dom = (0,coord_max[1],0,coord_max[2]);
    model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
    ## Define Γ_N and Γ_D
    xmax,ymax = coord_max
    prop_Γ_N = 0.4
    f_Γ_D(x) = (x[1] ≈ 0.0) ? true : false;
    f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= 
        ymax/2+ymax*prop_Γ_N/4 + eps()) ? true : false;
    update_labels!(1,model,f_Γ_D,coord_max,"Gamma_D")
    update_labels!(2,model,f_Γ_N,coord_max,"Gamma_N")
    ## Triangulations and measures
    Ω = Triangulation(model)
    Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
    dΩ = Measure(Ω,2order)
    dΓ_N = Measure(Γ_N,2order)
    ## Spaces
    reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
    V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["Gamma_D"],
        dirichlet_masks=[(true,true)],vector_type=Vector{PetscScalar})
    U = TrialFESpace(V,[VectorValue(0.0,0.0)])
    ## Assembler
    Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv=Vector{PetscScalar}
    assem=SparseMatrixAssembler(Tm,Tv,U,V)
    # Space for shape sensitivities
    reffe_scalar = ReferenceFE(lagrangian,Float64,order)
    V_L2 = TestFESpace(model,reffe_scalar;conformity=:L2)
    # Space in Hilb. ext.-reg. endowed with H-regularity
    V_reg = TestFESpace(model,reffe_scalar;conformity=:H1,
        dirichlet_tags=["Gamma_N"],dirichlet_masks=[true])
    U_reg = TrialFESpace(V_reg,[0.0])
    # FE space for LSF 
    V_φ = TestFESpace(model,reffe_scalar;conformity=:H1)
    ## Build stiffness matrix for Hilbertian ext-reg
    A(u,v) = α^2*∫(∇(u) ⊙ ∇(v))dΩ + ∫(u ⋅ v)dΩ;
    Hilb_assem=SparseMatrixAssembler(Tm,Tv,U_reg,V_reg)
    K = assemble_matrix(A,Hilb_assem,U_reg,V_reg)
    ## Collect data
    solve_data = (U=U,V=V,assem=assem,V_L2=V_L2,dΩ=dΩ,dΓ_N=dΓ_N)
    hilb_data = (U_reg=U_reg,V_reg=V_reg,assem=Hilb_assem,dΩ=dΩ,K=K)
    return model,Ω,V_φ,solve_data,hilb_data
end

function cantilever_3d(ranks,mesh_partition,order::T,coord_max::NTuple{3,M},
        el_size::NTuple{3,T},α::M) where {T<:Integer,M<:AbstractFloat}
    ## Model
    dom = (0,coord_max[1],0,coord_max[2],0,coord_max[3]);
    model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
    ## Define Γ_N and Γ_D
    xmax,ymax,zmax = coord_max
    prop_Γ_N = 0.4
    f_Γ_D(x) = (x[1] ≈ 0.0) ? true : false;
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
    reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
    V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["Gamma_D"],
        dirichlet_masks=[(true,true,true)])
    U = TrialFESpace(V,[VectorValue(0.0,0.0,0.0)])
    ## Assembler
    Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv=Vector{PetscScalar}
    assem=SparseMatrixAssembler(Tm,Tv,U,V)
    # Space for shape sensitivities
    reffe_scalar = ReferenceFE(lagrangian,Float64,order)
    V_L2 = TestFESpace(model,reffe_scalar;conformity=:L2)
    # Space in Hilb. ext.-reg. endowed with H-regularity
    V_reg = TestFESpace(model,reffe_scalar;conformity=:H1,
        dirichlet_tags=["Gamma_N"],dirichlet_masks=[true])
    U_reg = TrialFESpace(V_reg,[0.0])
    # FE space for LSF 
    V_φ = TestFESpace(model,reffe_scalar;conformity=:H1)
    ## Build stiffness matrix for Hilbertian ext-reg
    A(u,v) = α^2*∫(∇(u) ⊙ ∇(v))dΩ + ∫(u ⋅ v)dΩ;
    Hilb_assem=SparseMatrixAssembler(Tm,Tv,U_reg,V_reg)
    K = assemble_matrix(A,Hilb_assem,U_reg,V_reg)
    ## Collect data
    solve_data = (U=U,V=V,assem=assem,V_L2=V_L2,dΩ=dΩ,dΓ_N=dΓ_N)
    hilb_data = (U_reg=U_reg,V_reg=V_reg,assem=Hilb_assem,dΩ=dΩ,K=K)
    return model,Ω,V_φ,solve_data,hilb_data
end
