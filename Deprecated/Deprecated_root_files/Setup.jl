## User defined setups

##########################
#   Thermal Compliance   #
##########################
## Objective and shape sensitivity
function li2019_mumps_setup(ksp)
    pc       = Ref{GridapPETSc.PETSC.PC}()
    mumpsmat = Ref{GridapPETSc.PETSC.Mat}()
    @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
    @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPPREONLY)
    @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
    @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCLU)
    @check_error_code GridapPETSc.PETSC.PCFactorSetMatSolverType(pc[],GridapPETSc.PETSC.MATSOLVERMUMPS)
    @check_error_code GridapPETSc.PETSC.PCFactorSetUpMatSolverType(pc[])
    @check_error_code GridapPETSc.PETSC.PCFactorGetMatrix(pc[],mumpsmat)
    @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  4, 1)
    @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  7, 0)
  # @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 14, 1000)
    @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 28, 2)
    @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 2)
    @check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[], 3, 1.0e-6)
end

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
    ls = PETScLinearSolver(li2019_mumps_setup)
    uh = solve(ls,op)
    u = correct_ghost_layout(uh,K.col_partition)
    ## Compute J and v_J
    J = dot(u,(K*u))
    v_J = interpolate(-D*∇(uh)⋅∇(uh),solve_data.V_L2)
    return J,v_J,uh
end

## Problem setups
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
    el=(800,800); # Elements in axial directions
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
    g = 3.0 # Heat flow
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
    ## Collect data
    solve_data = (U=U,V=V,assem=assem,V_L2=V_L2,dΩ=dΩ,dΓ_N=dΓ_N)
    hilb_data = (U_reg=U_reg,V_reg=V_reg,assem=Hilb_assem,dΩ=dΩ,K=K)
    return model,Ω,V_φ,solve_data,hilb_data
end

function pipe2_setup_3d()
    ## FE setup name
    prob = pipe2_3d
    obj = thermal_compliance
    vf = 0.3; # Required volume fraction
    ## Material and loading
    D = 1.0 # Thermal coefficient
    g = 5.0 # Heat flow
    ## FE Parameters
    fe_order = 1;
    coord_max=(1.0,1.0,1.0);
    el=(100,100,100); # Elements in axial directions
    ## LS Paramters
    lsf = gen_lsf(2,0.4); # Initial LSF
    η_coeff = 2 # Interpolation radius coefficent
    α_coeff = 4 # Hilbertian smoothing coefficent
    μ=0.005 # Line search parameter
    γ,γ_min,γ_max = [0.1,0.001,0.1] # CFL Coefficent
    steps = Int(floor(minimum(el)/3)); # Finite diff. steps
    reinit_tol = min(1000*prod(inv,el),10^-3) # Reinit tol.

    return prob,obj,lsf,vf,D,g,fe_order,coord_max,el,
        η_coeff,α_coeff,μ,γ_max,γ_min,γ,steps,reinit_tol
end

function pipe2_3d(ranks,mesh_partition,order::T,coord_max::NTuple{3,M},
        el_size::NTuple{3,T},α::M) where {T<:Integer,M<:AbstractFloat}
    ## Model
    @assert isone(coord_max[1]) && isone(coord_max[2]) && isone(coord_max[3])
    dom = (0,coord_max[1],0,coord_max[2],0,coord_max[3]);
    model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
    ## Define Γ_N and Γ_D
    f_Γ_D(x) = (max(abs(x[1]-0.2),abs(x[2]-0.5),abs(x[3]-0.5)) <= 0.05+eps() ||
        max(abs(x[1]-0.8),abs(x[2]-0.5),abs(x[3]-0.5)) <= 0.05+eps() ||
        max(abs(x[1]-0.5),abs(x[2]-0.2),abs(x[3]-0.5)) <= 0.05+eps() ||
        max(abs(x[1]-0.5),abs(x[2]-0.8),abs(x[3]-0.5)) <= 0.05+eps()) ? true : false;
    f_Γ_N(x) = (max(abs(x[1]),abs(x[2]),abs(x[3])) <= 0.05+eps() && (iszero(x[1]) || iszero(x[2]) || iszero(x[3])))||
        (max(abs(x[1]-1),abs(x[2]),abs(x[3])) <= 0.05+eps() && (isone(x[1]) || iszero(x[2]) || iszero(x[3])))||
        (max(abs(x[1]),abs(x[2]-1),abs(x[3])) <= 0.05+eps() && (iszero(x[1]) || isone(x[2]) || iszero(x[3])))||
        (max(abs(x[1]-1),abs(x[2]-1),abs(x[3])) <= 0.05+eps() && (isone(x[1]) || isone(x[2]) || iszero(x[3])))||
        (max(abs(x[1]),abs(x[2]),abs(x[3]-1)) <= 0.05+eps() && (iszero(x[1]) || iszero(x[2]) || isone(x[3])))||
        (max(abs(x[1]-1),abs(x[2]),abs(x[3]-1)) <= 0.05+eps() && (isone(x[1]) || iszero(x[2]) || isone(x[3])))||
        (max(abs(x[1]),abs(x[2]-1),abs(x[3]-1)) <= 0.05+eps() && (iszero(x[1]) || isone(x[2]) || isone(x[3])))||
        (max(abs(x[1]-1),abs(x[2]-1),abs(x[3]-1)) <= 0.05+eps() && (isone(x[1]) || isone(x[2]) || isone(x[3]))) ? true : false;
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
        dirichlet_tags=["Gamma_N","Gamma_D"],dirichlet_masks=[true,true])
    U = TrialFESpace(V,[0.0,0.0])
    ## Assembler
    Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv=Vector{PetscScalar}
    assem=SparseMatrixAssembler(Tm,Tv,U,V)
    # Space for shape sensitivities
    V_L2 = TestFESpace(model,reffe;conformity=:L2)
    # Space in Hilb. ext.-reg. endowed with H-regularity
    V_reg = TestFESpace(model,reffe;conformity=:H1,
        dirichlet_tags=["Gamma_N","Gamma_D"],dirichlet_masks=[true,true])
    U_reg = TrialFESpace(V_reg,[0.0,0.0])
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

##########################
#   Elastic Compliance   #
##########################
## Objective and shape sensitivity
function elastic_compliance(φh::DistributedCellField,g,C::S,solve_data::NT,interp::T) where {
        S<:SymFourthOrderTensorValue,T<:SmoothErsatzMaterialInterpolation,NT<:NamedTuple}
    I = interp.I; dΩ=solve_data.dΩ; dΓ_N=solve_data.dΓ_N;
    ## Weak form
    a(u,v) = ∫((I ∘ φh)*(C ⊙ ε(u) ⊙ ε(v)))dΩ
    l(v) = ∫(v ⋅ g)dΓ_N
    ## Solve finite element problem
    op = AffineFEOperator(a,l,solve_data.U,solve_data.V,solve_data.assem)
    K = op.op.matrix;
    ## Solve
    ls = PETScLinearSolver()
    uh = solve(ls,op)
    u = correct_ghost_layout(uh,K.col_partition)
    ## Compute J and v_J
    J = dot(u,(K*u))
    v_J = interpolate(-C ⊙ ε(uh) ⊙ ε(uh),solve_data.V_L2)
    return J,v_J,uh
end

function cantilever_setup_2d()
    ## FE setup name
    prob = cantilever_2d
    obj = elastic_compliance
    vf = 0.5; # Required volume fraction
    ## Material and loading
    C = isotropic_2d(1.0,0.3) # Stiffness tensor
    g = VectorValue(0,-0.1); # Loading
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
    g = VectorValue(0,0,-0.1); # Loading
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

##########################
#         Volume         #
##########################
function vol(φh::DistributedCellField,interp::T,dΩ::ME,V_L2::V,vol_D::M) where {
        M<:AbstractFloat,T<:SmoothErsatzMaterialInterpolation,
        V<:DistributedFESpace,ME<:DistributedMeasure}
    ρ = interp.ρ;
    vol = sum(∫(ρ ∘ φh)dΩ)/vol_D;
    v_vol = interpolate(x->one(M)/vol_D,V_L2)
    return vol,v_vol
end