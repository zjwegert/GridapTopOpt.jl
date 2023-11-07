using SparseMatricesCSR
using SparseArrays
using Gridap, Gridap.TensorValues, Gridap.Geometry
using GridapDistributed
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code
using PartitionedArrays
using DelimitedFiles
using LinearAlgebra
using Printf

import GridapDistributed.DistributedCellField
import GridapDistributed.DistributedFESpace
import GridapDistributed.DistributedDiscreteModel
import GridapDistributed.DistributedMeasure

include("Utilities.jl");
include("LevelSet.jl");
include("Setup.jl");

function setup_mock_2d()
    ## FE setup name and objective
    prob = mock_2d
    obj = thermal_compliance
    vf = 0.5; # Required volume fraction
    ## Material and loading
    D = 1.0 # Thermal coefficient
    g = 1.0; # Heat flow
    ## FE Parameters
    fe_order = 1;
    coord_max=(1.0,1.0);
    el=(200,200); # Elements in axial directions
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

function mock_2d(ranks,mesh_partition,order::T,coord_max::NTuple{2,M},
        el_size::NTuple{2,T},α::M) where {T<:Integer,M<:AbstractFloat}
    ## Model
    dom = (0,coord_max[1],0,coord_max[2]);
    model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size,isperiodic=(true,true));
    ## Triangulations and measures
    Ω = Triangulation(model)
    dΩ = Measure(Ω,2order)
    ## Spaces
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,reffe;conformity=:H1)
    U = TrialFESpace(V)
    ## Assembler
    Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv=Vector{PetscScalar}
    assem=SparseMatrixAssembler(Tm,Tv,U,V)
    # Space for shape sensitivities
    V_L2 = TestFESpace(model,reffe;conformity=:L2)
    # Space in Hilb. ext.-reg. endowed with H-regularity
    V_reg = TestFESpace(model,reffe;conformity=:H1)
    U_reg = TrialFESpace(V_reg)
    # FE space for LSF
    V_φ = TestFESpace(model,reffe;conformity=:H1)
    ## Build stiffness matrix for Hilbertian ext-reg
    A(u,v) = α^2*∫(∇(u) ⊙ ∇(v))dΩ + ∫(u ⋅ v)dΩ;
    Hilb_assem=SparseMatrixAssembler(Tm,Tv,U_reg,V_reg)
    K = assemble_matrix(A,Hilb_assem,U_reg,V_reg)
    ## Collect data
    solve_data = (U=U,V=V,assem=assem,V_L2=V_L2,dΩ=dΩ,dΓ_N=nothing)
    hilb_data = (U_reg=U_reg,V_reg=V_reg,assem=Hilb_assem,dΩ=dΩ,K=K)
    return model,Ω,V_φ,solve_data,hilb_data
end

function setup_mock_3d()
    ## FE setup name
    prob = mock_3d
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

function mock_3d(ranks,mesh_partition,order::T,coord_max::NTuple{3,M},
        el_size::NTuple{3,T},α::M,isperiodic) where {T<:Integer,M<:AbstractFloat}
    ## Model
    dom = (0,coord_max[1],0,coord_max[2],0,coord_max[3]);
    model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size,isperiodic=(isperiodic,isperiodic,isperiodic));
    ## Triangulations and measures
    Ω = Triangulation(model)
    dΩ = Measure(Ω,2order)
    ## Spaces
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,reffe;conformity=:H1)
    U = TrialFESpace(V)
    ## Assembler
    Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv=Vector{PetscScalar}
    assem=SparseMatrixAssembler(Tm,Tv,U,V)
    # Space for shape sensitivities
    V_L2 = TestFESpace(model,reffe;conformity=:L2)
    # Space in Hilb. ext.-reg. endowed with H-regularity
    V_reg = TestFESpace(model,reffe;conformity=:H1)
    U_reg = TrialFESpace(V_reg)
    # FE space for LSF
    V_φ = TestFESpace(model,reffe;conformity=:H1)
    ## Build stiffness matrix for Hilbertian ext-reg
    A(u,v) = α^2*∫(∇(u) ⊙ ∇(v))dΩ + ∫(u ⋅ v)dΩ;
    Hilb_assem=SparseMatrixAssembler(Tm,Tv,U_reg,V_reg)
    K = assemble_matrix(A,Hilb_assem,U_reg,V_reg)
    ## Collect data
    solve_data = (U=U,V=V,assem=assem,V_L2=V_L2,dΩ=dΩ,dΓ_N=nothing)
    hilb_data = (U_reg=U_reg,V_reg=V_reg,assem=Hilb_assem,dΩ=dΩ,K=K)
    return model,Ω,V_φ,solve_data,hilb_data
end

mesh_partition = (3,3,3);
isperiodic = true;
prob_setup = setup_mock_3d;

ranks = with_debug() do distribute 
    ranks = distribute(LinearIndices((prod(mesh_partition),)));
end

prob,obj,lsf,vf,mat,g,fe_order,coord_max,el,η_coeff,α_coeff,_,_,_,γ,steps,reinit_tol = prob_setup()  
Δ = coord_max./el # Element size
η = η_coeff*maximum(Δ); α = α_coeff*maximum(Δ) # Smoothing parameters
interp = SmoothErsatzMaterialInterpolation(η=η) # Material interpolation
model,Ω,V_φ,solve_data,hilb_data = prob(ranks,mesh_partition,fe_order,coord_max,el,α,isperiodic);

φh = interpolate(lsf,V_φ)
φ = get_free_dof_values(φh)
reinit!(φ,model,Δ,0.5,2000,reinit_tol)

writevtk(Ω,(@__DIR__)*"\\Results\\test_reinit_3d_periodic",cellfields=["phi"=>φh,"H(phi)"=>(interp.H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh))]);