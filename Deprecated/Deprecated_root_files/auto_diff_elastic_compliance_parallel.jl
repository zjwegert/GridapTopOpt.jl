using Gridap, Gridap.TensorValues, Gridap.Geometry, Gridap.FESpaces, Gridap.Helpers
using GridapDistributed
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code
using PartitionedArrays
using SparseMatricesCSR
using ChainRulesCore

include("src/ChainRules.jl")
include("src/Utilities.jl")
include("src/MaterialInterpolation.jl")

######################################################
## FE Setup
function main(mesh_partition,distribute)
    ranks  = distribute(LinearIndices((prod(mesh_partition),)))
    order = 1;
    el_size = (200,200);
    dom = (0.,1.,0.,1.);
    coord_max = dom[2],dom[4]
    model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
    ## Define Γ_N and Γ_D
    xmax,ymax = coord_max
    prop_Γ_N = 0.4
    f_Γ_D(x) = (x[1] ≈ 0.0) ? true : false;
    f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= 
        ymax/2+ymax*prop_Γ_N/4 + eps()) ? true : false;
    update_labels!(1,model,f_Γ_D,"Gamma_D")
    update_labels!(2,model,f_Γ_N,"Gamma_N")
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
    # Space for shape sensitivities
    reffe_scalar = ReferenceFE(lagrangian,Float64,order)
    # FE space for LSF 
    V_φ = TestFESpace(model,reffe_scalar;conformity=:H1)
    # FE Space for shape derivatives
    V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
    U_reg = TrialFESpace(V_reg,0.0)
    ######################################################
    eΔ = (xmax,ymax)./el_size;
    interp = SmoothErsatzMaterialInterpolation(η = 2*maximum(eΔ))
    C = isotropic_2d(1.,0.3)
    g = VectorValue(0.,-1.0)
    φh = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25,V_φ)

    ## Weak form
    I = interp.I;
    DH = interp.DH

    a(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(v)))dΩ
    l(v,φh,dΩ,dΓ_N) = ∫(v ⋅ g)dΓ_N
    res(u,v,φ,dΩ,dΓ_N) = a(u,v,φ,dΩ,dΓ_N) - l(v,φ,dΩ,dΓ_N)

    ## Functionals J and DJ
    J = (u,φ,dΩ,dΓ_N) -> a(u,u,φ,dΩ,dΓ_N)
    DJ = (q,u,φ,dΩ,dΓ_N) -> ∫((C ⊙ ε(u) ⊙ ε(u))*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;
    state_map = AffineFEStateMap(a,l,res,U,V,V_φ,U_reg,φh,dΩ,dΓ_N)
    pcfs = PDEConstrainedFunctionals(J,Function[J],state_map,analytic_dC=[DJ])

    φ = get_free_dof_values(φh)
    j,c,dJ,dC = Gridap.evaluate!(pcfs,φ)
    uh = get_state(pcfs)

    println("J = $j | C = $c")
    
    ## Shape derivative
    # Autodiff
    dF = dJ
    # Analytic
    dF_analytic = first(dC)

    abs_error = maximum(abs,dF-dF_analytic)
    rel_error = (abs_error)/maximum(abs,dF_analytic)

    ## Hilb ext reg
    α = 4*maximum(eΔ)
    A(u,v) = α^2*∫(∇(u) ⊙ ∇(v))dΩ + ∫(u ⋅ v)dΩ;
    Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv=Vector{PetscScalar}
    Hilb_assem=SparseMatrixAssembler(Tm,Tv,U_reg,V_reg)
    hilb_K = assemble_matrix(A,Hilb_assem,U_reg,V_reg)
    ## Autodiff result
    op = AffineFEOperator(U_reg,V_reg,hilb_K,-dF)
    dF_Ω = get_free_dof_values(solve(op))
    ## Analytic result
    op = AffineFEOperator(U_reg,V_reg,hilb_K,-dF_analytic)
    dF_analytic_Ω = get_free_dof_values(solve(op))

    hilb_abs_error = maximum(abs,dF_Ω-dF_analytic_Ω)
    hilb_rel_error = (hilb_abs_error)/maximum(abs,dF_analytic_Ω)
    # path = dirname(dirname(@__DIR__))*"/results/AutoDiffTesting_Parallel";
    # writevtk(Ω,path,cellfields=["phi"=>φh,
    #     "H(phi)"=>(interp.H ∘ φh),
    #     "|nabla(phi))|"=>(norm ∘ ∇(φh)),
    #     "uh"=>uh,
    #     "J′_analytic"=>FEFunction(U_reg,dF_analytic),
    #     "J′_autodiff"=>FEFunction(U_reg,dF),
    #     "v_J_Ω"=>FEFunction(U_reg,dF_analytic_Ω),
    #     "dJφh_Ω"=>FEFunction(U_reg,dF_Ω)
    # ])
    abs_error,rel_error,hilb_abs_error,hilb_rel_error
end

out = with_debug() do distribute
    main((3,3),distribute)
end

####################
#   Debug Testing  #
####################
using BenchmarkTools
include("ChainRules.jl");
function test(;run_as_serial::Bool=true)
    ranks = with_debug() do distribute
        distribute(LinearIndices((1,)))
    end

    model = if run_as_serial
        CartesianDiscreteModel((0,1,0,1),(2,2))
    else
        CartesianDiscreteModel(ranks,(1,1),(0,1,0,1),(2,2))
    end

    Ω = Triangulation(model)
    dΩ = Measure(Ω,2)
    V_φ = FESpace(model,ReferenceFE(lagrangian,Float64,1))
    V_reg = FESpace(model,ReferenceFE(lagrangian,Float64,1),dirichlet_tags=["tag_1"])
    U_reg = TrialFESpace(V_reg,1)
    φh = interpolate(x->1,V_φ);
    φ = get_free_dof_values(φh)
    
    # FE Problem
    V = FESpace(model,ReferenceFE(lagrangian,Float64,1),dirichlet_tags=["boundary"])
    U = TrialFESpace(V,2)
    a(u,v,φ,dΩ) = ∫( φ*∇(v)⋅∇(u) )*dΩ
    l(v,φ,dΩ) = ∫(v)*dΩ
    res(u,v,φ,dΩ) = a(u,v,φ,dΩ) - l(v,φ,dΩ)

    J = (u,φ,dΩ) -> ∫(1+(u⋅u)*(u⋅u)+sqrt ∘ φ)dΩ
    state_map = AffineFEStateMap(a,l,res,U,V,V_φ,U_reg,φh,dΩ)
    pcfs = PDEConstrainedFunctionals(J,state_map)


    j,c,dJ,dC = Gridap.evaluate!(pcfs,φ)

    println("J = $j | C = $c")

    ### Connor's implementation:
    if run_as_serial
        function _a(u,v,φ) 
            __φh = φ_to_φₕ(φ,V_φ)
            a(u,v,__φh,dΩ)
        end
        function _l(v,φ) 
            __φh = φ_to_φₕ(φ,V_φ)
            l(v,__φh,dΩ)
        end
        _a(φ) = (u,v) -> _a(u,v,φ)
        _l(φ) = v -> _l(v,φ)
        _res(u,v,φ) = _a(u,v,φ) - _l(v,φ)

        _φ_to_u = _AffineFEStateMap(_a,_l,_res,V_φ,U,V)
        _u_to_j =  LossFunction((u,φ) -> J(u,φ,dΩ),V_φ,U)

        _u, _u_pullback   = rrule(_φ_to_u,φ)
        _j, _j_pullback   = rrule(_u_to_j,_u,φ)
        _,  _du, _dφ₍ⱼ₎   = _j_pullback(1) # dj = 1
        _,  _dφ₍ᵤ₎        = _u_pullback(_du)
            dφ_connor     = _dφ₍ᵤ₎ + _dφ₍ⱼ₎
        if ~(V_φ === U_reg)
            dφ_connor = get_free_dof_values(interpolate(FEFunction(V_φ,dφ_connor),U_reg))
        end

        return dJ,dJ-dφ_connor
    else 
        return dJ,nothing
    end
end

_out_serial,_diff = test(run_as_serial=true);

_out,_ = test(run_as_serial=false);

@show norm(_out_serial-_out.vector_partition.items[1],Inf)

"""
assemble_matrix_and_vector!:

	Line 310 Assemblers.jl: assemble_matrix_and_vector!(f::Function,b::Function,M::AbstractMatrix,r::AbstractVector,a::Assembler,U::FESpace,V::FESpace)

	-> Calls Line 453 Assemblers.jl: data = collect_cell_matrix_and_vector(trial::FESpace,test::FESpace,biform::DomainContribution,liform::DomainContribution)

	-> Line 85 SparseMatrixAssembler.jl: assemble_matrix_and_vector!(A,b,a::SparseMatrixAssembler, data) -> assemble_matrix_and_vector_add!...

AffineFEOperator:

	Line 23 AffineFEOperator.jl: AffineFEOperator(weakform::Function,trial::FESpace,test::FESpace,assem::Assembler)
	
	-> Line 464 Assemblers.jl: data = collect_cell_matrix_and_vector(trial::FESpace,test::FESpace,biform::DomainContribution,liform::DomainContribution,uhd::FEFunction)
	
	-> Line 244 Assemblers.jl: assemble_matrix_and_vector(a::Assembler,data)
	
	->-> Line 73 SparseMatrixAssembler.jl: A,b = allocate_matrix_and_vector(a,data)
	
	->-> Line 85 SparseMatrixAssembler.jl: assemble_matrix_and_vector!(A,b,a::SparseMatrixAssembler, data) -> assemble_matrix_and_vector_add!...
	
Why does assemble_matrix_and_vector! at the top not take uhd???
"""