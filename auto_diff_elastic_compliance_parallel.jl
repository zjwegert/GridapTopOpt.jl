using Gridap, Gridap.TensorValues, Gridap.Geometry, Gridap.FESpaces, Gridap.Helpers
using GridapDistributed
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code
using PartitionedArrays
using SparseMatricesCSR

using ChainRulesCore
# using Zygote # <- I don't think we actually need Zygote as chainrules done manually.
include("src/ChainRules.jl")

# Heaviside function
function H_η(t;η)
    M = typeof(η*t)
    if t<-η
        return zero(M)
    elseif abs(t)<=η
        return M(1/2*(1+t/η+1/pi*sin(pi*t/η)))
    elseif t>η
        return one(M)
    end
end

function DH_η(t::M;η::M) where M<:AbstractFloat
    if t<-η
        return zero(M)
    elseif abs(t)<=η
        return M(1/2/η*(1+cos(pi*t/η)))
    elseif t>η
        return zero(M)
    end
end

# Material interpolation
Base.@kwdef struct SmoothErsatzMaterialInterpolation{M<:AbstractFloat}
    η::M # Smoothing radius
    ϵₘ::M = 10^-3 # Void material multiplier
    H = x -> H_η(x,η=η)
    DH = x -> DH_η(x,η=η)
    I = φ -> (1 - H(φ)) + ϵₘ*H(φ)
    ρ = φ -> 1 - H(φ)
end

function isotropic_2d(E::M,ν::M) where M<:AbstractFloat
    λ = E*ν/((1+ν)*(1-ν)); μ = E/(2*(1+ν))
    C = [λ+2μ  λ     0
         λ    λ+2μ   0
         0     0     μ];
    SymFourthOrderTensorValue(
        C[1,1], C[3,1], C[2,1],
        C[1,3], C[3,3], C[2,3],
        C[1,2], C[3,2], C[2,2])
end

function update_labels!(e::Int,model,f_Γ::F,::T,name::String) where {
        M<:AbstractFloat,F<:Function,T<:NTuple{2,M}}
    cell_to_entity = map(local_views(model)) do model
        labels = get_face_labeling(model)
        cell_to_entity = labels.d_to_dface_to_entity[end]
        entity = maximum(cell_to_entity) + e
        # Vertices
        vtx_coords = model.grid_topology.vertex_coordinates
        vtxs_Γ = findall(isone,f_Γ.(vtx_coords))
        vtx_edge_connectivity = Array(Geometry.get_faces(model.grid_topology,0,1)[vtxs_Γ])
        # Edges
        edge_entries = [findall(x->any(x .∈  vtx_edge_connectivity[1:end.!=j]),
            vtx_edge_connectivity[j]) for j = 1:length(vtx_edge_connectivity)]
        edge_Γ = unique(reduce(vcat,getindex.(vtx_edge_connectivity,edge_entries),init=[]))
        labels.d_to_dface_to_entity[1][vtxs_Γ] .= entity
        labels.d_to_dface_to_entity[2][edge_Γ] .= entity
        add_tag!(labels,name,[entity])
        cell_to_entity
    end
    cell_gids=get_cell_gids(model)
    cache=GridapDistributed.fetch_vector_ghost_values_cache(cell_to_entity,partition(cell_gids))
    GridapDistributed.fetch_vector_ghost_values!(cell_to_entity,cache)
end

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
    Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv=Vector{PetscScalar}
    assem=SparseMatrixAssembler(Tm,Tv,U,V)
    # Space for shape sensitivities
    reffe_scalar = ReferenceFE(lagrangian,Float64,order)
    # FE space for LSF 
    V_φ = TestFESpace(model,reffe_scalar;conformity=:H1)
    # FE Space for shape derivatives
    V_reg = TestFESpace(model,reffe_scalar;conformity=:H1,
            dirichlet_tags=["Gamma_N"],dirichlet_masks=[true])
    U_reg = TrialFESpace(V_reg,[0.0])
    ######################################################
    eΔ = (xmax,ymax)./el_size;
    interp = SmoothErsatzMaterialInterpolation(η = 2*maximum(eΔ))
    C = isotropic_2d(1.,0.3)
    g = VectorValue(0.,-1.0)
    φh = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25,V_φ)
    uh = zero(U)

    ## Weak form
    I = interp.I;
    DH = interp.DH

    function a(u,v,φ,dΩ,dΓ_N)
        _φh = φ_to_φₕ(φ,V_φ)
        ∫((I ∘ _φh)*(C ⊙ ε(u) ⊙ ε(v)))dΩ
    end
    a(φ,dΩ,dΓ_N) = (u,v) -> a(u,v,φ,dΩ,dΓ_N)
    l(v,φh,dΩ,dΓ_N) = ∫(v ⋅ g)dΓ_N
    l(φ,dΩ,dΓ_N) = v -> l(v,φ,dΩ,dΓ_N)
    res(u,v,φ,dΩ,dΓ_N) = a(u,v,φ,dΩ,dΓ_N) - l(v,φ,dΩ,dΓ_N)

    ## Functionals J and DJ
    J = (u,φ,dΩ,dΓ_N) -> a(u,u,φ,dΩ,dΓ_N)
    DJ = (q,u,φ,dΩ,dΓ_N) -> ∫((C ⊙ ε(u) ⊙ ε(u))*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;
    # Functional types
    J_func = Functional(J,[dΩ,dΓ_N],uh,φh)
    J_func_analytic = Functional(J,[dΩ,dΓ_N],uh,φh;DF=DJ)
    # FE state map
    J_smap, C_smaps = AffineFEStateMap(J_func,[J_func_analytic],U,V,V_φ,a,l,res;ls = LUSolver())
    # Solve
    J_smap(φh)
    
    ## Shape derivative
    # Autodiff
    dFh = compute_shape_derivative!(φh,J_smap)
    uh,φh = J_func.state # <- due to weird bug
    # Analytic
    dFh_analytic = compute_shape_derivative!(φh,C_smaps[1])
    uh,φh = J_func.state # <- due to weird bug

    abs_error = abs(dFh-dFh_analytic)
    rel_error = (abs(dFh-dFh_analytic))/abs(dFh_analytic)

    ## Hilb ext reg
    α = 4*maximum(eΔ)
    A(u,v) = α^2*∫(∇(u) ⊙ ∇(v))dΩ + ∫(u ⋅ v)dΩ;
    Hilb_assem=SparseMatrixAssembler(Tm,Tv,U_reg,V_reg)
    hilb_K = assemble_matrix(A,Hilb_assem,U_reg,V_reg)
    ## Autodiff result
    op = AffineFEOperator(U_reg,V_reg,hilb_K,-get_free_dof_values(interpolate_everywhere(dFh,U_reg)))
    dFh_Ω = solve(op)
    ## Analytic result
    op = AffineFEOperator(U_reg,V_reg,hilb_K,-get_free_dof_values(interpolate_everywhere(dFh_analytic,U_reg)))
    dFh_analytic_Ω = solve(op)

    hilb_abs_error = abs(dFh_Ω-dFh_analytic_Ω)
    hilb_rel_error = (abs(dFh_Ω-dFh_analytic_Ω))/abs(dFh_analytic_Ω)

    path = "./Results/AutoDiffTesting_Parallel";
    writevtk(Ω,path,cellfields=["phi"=>φh,
        "H(phi)"=>(interp.H ∘ φh),
        "|nabla(phi))|"=>(norm ∘ ∇(φh)),
        "uh"=>uh,
        "J′_abs_error"=>abs_error,
        "J′_rel_error"=>rel_error,
        "J′_analytic"=>dFh_analytic,
        "J′_autodiff"=>dFh,
        "hilb_abs_error"=>hilb_abs_error,
        "hilb_rel_error"=>hilb_rel_error,
        "v_J_Ω"=>dFh_analytic_Ω,
        "dJφh_Ω"=>dFh_Ω
    ])
end

with_debug() do distribute
    main((3,3),distribute)
end;

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
    
    V = FESpace(model,ReferenceFE(lagrangian,VectorValue{2,Float64},1),dirichlet_tags=["tag_5"])
    U = TrialFESpace(V,VectorValue(0,0))
    V_φ = FESpace(model,ReferenceFE(lagrangian,Float64,1))

    Ω = Triangulation(model)
    dΩ = Measure(Ω,2)

    # Weak forms
    function a(u,v,φ,dΩ)
        φh = φ_to_φₕ(φ,V_φ)
        ∫((φh)*(u⋅v))dΩ
    end
    a(φ,dΩ) = (u,v) -> a(u,v,φ,dΩ)
    l(v,φh,dΩ) = ∫(v ⋅ VectorValue(1,0))dΩ
    l(φ,dΩ) = v -> l(v,φ,dΩ)
    res(u,v,φ,dΩ) = a(u,v,φ,dΩ) - l(v,φ,dΩ)

    φh = interpolate(x->1,V_φ);
    uh = zero(U);

    _J = (u,φ,dΩ) -> ∫(1+(u⋅u)*(u⋅u)+φ)dΩ
    J = Functional(_J,dΩ,uh,φh)
    J_smap, C_smaps = AffineFEStateMap(J,typeof(J)[],U,V,V_φ,a,l,res;ls = LUSolver())
    J_smap(φh) # <- compute uh in place
    J_smap.F.F() # <- compute objective

    ## Printing
    # @show get_free_dof_values(uh)
    # println("Objective value = $(J_smap.F.F())")

    ## Shape derivative
    # @show typeof(φh)
    dFh = compute_shape_derivative!(φh,J_smap)
    # @show typeof(φh) # <- The type changes! Unclear why. Jordi?
    dφ = get_free_dof_values(dFh)

    ### Connor's implementation:
    uh,φh = J.state
    φ = get_free_dof_values(φh)
    if run_as_serial
        function _a(u,v,φ) 
            φh = φ_to_φₕ(φ,V_φ)
            ∫((φh)*(u⋅v))dΩ
        end
        _a(φ) = (u,v) -> _a(u,v,φ)
        _l(v,φh) = ∫(v ⋅ VectorValue(1,0))dΩ
        _l(φ) = v -> _l(v,φ)
        _res(u,v,φ,V_φ) = _a(u,v,φ) - _l(v,φ)

        _φ_to_u = _AffineFEStateMap(_a,_l,_res,V_φ,U,V)
        _u_to_j =  LossFunction((u,φ) -> _J(u,φ,dΩ),V_φ,U)

        _u, _u_pullback   = rrule(_φ_to_u,φ)
        _j, _j_pullback   = rrule(_u_to_j,_u,φ)
        _,  _du, _dφ₍ⱼ₎   = _j_pullback(1) # dj = 1
        _,  _dφ₍ᵤ₎        = _u_pullback(_du)
            dφ_connor     = _dφ₍ᵤ₎ + _dφ₍ⱼ₎

        dφ_connor - dφ
        return dφ,dφ_connor - dφ
    else 
        return dφ,nothing
    end
end
PartitionedArrays.consistent!(::Vector{M}) where M = nothing # <- is this OK Jordi?
_out_serial,_diff = test(run_as_serial=true);

_out,_ = test(run_as_serial=false);

@show norm(_out_serial-_out.vector_partition.items[1],Inf) 