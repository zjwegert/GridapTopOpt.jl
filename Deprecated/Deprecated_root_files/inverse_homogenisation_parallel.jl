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
    el_size = (100,100);
    dom = (0.,1.,0.,1.);
    coord_max = dom[2],dom[4]
    model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size,isperiodic=(true,true));
    ## Define Γ_N and Γ_D
    xmax,ymax = coord_max
    f_Γ_D(x) = iszero(x)
    update_labels!(1,model,f_Γ_D,"origin")
    ## Triangulations and measures
    Ω = Triangulation(model)
    dΩ = Measure(Ω,2order)
    ## Spaces
    reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
    _V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["origin"])
    _U = TrialFESpace(_V,[VectorValue(0.0,0.0)])
    V = MultiFieldFESpace([_V,_V,_V])
    U = MultiFieldFESpace([_U,_U,_U])
    # Space for shape sensitivities
    reffe_scalar = ReferenceFE(lagrangian,Float64,order)
    # FE space for LSF 
    V_φ = TestFESpace(model,reffe_scalar;conformity=:H1)
    # FE Space for shape derivatives
    V_reg = TestFESpace(model,reffe_scalar;conformity=:H1)
    U_reg = TrialFESpace(V_reg)
    ######################################################
    eΔ = (xmax,ymax)./el_size;
    interp = SmoothErsatzMaterialInterpolation(η = 2*maximum(eΔ))
    C = isotropic_2d(1.,0.3)
    φh = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25,V_φ)

    ## Weak form
    I = interp.I;
    DH = interp.DH

    εᴹ = (TensorValue(1.,0.,0.,0.),     # ϵᵢⱼ⁽¹¹⁾≡ϵᵢⱼ⁽¹⁾
          TensorValue(0.,0.,0.,1.),     # ϵᵢⱼ⁽²²⁾≡ϵᵢⱼ⁽²⁾
          TensorValue(0.,1/2,1/2,0.))   # ϵᵢⱼ⁽¹²⁾≡ϵᵢⱼ⁽³⁾

    a((u1,u2,u3),(v1,v2,v3),φ,dΩ) = ∫((I ∘ φ)*(C ⊙ ε(u1) ⊙ ε(v1)) + 
                                    (I ∘ φ)*(C ⊙ ε(u2) ⊙ ε(v2)) + 
                                    (I ∘ φ)*(C ⊙ ε(u3) ⊙ ε(v3)))dΩ
    l((v1,v2,v3),φ,dΩ) = ∫(-(I ∘ φ)*(C ⊙ εᴹ[1] ⊙ ε(v1)) - 
                            (I ∘ φ)*(C ⊙ εᴹ[2] ⊙ ε(v2)) -
                            (I ∘ φ)*(C ⊙ εᴹ[3] ⊙ ε(v3)))dΩ;

    res(u,v,φ,dΩ) = a(u,v,φ,dΩ) - l(v,φ,dΩ)

    _C(ε_p,ε_q) = C ⊙ ε_p ⊙ ε_q;
    _K((u1,u2,u3),εᴹ) = (_C(ε(u1)+εᴹ[1],εᴹ[1]) + _C(ε(u2)+εᴹ[2],εᴹ[2]) + 2*_C(ε(u1)+εᴹ[1],εᴹ[2]))/4
    _v_K((u1,u2,u3),εᴹ) = (_C(ε(u1)+εᴹ[1],ε(u1)+εᴹ[1]) + _C(ε(u2)+εᴹ[2],ε(u2)+εᴹ[2]) + 2*_C(ε(u1)+εᴹ[1],ε(u2)+εᴹ[2]))/4

    K_mod = (u,φ,dΩ) -> ∫((I ∘ φ)*_K(u,εᴹ))dΩ;
    DK_mod = (q,u,φ,dΩ) -> ∫(-_v_K(u,εᴹ)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;

    state_map = AffineFEStateMap(a,l,res,U,V,V_φ,U_reg,φh,dΩ)
    pcfs = PDEConstrainedFunctionals(K_mod,[K_mod],state_map,analytic_dC=[DK_mod])

    φ = get_free_dof_values(φh)
    j,c,dJ,dC = Gridap.evaluate!(pcfs,φ)
    uh = get_state(pcfs)

    dFh = interpolate(FEFunction(V_φ,dJ),U_reg)
    dF = get_free_dof_values(dFh) 

    dFh_analytic = interpolate(FEFunction(V_φ,first(dC)),U_reg)
    dF_analytic = get_free_dof_values(dFh_analytic)

    pre_hilb_abs_error = maximum(abs,dF-dF_analytic)
    pre_hilb_rel_error = maximum(abs,dF-dF_analytic)/maximum(abs,dF_analytic)

    abs_error = abs(dFh-dFh_analytic)
    rel_error = (abs(dFh-dFh_analytic))/abs(dFh_analytic)

    ## Hilb ext reg
    α = 4*maximum(eΔ)
    A(u,v) = α^2*∫(∇(u) ⊙ ∇(v))dΩ + ∫(u ⋅ v)dΩ;
    Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv=Vector{PetscScalar}
    Hilb_assem=SparseMatrixAssembler(Tm,Tv,U_reg,V_reg)
    hilb_K = assemble_matrix(A,Hilb_assem,U_reg,V_reg)
    ## Autodiff result
    op = AffineFEOperator(U_reg,V_reg,hilb_K,-get_free_dof_values(dFh))
    dFh_Ω = solve(op)
    dF_Ω = get_free_dof_values(dFh_Ω)
    ## Analytic result
    op = AffineFEOperator(U_reg,V_reg,hilb_K,-get_free_dof_values(dFh_analytic))
    dFh_analytic_Ω = solve(op)
    dF_analytic_Ω = get_free_dof_values(dFh_analytic_Ω)

    hilb_abs_error = abs(dFh_Ω-dFh_analytic_Ω)
    hilb_rel_error = (abs(dFh_Ω-dFh_analytic_Ω))/abs(dFh_analytic_Ω)

    post_hilb_abs_error = maximum(abs,dF_Ω-dF_analytic_Ω)
    post_hilb_rel_error = maximum(abs,dF_Ω-dF_analytic_Ω)/maximum(abs,dF_analytic_Ω)

    path = dirname(dirname(@__DIR__))*"/results/InvHomLE_AutoDiffTesting_Parallel";
    writevtk(Ω,path,cellfields=["phi"=>φh,
        "H(phi)"=>(interp.H ∘ φh),
        "|nabla(phi)|"=>(norm ∘ ∇(φh)),
        "uh1"=>uh[1],
        "uh2"=>uh[2],
        "uh3"=>uh[3],
        "J′_abs_error"=>abs_error,
        "J′_rel_error"=>rel_error,
        "J′_analytic"=>dFh_analytic,
        "J′_autodiff"=>dFh,
        "hilb_abs_error"=>hilb_abs_error,
        "hilb_rel_error"=>hilb_rel_error,
        "v_J_Ω"=>dFh_analytic_Ω,
        "dJφh_Ω"=>dFh_Ω
    ])

    return pre_hilb_abs_error,pre_hilb_rel_error,post_hilb_abs_error,post_hilb_rel_error
end

out = with_debug() do distribute
    main((3,3),distribute)
end

####################
#  Serial Testing  #
####################
# function update_labels!(e::Int,model::CartesianDiscreteModel,f_Γ::F,::T,name::String) where {
#         M<:AbstractFloat,F<:Function,T<:NTuple{2,M}}
#     labels = get_face_labeling(model)
#     cell_to_entity = labels.d_to_dface_to_entity[end]
#     entity = maximum(cell_to_entity) + e
#     # Vertices
#     vtx_coords = model.grid_topology.vertex_coordinates
#     vtxs_Γ = findall(isone,f_Γ.(vtx_coords))
#     vtx_edge_connectivity = Array(Geometry.get_faces(model.grid_topology,0,1)[vtxs_Γ])
#     # Edges
#     edge_entries = [findall(x->any(x .∈  vtx_edge_connectivity[1:end.!=j]),
#         vtx_edge_connectivity[j]) for j = 1:length(vtx_edge_connectivity)]
#     edge_Γ = unique(reduce(vcat,getindex.(vtx_edge_connectivity,edge_entries),init=[]))
#     labels.d_to_dface_to_entity[1][vtxs_Γ] .= entity
#     labels.d_to_dface_to_entity[2][edge_Γ] .= entity
#     add_tag!(labels,name,[entity])
# end

# order = 1;
# el_size = (100,100);
# dom = (0.,1.,0.,1.);
# coord_max = dom[2],dom[4]
# model = CartesianDiscreteModel(dom,el_size,isperiodic=(true,true));
# ## Define Γ_N and Γ_D
# xmax,ymax = coord_max
# # f_Γ_D(x) = iszero(x) ? true : false;
# # update_labels!(1,model,f_Γ_D,coord_max,"origin")
# ## Triangulations and measures
# Ω = Triangulation(model)
# dΩ = Measure(Ω,2order)
# ## Spaces
# reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
# V = TestFESpace(model,reffe;conformity=:H1,constraint = :zeromean)
# U = TrialFESpace(V,[VectorValue(0.0,0.0)])

# all_V = MultiFieldFESpace([V,V,V])
# all_U = MultiFieldFESpace([U,U,U])

# # Space for shape sensitivities
# reffe_scalar = ReferenceFE(lagrangian,Float64,order)
# # FE space for LSF 
# V_φ = TestFESpace(model,reffe_scalar;conformity=:H1)
# # FE Space for shape derivatives
# V_reg = TestFESpace(model,reffe_scalar;conformity=:H1)
# U_reg = TrialFESpace(V_reg)
# ######################################################
# eΔ = (xmax,ymax)./el_size;
# interp = SmoothErsatzMaterialInterpolation(η = 2*maximum(eΔ))
# C = isotropic_2d(1.,0.3)
# g = VectorValue(0.,-1.0)
# φh = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25,V_φ)
# uh = zero(all_U)

# ## Weak form
# I = interp.I;
# DH = interp.DH

# εᴹ = (TensorValue(1.,0.,0.,0.),     # ϵᵢⱼ⁽¹¹⁾≡ϵᵢⱼ⁽¹⁾
#       TensorValue(0.,0.,0.,1.),     # ϵᵢⱼ⁽²²⁾≡ϵᵢⱼ⁽²⁾
#       TensorValue(0.,1/2,1/2,0.))   # ϵᵢⱼ⁽¹²⁾≡ϵᵢⱼ⁽³⁾

# a((u1,u2,u3),(v1,v2,v3),φ,dΩ) = ∫((I ∘ φ)*(C ⊙ ε(u1) ⊙ ε(v1)) + 
#                                   (I ∘ φ)*(C ⊙ ε(u2) ⊙ ε(v2)) + 
#                                   (I ∘ φ)*(C ⊙ ε(u3) ⊙ ε(v3)))dΩ
# l((v1,v2,v3),φ,dΩ) = ∫(-(I ∘ φ)*(C ⊙ εᴹ[1] ⊙ ε(v1)) - 
#                         (I ∘ φ)*(C ⊙ εᴹ[2] ⊙ ε(v2)) -
#                         (I ∘ φ)*(C ⊙ εᴹ[2] ⊙ ε(v3)))dΩ;

# a(φ,dΩ) = (u,v) -> a(u,v,φ,dΩ)
# l(φ,dΩ) = (v) -> l(v,φ,dΩ)
# res(u,v,φ,dΩ) = a(u,v,φ,dΩ) - l(v,φ,dΩ)

# _C(ε_p,ε_q) = C ⊙ ε_p ⊙ ε_q;
# _K((u1,u2,u3),εᴹ) = (_C(ε(u1)+εᴹ[1],εᴹ[1]) + _C(ε(u2)+εᴹ[2],εᴹ[2]) + 2*_C(ε(u1)+εᴹ[1],εᴹ[2]))/4
# _v_K((u1,u2,u3),εᴹ) = (_C(ε(u1)+εᴹ[1],ε(u1)+εᴹ[1]) + _C(ε(u2)+εᴹ[2],ε(u2)+εᴹ[2]) + 2*_C(ε(u1)+εᴹ[1],ε(u2)+εᴹ[2]))/4

# K_mod = (u,φ,dΩ) -> ∫((I ∘ φ)*_K(u,εᴹ))dΩ;
# DK_mod = (q,u,φ,dΩ) -> ∫(-_v_K(u,εᴹ)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;

# K_mod_func = Functional(K_mod,dΩ,uh,φh);
# K_mod_func_analytic = Functional(K_mod,dΩ,uh,φh;DF=DK_mod);
# K_mod_smap, C_smaps = AffineFEStateMap(K_mod_func,[K_mod_func_analytic],all_U,all_V,V_φ,a,l,res;ls = LUSolver());

# K_mod_smap(φh)
# K_mod_smap.F.F()
# C_smaps[1].F.F()

# dFh = compute_shape_derivative!(φh,K_mod_smap)
# dF = get_free_dof_values(dFh)

# dFh_analytic = compute_shape_derivative!(φh,C_smaps[1])
# dF_analytic = get_free_dof_values(dFh_analytic)

# norm(dF-dF_analytic,Inf)
# norm(dF-dF_analytic,Inf)/norm(dF_analytic,Inf)