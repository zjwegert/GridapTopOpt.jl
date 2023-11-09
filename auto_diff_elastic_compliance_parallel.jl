using Gridap, Gridap.TensorValues, Gridap.Geometry, Gridap.FESpaces
using GridapDistributed
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code
using PartitionedArrays
using SparseMatricesCSR

using ChainRulesCore
using Zygote
include("ChainRules.jl");

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
    V_L2 = TestFESpace(model,reffe_scalar;conformity=:L2)
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

    ## Weak form
    I = interp.I;

    function ϕ_to_ϕₕ(ϕ::AbstractArray,Q)
        ϕ = FEFunction(Q,ϕ)
    end
    function ϕ_to_ϕₕ(ϕ,Q)
        ϕ
    end
    # function ϕ_to_ϕₕ(ϕ::CellField,Q)
    #     ϕ
    # end

    function a(u,v,φ) 
        φh = ϕ_to_ϕₕ(φ,V_φ)
        ∫((I ∘ φh)*(C ⊙ ε(u) ⊙ ε(v)))dΩ
    end
    a(φ) = (u,v) -> a(u,v,φ)

    l(v,φh) = ∫(v ⋅ g)dΓ_N
    l(φ) = v -> l(v,φ)

    res(u,v,φ,V_φ) = a(u,v,φ) - l(v,φ)

    φ = get_free_dof_values(φh)

    ## Solve finite element problem
    op = AffineFEOperator(a(φ),l(φ),U,V,assem)
    K = op.op.matrix;
    ## Solve
    uh = solve(op)
    ## Compute J and v_J
    _J(u,φ) = (a(uh,uh,φ)) # ∫(interp.ρ ∘ φ)dΩ

    φ_to_u = AffineFEStateMap(a,l,res,V_φ,U,V)
    u_to_j =  LossFunction(_J,V_φ,U)

    u, u_pullback   = rrule(φ_to_u,φ)
    j, j_pullback   = rrule(u_to_j,u,φ)
    _, du, dϕ₍ⱼ₎    = j_pullback(1) # dj = 1
    _, dϕ₍ᵤ₎        = u_pullback(du)
    dϕ           = dϕ₍ᵤ₎ + dϕ₍ⱼ₎

    function φ_to_j(φ)
        u = φ_to_u(φ)
        j = u_to_j(u,φ)
    end

    j,dφ = Zygote.withgradient(φ_to_j,φ)

    sum(_J(uh,φh))
    j

    ## Shape derivative
    # Autodiff
    dϕh  = interpolate_everywhere(FEFunction(V_φ,dϕ),U_reg)

    # Analytic
    J′(v,v_h) = ∫(-v_h*v*(interp.DH ∘ φh)*(norm ∘ ∇(φh)))dΩ;
    v_J = -(C ⊙ ε(uh) ⊙ ε(uh))
    b = assemble_vector(v->J′(v,-v_J),V_reg)
    analytic_J′ = FEFunction(V_reg,b)

    abs_error = abs(dϕh-analytic_J′)
    rel_error = (abs(dϕh-analytic_J′))/abs(analytic_J′)

    #############################
    ## Hilb ext reg
    α = 4*maximum(eΔ)
    A(u,v) = α^2*∫(∇(u) ⊙ ∇(v))dΩ + ∫(u ⋅ v)dΩ;
    Hilb_assem=SparseMatrixAssembler(Tm,Tv,U_reg,V_reg)
    hilb_K = assemble_matrix(A,Hilb_assem,U_reg,V_reg)

    ## Autodiff result
    # -dϕh is AD version of J′ that we plug in usually!
    op = AffineFEOperator(U_reg,V_reg,hilb_K,-dϕh.free_values)
    dϕh_Ω = solve(op)

    ## Analytic result
    b = assemble_vector(v->J′(v,v_J),V_reg)
    op = AffineFEOperator(U_reg,V_reg,hilb_K,b)
    v_J_Ω = solve(op)

    hilb_abs_error = abs(dϕh_Ω-v_J_Ω)
    hilb_rel_error = (abs(dϕh_Ω-v_J_Ω))/abs(v_J_Ω)

    path = "./Results/AutoDiffTesting_Parallel";
    writevtk(Ω,path,cellfields=["phi"=>φh,
        "H(phi)"=>(interp.H ∘ φh),
        "|nabla(phi))|"=>(norm ∘ ∇(φh)),
        "uh"=>uh,
        "J′_abs_error"=>abs_error,
        "J′_rel_error"=>rel_error,
        "J′_analytic"=>analytic_J′,
        "J′_autodiff"=>dϕh,
        "hilb_abs_error"=>hilb_abs_error,
        "hilb_rel_error"=>hilb_rel_error,
        "v_J_Ω"=>v_J_Ω,
        "dJϕh_Ω"=>dϕh_Ω
    ])
end

t = with_debug() do distribute
    main((3,3),distribute)
end;
display(t)

########################################################################################

# struct DistributedFunctional
#     f
#     dΩ
# end

# function Gridap.Fields.gradient(f::DistributedFunctional,uh::GridapDistributed.DistributedCellField)
#     contribs = map(local_views(uh),local_views(f.dΩ)) do uh, dΩ
#         _f = uh -> f.f(uh,dΩ)
#         return Gridap.Fields.gradient(_f,uh)
#     end
#     return GridapDistributed.DistributedDomainContribution(contribs)
# end

struct Functional{N}
    F
    dΩ
    state::NTuple{N}
    function Functional(F,dΩ,args...)
        N = length(args)
        new{N}(F,dΩ,args)
    end
end

struct FunctionalGradient{N,K}
    F::Functional{N}
    function FunctionalGradient(F::Functional{N},K) where N
        @assert 0<K<=N
        new{N,K}(F)
    end
end

function (fg::FunctionalGradient{N,K})(uh::GridapDistributed.DistributedCellField) where {N,K}
    fields = map(i->i==K ? uh : fg.F.state[K],1:N)
    local_fields = map(local_views,fields)
    contribs = map(local_views(fg.F.dΩ),local_fields...) do dΩ,lf...
        _f = u -> fg.F.F(lf[1:K-1]...,u,lf[K+1:end]...,dΩ)
        @show _f 
        return Gridap.Fields.gradient(_f,lf[K])
    end
    return GridapDistributed.DistributedDomainContribution(contribs)
end

function (fg::FunctionalGradient{N,K})(uh::FEFunction) where {N,K}
    fields = map(i->i==K ? uh : fg.F.state[K],1:N)
    _f = u -> fg.F.F(fields[1:K-1]...,u,fields[K+1:end]...,dΩ)
    return Gridap.Fields.gradient(_f,fields[K])
end


# struct DistributedLossFunction{P,U}
#     loss::DistributedFunctional
#     param_sp::P
#     state_sp::U
#     # assem::Assembler
#   end

# function (u_to_j::DistributedLossFunction)(u,ϕ)
#     loss=u_to_j.loss
#     U=u_to_j.state_sp
#     Q=u_to_j.param_sp
#     uₕ=FEFunction(U,u)
#     ϕₕ=FEFunction(Q,ϕ)
#     sum(loss.f(uₕ,ϕₕ,loss.dΩ))
# end


ranks = with_debug() do distribute
    distribute(LinearIndices((1,)))
end

model = CartesianDiscreteModel(ranks,(1,1),(0,1,0,1),(2,2))

#model = CartesianDiscreteModel((0,1,0,1),(2,2))
reffe = ReferenceFE(lagrangian,Float64,1)
V = FESpace(model,reffe)

Ω = Triangulation(model)
dΩ = Measure(Ω,2)

uh = zero(V)
ϕh = zero(V)

J = (u,ϕh,dΩ) -> ∫(u⋅u+ϕh)dΩ
J_func = Functional(J,dΩ,uh,ϕh)

dJdu = FunctionalGradient(J_func,1)(uh)

nothing

# J = ((u,v) -> ∫(u⋅u)dΩ)
# loss = LossFunction(J,V,V)

# u = get_free_dof_values(uh)
# loss(u,u)

# J2 = (u -> ∫(u⋅u)dΩ)
# djdu = ∇(uh->J2(uh))
# contr = djdu(uh)

# J3 = (u,dΩ) -> ∫(u⋅u)dΩ
# df = DistributedFunctional(J3,dΩ)

# djdu = ∇(df,uh)
# assemble_vector(djdu,V)

# ## Loss function
# _J4 = (u,φ) -> ∫(u⋅u)dΩ
# _loss = LossFunction(_J4,V,V)
# _loss(get_free_dof_values(uh),get_free_dof_values(uh))

# ## Distributed loss function
# J4 = (u,φ,dΩ) -> ∫(u⋅u+φ)dΩ
# df = DistributedFunctional(J4,dΩ)
# loss = DistributedLossFunction(df,V,V)
# loss(get_free_dof_values(uh),get_free_dof_values(ϕh))

# ## u_to_j_pullback
# jp, u_to_j_pullback = ChainRulesCore.rrule(loss,get_free_dof_values(uh),get_free_dof_values(uh))

# # This breaks:
# NEW_df = DistributedFunctional((uh,dΩ)->loss.loss.f(uh,ϕh,dΩ),loss.loss.dΩ)
# NEW_df()



# djdu = ∇(DistributedFunctional((uh,dΩ)->loss.loss.f(uh,ϕh,dΩ),loss.loss.dΩ),uh)

# djdϕ = ∇(DistributedFunctional((ϕh,dΩ)->loss.loss.f(uh,ϕh,dΩ),loss.loss.dΩ),ϕh)

# u_to_j_pullback(1)


# function ChainRulesCore.rrule(u_to_j::DistributedLossFunction,u,ϕ)
#     loss=u_to_j.loss
#     U=u_to_j.state_sp
#     Q=u_to_j.param_sp
#     uh=FEFunction(U,u)
#     ϕh=FEFunction(Q,ϕ)
#     jp=u_to_j(u,ϕ) # === jp=sum(loss.f(uₕ,ϕₕ,loss.dΩ))
#     function u_to_j_pullback(dj)
#         # djdu = ∇(uₕ->loss.f(uₕ,ϕₕ,loss.dΩ),uₕ)
#         djdu = ∇(DistributedFunctional((uh,dΩ)->loss.f(uh,ϕh,dΩ),loss.dΩ),uh)
#         djdu_vec = assemble_vector(djdu,U)
#         # djdϕ = ∇(ϕₕ->loss.f(uh,ϕh,loss.dΩ),ϕh)
#         djdϕ =  ∇(DistributedFunctional((ϕh,dΩ)->loss.f(uh,ϕh,dΩ),loss.dΩ),ϕh)
#         djdϕ_vec = assemble_vector(djdϕ,Q)
#         (  NoTangent(), dj*djdu_vec, dj*djdϕ_vec )
#     end
#     jp, u_to_j_pullback
# end