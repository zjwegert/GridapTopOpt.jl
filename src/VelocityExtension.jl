# User supplies:
#   model
#   A(u,v,dΩ) (Usually ∫ β²∇u⋅∇v + uv dΩ)
#   order -> fe order
#
# Optional:
#   qorder -> quadrature order
#   dirichlet_tags
#   dirichlet_values 

struct VelocityExtension{A,B,C,D,E,F}
    model   :: A
    spaces  :: B
    interp  :: C
    assem   :: D
    measure :: E
    caches  :: F
end

function VelocityExtension(model,
                           interp,
                           biform,
                           order::Int;
                           qorder::Int=2*order,
                           dirichlet_tags::Vector{String}=[],
                           dirichlet_values=zeros(PetscScalar,length(dirichlet_tags)),
                           ls = PETScLinearSolver())
    @assert isequal(length(dtags),length(dvals))

    ## FE Spaces
    reffe = ReferenceFE(lagrangian,PetscScalar,order)
    V_reg = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=dirichlet_tags)
    U_reg = TrialFESpace(V_reg,dirichlet_values)
    Ω  = Triangulation(model)
    dΩ = Measure(Ω,qorder)
    
    ## Assembly
    Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv = Vector{PetscScalar}
    assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg)
    K  = assemble_matrix((U,V) -> biform(U,V,dΩ),Hilb_assem,U_reg,V_reg)
    ns = numerical_setup(symbolic_setup(ls,K),K)
    b  = pfill(PetscScalar(0.0),partition(axes(K,1)))
    x  = pfill(PetscScalar(0.0),partition(axes(K,2)))

    spaces = (U_reg,V_reg); caches = (ns,b,x)
    return VelocityExtension(model,spaces,interp,assem,dΩ,caches)
end

function project!(vel_ext::VelocityExtension,vh,vh_L2,φh)
    U_reg, V_reg = vel_ext.spaces
    ns, b, x = vel_ext.caches
    dΩ = vel_ext.measure; assem = vel_ext.assem; DH = vel_ext.interp.DH
    dJ(v) = ∫(-vh_L2⋅v⋅(DH ∘ φh)⋅(norm ∘ ∇(φh)))*dΩ;
    assemble_vector!(b,dJ,assem,V_reg)
    solve!(x,ns,b)
    copy!(get_free_dof_values(vh),x)
    consistent!(get_free_dof_values(vh)) |> fetch # From matrix layout to finite element layout
    return nothing
end

function project(vel_ext::VelocityExtension,vh_L2,φh)
    U_reg, V_reg = vel_ext.spaces
    vh = zero(U_reg)
    project!(vel_ext,vh,vh_L2,φh)
    return vh
end
