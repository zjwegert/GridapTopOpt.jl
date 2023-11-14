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
    @assert isequal(length(dirichlet_tags),length(dirichlet_values))

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
    x  = pfill(PetscScalar(0.0),partition(axes(K,2)))

    spaces = (U_reg,V_reg); caches = (ns,x)
    return VelocityExtension(model,spaces,interp,assem,dΩ,caches)
end

function project!(vel_ext::VelocityExtension,dJh::GridapDistributed.DistributedCellField)
    ns, x = vel_ext.caches
    solve!(x,ns,-get_free_dof_values(dJh))
    copy!(get_free_dof_values(dJh),x)
    consistent!(get_free_dof_values(dJh)) |> fetch # From matrix layout to finite element layout
    return dJh
end

function project!(vel_ext::VelocityExtension,dJh::FEFunction)
    ns, x = vel_ext.caches
    solve!(x,ns,-get_free_dof_values(dJh))
    copy!(get_free_dof_values(dJh),x)
    return dJh
end

project!(vel_ext::VelocityExtension,dJh_vec::Vector) = map(dJh -> project!(vel_ext,dJh),dJh_vec)

# function project(vel_ext::VelocityExtension,dJh)
#     U_reg, _ = vel_ext.spaces
#     vh = zero(U_reg)
#     project!(vel_ext,dJh)
#     return vh
# end
