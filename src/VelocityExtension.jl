# User supplies:
#   model
#   A(u,v,dΩ) (Usually ∫ β²∇u⋅∇v + uv dΩ)
#   order -> fe order
#
# Optional:
#   qorder -> quadrature order
#   dirichlet_tags
#   dirichlet_values 

struct VelocityExtension
    U_reg
    V_reg
    assem
    dΩ
    DH
    ns
    b
    x
    function VelocityExtension(model,interp,A,order::Int;qorder::Int=2*order,
            dirichlet_tags::Vector{String}=[],dirichlet_values=[],ls = PETScLinearSolver())
        if isempty(dvals) && length(dtags) > 0
            dvals = zeros(length(dtags))
        else
            @assert isequal(length(dtags),length(dvals))
        end
        ## FE Spaces
        reffe = ReferenceFE(lagrangian,PetscScalar,order)
        V_reg = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=dirichlet_tags)
        U_reg = TrialFESpace(V_reg,dirichlet_values)
        Ω = Triangulation(model)
        dΩ = Measure(Ω,qorder)
        ## Assembly
        Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
        Tv=Vector{PetscScalar}
        Hilb_assem=SparseMatrixAssembler(Tm,Tv,U_reg,V_reg)
        K = assemble_matrix((U,V) -> A(U,V,dΩ),Hilb_assem,U_reg,V_reg)
        ns = numerical_setup(symbolic_setup(ls,K),K)
        b = pfill(PetscScalar(0.0),partition(axes(K,1)))
        x = pfill(PetscScalar(0.0),partition(axes(K,2)))
        new(U_reg,V_reg,assem,dΩ,interp.DH,ns,b,x)
    end
end

function project!(vel_ext::VelocityExtension,vh,vh_L2,φh)
    ns=vel_ext.ns; b=vel_ext.b; dΩ=vel_ext.dΩ; 
    V_reg = vel_ext.V_reg; assem = vel_ext.assem;
    DH = vel_ext.DH
    ## Linear Form
    J′(v) = ∫(-vh_L2*v*(DH ∘ φh)*(norm ∘ ∇(φh)))dΩ;
    ## Assembly and solve
    assemble_vector!(b,J′,assem,V_reg)
    solve!(x,ns,b)
    copy!(get_free_dof_values(vh),x)
    consistent!(get_free_dof_values(vh)) |> fetch # From matrix layout to finite element layout
    return nothing
end

function project(vel_ext::VelocityExtension,vh_L2,φh)
    vh = zero(vel_ext.U_reg)
    project!(vel_ext,vh,vh_L2,φh)
    return vh
end

