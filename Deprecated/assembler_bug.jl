using Gridap, Gridap.FESpaces, Gridap.Helpers

## Current version @ Line 310 of Assemblers.jl

## Proposed version
function Gridap.FESpaces.assemble_matrix_and_vector!(a,l,K,b,assem_U,U,V,uhd)
    du = get_trial_fe_basis(U)
    dv = get_fe_basis(V);
    data = collect_cell_matrix_and_vector(U,V,a(du,dv),l(dv),uhd)
    assemble_matrix_and_vector!(K,b,assem_U,data)
end

function main(with_standard::Bool)
    model = CartesianDiscreteModel((0,1,0,1),(2,2));

    Ω = Triangulation(model)
    dΩ = Measure(Ω,2)

    # FE Problem
    V = FESpace(model,ReferenceFE(lagrangian,Float64,1),dirichlet_tags=["boundary"])
    U = TrialFESpace(V,2)
    a(u,v) = ∫( ∇(v)⋅∇(u) )*dΩ
    l(v) = ∫(v)*dΩ

    assem_U = SparseMatrixAssembler(U,V);
    op = AffineFEOperator(a,l,U,V,assem_U);

    K = get_matrix(op);
    b = get_vector(op);
    b_old = copy(b);

    if with_standard
        assemble_matrix_and_vector!(a,l,K,b,U,V)
    else
        assemble_matrix_and_vector!(a,l,K,b,assem_U,U,V,zero(U))
    end

    norm(b - b_old,Inf)
end

main(true)

main(false)