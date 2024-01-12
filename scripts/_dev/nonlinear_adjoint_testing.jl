
using LinearAlgebra
using Gridap, Gridap.FESpaces, Gridap.CellData

function transpose_contributions(b::DomainContribution)
  c = DomainContribution()
  for (trian,array_old) in b.dict
    array_new = lazy_map(transpose,array_old)
    add_contribution!(c,trian,array_new)
  end
  return c
end

function assemble_adjoint_matrix!(f::Function,A::AbstractMatrix,a::Assembler,U::FESpace,V::FESpace)
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  contr = transpose_contributions(f(u,v))
  assemble_matrix!(A,a,collect_cell_matrix(V,U,contr))
end

model = CartesianDiscreteModel((0,1,0,1),(4,4))

reffe = ReferenceFE(lagrangian,Float64,1)
V = FESpace(model,reffe)
U = TrialFESpace(V)

Ω  = Triangulation(model)
dΩ = Measure(Ω,4)

e = VectorValue(1.0,-1.0)
a(u,v) = ∫((∇(u)⋅e)⋅v)*dΩ

A  = assemble_matrix(a,U,V)
At = assemble_matrix((u,v) -> a(v,u), V, U)

A ≈ transpose(At) # true

assem = SparseMatrixAssembler(V,U)
Ad = copy(At); LinearAlgebra.fillstored!(Ad,0.0)
Ad = assemble_adjoint_matrix!(a,Ad,assem,U,V)
Ad ≈ transpose(A) # true
