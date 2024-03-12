using Gridap, Gridap.TensorValues, Gridap.Geometry, Gridap.ReferenceFEs
using GridapDistributed
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code
using PartitionedArrays
using SparseMatricesCSR
using MPI

import GridapDistributed.DistributedCellField
import GridapDistributed.DistributedFESpace
import GridapDistributed.DistributedDiscreteModel
import GridapDistributed.DistributedMeasure
import GridapDistributed.DistributedTriangulation

using LevelSetTopOpt

function isotropic_3d(E::M,nu::M) where M<:AbstractFloat
    λ = E*nu/((1+nu)*(1-2nu)); μ = E/(2*(1+nu))
    C =[λ+2μ   λ      λ      0      0      0
        λ     λ+2μ    λ      0      0      0
        λ      λ     λ+2μ    0      0      0
        0      0      0      μ      0      0
        0      0      0      0      μ      0
        0      0      0      0      0      μ];
    return SymFourthOrderTensorValue(
        C[1,1], C[6,1], C[5,1], C[2,1], C[4,1], C[3,1],
        C[1,6], C[6,6], C[5,6], C[2,6], C[4,6], C[3,6],
        C[1,5], C[6,5], C[5,5], C[2,5], C[4,5], C[3,5],
        C[1,2], C[6,2], C[5,2], C[2,2], C[4,2], C[3,2],
        C[1,4], C[6,4], C[5,4], C[2,4], C[4,4], C[3,4],
        C[1,3], C[6,3], C[5,3], C[2,3], C[4,3], C[3,3])
end

function isotropic_2d(E::M,nu::M) where M<:AbstractFloat
  λ = E*nu/((1+nu)*(1-nu)); μ = E/(2*(1+nu))
  C = [λ+2μ  λ     0
       λ    λ+2μ   0
       0     0     μ];
  SymFourthOrderTensorValue(C[1,1],C[3,1],C[2,1],C[1,3],
      C[3,3],C[2,3],C[1,2],C[3,2],C[2,2])
end

############################################################################################

function main(distribute,np,n,order)
  D = length(np)
  ranks = distribute(LinearIndices((prod(np),)))

  n_tags = (D==2) ? "tag_6" : "tag_22"
  d_tags = (D==2) ? ["tag_5"] : ["tag_21"]

  nc = (D==2) ? (n,n) : (n,n,n)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  model  = CartesianDiscreteModel(ranks,np,domain,nc)
  Ω = Triangulation(model)
  Γ = Boundary(model,tags=n_tags)

  poly  = (D==2) ? QUAD : HEX
  reffe = LagrangianRefFE(VectorValue{D,Float64},poly,order)
  V = TestFESpace(model,reffe;dirichlet_tags="boundary")
  U = TrialFESpace(V)
  assem = SparseMatrixAssembler(SparseMatrixCSR{0,PetscScalar,PetscInt},Vector{PetscScalar},U,V,FullyAssembledRows())

  dΩ = Measure(Ω,2*order)
  dΓ = Measure(Γ,2*order)
  C = (D == 2) ? isotropic_2d(1.,0.3) : isotropic_3d(1.,0.3)
  g = (D == 2) ? VectorValue(0.0,1.0) : VectorValue(1.0,1.0,1.0)
  a(u,v) = ∫((C ⊙ ε(u) ⊙ ε(v)))dΩ
  l(v)   = ∫(v ⋅ g)dΓ

  op   = AffineFEOperator(a,l,U,V,assem)
  A, b = get_matrix(op), get_vector(op);

  options = "
    -ksp_type gmres -ksp_rtol 1.0e-12 -ksp_max_it 200
    -pc_type gamg
    -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg
    -ksp_converged_reason -ksp_error_if_not_converged true -ksp_monitor_short
    -mat_use
    "
  GridapPETSc.with(args=split(options)) do
    solver = ElasticitySolver(V;rtol=1.e-12,maxits=500)
    ss = symbolic_setup(solver,A)
    ns = numerical_setup(ss,A)

    x  = pfill(PetscScalar(0.0),partition(axes(A,2)))
    solve!(x,ns,b)
  end
end

with_mpi() do distribute
  np = (2,2)
  n  = 100
  order = 1
  main(distribute,np,n,order)
end