using Gridap, Gridap.TensorValues, Gridap.Geometry, Gridap.FESpaces, Gridap.Helpers
using GridapDistributed
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code
using PartitionedArrays
using SparseMatricesCSR
using ChainRulesCore
using LSTO_Distributed

using GridapSolvers
using Gridap.MultiField

function PZT5A(D::Int)
  ε_0 = 8.854e-12;
  if D == 3
    C_Voigt = [12.0400e10  7.52000e10  7.51000e10  0.0        0.0        0.0
          7.52000e10  12.0400e10  7.51000e10  0.0        0.0        0.0
          7.51000e10  7.51000e10  11.0900e10  0.0        0.0        0.0
          0.0         0.0         0.0          2.1000e10  0.0        0.0
          0.0         0.0         0.0          0.0        2.1000e10  0.0
          0.0         0.0         0.0          0.0        0.0       2.30e10]
    e_Voigt = [0.0       0.0       0.0        0.0       12.30000  0.0
                0.0       0.0       0.0        12.30000  0.0       0.0
                -5.40000  -5.40000   15.80000   0.0       0.0       0.0]
    K_Voigt = [540*ε_0     0        0
                  0      540*ε_0    0
                  0        0    830*ε_0]
  elseif D == 2
    C_Voigt = [12.0400e10  7.51000e10     0.0
                7.51000e10  11.0900e10     0.0
                  0.0         0.0      2.1000e10]
    e_Voigt = [0.0       0.0       12.30000
                -5.40000   15.80000   0.0]
    K_Voigt = [540*ε_0        0
                  0    830*ε_0]
  else
    @notimplemented
  end
  C = voigt2tensor4(C_Voigt)
  e = voigt2tensor3(e_Voigt)
  κ = voigt2tensor2(K_Voigt)
  C,e,κ
end

"""
  Given a material constant given in Voigt notation,
  return a SymFourthOrderTensorValue using ordering from Gridap
"""
function voigt2tensor4(A::Array{M,2}) where M
  if isequal(size(A),(3,3))
    return SymFourthOrderTensorValue(A[1,1], A[3,1], A[2,1],
                                     A[1,3], A[3,3], A[2,3],
                                     A[1,2], A[3,2], A[2,2])
  elseif isequal(size(A),(6,6))
    return SymFourthOrderTensorValue(A[1,1], A[6,1], A[5,1], A[2,1], A[4,1], A[3,1],
                                     A[1,6], A[6,6], A[5,6], A[2,6], A[4,6], A[3,6],
                                     A[1,5], A[6,5], A[5,5], A[2,5], A[4,5], A[3,5],
                                     A[1,2], A[6,2], A[5,2], A[2,2], A[4,2], A[3,2],
                                     A[1,4], A[6,4], A[5,4], A[2,4], A[4,4], A[3,4],
                                     A[1,3], A[6,3], A[5,3], A[2,3], A[4,3], A[3,3])
  else
      @notimplemented
  end
end

"""
  Given a material constant given in Voigt notation,
  return a ThirdOrderTensorValue using ordering from Gridap
"""
function voigt2tensor3(A::Array{M,2}) where M
  if isequal(size(A),(2,3))
    return ThirdOrderTensorValue(A[1,1], A[2,1], A[1,3], A[2,3], A[1,3], A[2,3], A[1,2], A[2,2])
  elseif isequal(size(A),(3,6))
    return ThirdOrderTensorValue(
      A[1,1], A[2,1], A[3,1], A[1,6], A[2,6], A[3,6], A[1,5], A[2,5], A[3,5],
      A[1,6], A[2,6], A[3,6], A[1,2], A[2,2], A[3,2], A[1,4], A[2,4], A[3,4],
      A[1,5], A[2,5], A[3,5], A[1,4], A[2,4], A[3,4], A[1,3], A[2,3], A[3,3])
  else
    @notimplemented
  end
end

"""
  Given a material constant given in Voigt notation,
  return a SymTensorValue using ordering from Gridap
"""
function voigt2tensor2(A::Array{M,2}) where M
  return TensorValue(A)
end

function main(mesh_partition,distribute)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  order = 1;
  el_size = (30,30,30);
  dom = (0.,1.,0.,1.,0.,1.);
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size,isperiodic=(true,true,true));
  ## Define Γ_N and Γ_D
  f_Γ_D(x) = iszero(x)
  update_labels!(1,model,f_Γ_D,"origin")
  ## Triangulations and measures
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2order)
  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["origin"])
  U = TrialFESpace(V,VectorValue(0.0,0.0,0.0))
  Q = TestFESpace(model,reffe_scalar;conformity=:H1,dirichlet_tags=["origin"])
  P = TrialFESpace(Q,0)

  mfs = BlockMultiFieldStyle()
  UP = MultiFieldFESpace([U,P];style=mfs)
  VQ = MultiFieldFESpace([V,Q];style=mfs)

  C, e, κ = PZT5A(3);
  k0 = norm(C.data,Inf); 
  α0 = norm(e.data,Inf); 
  β0 = norm(κ.data,Inf);
  γ = β0*k0/α0^2;

  ϵ⁰ = TensorValue(1.,0.,0.,0.,0.,0.,0.,0.,0.);

  a((u, ϕ),(v, q)) = ∫(1/k0*C ⊙ ε(u) ⊙ ε(v) - 
                      1/α0*((-∇(ϕ) ⋅ e) ⊙ ε(v)) +
                      -1/α0*((e ⋅² ε(u)) ⋅ -∇(q)) + 
                      -γ/β0*((κ ⋅ -∇(ϕ)) ⋅ -∇(q)) )dΩ;

  l((v, q)) = ∫(-C ⊙ ϵ⁰ ⊙ ε(v) + k0/α0*((e ⋅² ϵ⁰) ⋅ -∇(q)))dΩ;

  assem = SparseMatrixAssembler(SparseMatrixCSR{0,PetscScalar,PetscInt},Vector{PetscScalar},UP,VQ)
  op = AffineFEOperator(a,l,UP,VQ,assem)
  ## Solve
  #options = "-pc_type bjacobi -ksp_type gmres -ksp_error_if_not_converged true
  #  -ksp_converged_reason -ksp_rtol 1.0e-10 -ksp_monitor_short"
  options = "-ksp_converged_reason -pc_type gamg -ksp_type cg "
  GridapPETSc.with(args=split(options)) do
    # ls = PETScLinearSolver()
    # xh = solve(ls,op)
    solver_u = ElasticitySolver(Ω,V)
    solver_ϕ = PETScLinearSolver()
    P = BlockDiagonalPreconditioner([solver_u,solver_ϕ])
    solver = GridapSolvers.LinearSolvers.GMRESSolver(100;Pr=P,rtol=1.e-8,verbose=i_am_main(ranks))
    A  = get_matrix(op)
    b  = get_vector(op)
    ns = numerical_setup(symbolic_setup(solver,A),A)
    x  = allocate_col_vector(A)
    solve!(x,ns,b)
  end
end;

with_mpi() do distribute
  main((2,2,1),distribute)
end;
