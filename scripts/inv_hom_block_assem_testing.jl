using Gridap, Gridap.MultiField, GridapDistributed, GridapPETSc, GridapSolvers, 
  PartitionedArrays, LSTO_Distributed, SparseMatricesCSR

nothing
  
np = (2,3);

ranks = with_debug() do distribute
    distribute(LinearIndices((prod(np),)))
end;

## Parameters
order = 1;
xmax,ymax=(1.0,1.0)
dom = (0,xmax,0,ymax);
el_size = (10,10);
γ = 0.05;
γ_reinit = 0.5;
max_steps = floor(Int,minimum(el_size)/10)
tol = 1/(order^2*10)*prod(inv,minimum(el_size))
C = isotropic_2d(1.,0.3);
η_coeff = 2;
α_coeff = 4;
path = dirname(dirname(@__DIR__))*"/results/block_testing"

## FE Setup
model = CartesianDiscreteModel(ranks,(2,3),dom,el_size,isperiodic=(true,true));
# model = CartesianDiscreteModel(dom,el_size,isperiodic=(true,true));
Δ = get_Δ(model)
f_Γ_D(x) = iszero(x)
update_labels!(1,model,f_Γ_D,"origin")

## Triangulations and measures
Ω = Triangulation(model)
dΩ = Measure(Ω,2order)
vol_D = sum(∫(1)dΩ)

## Spaces
reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
_V = TestFESpace(model,reffe;dirichlet_tags=["origin"])
_U = TrialFESpace(_V,VectorValue(0.0,0.0))
mfs = BlockMultiFieldStyle()
U = MultiFieldFESpace([_U,_U,_U];style=mfs);
V = MultiFieldFESpace([_V,_V,_V];style=mfs);
V_reg = V_φ = TestFESpace(model,reffe_scalar)
U_reg = TrialFESpace(V_reg)

## Create FE functions
lsf_fn = x->max(gen_lsf(2,0.4)(x),gen_lsf(2,0.4;b=VectorValue(0,0.5))(x));
φh = interpolate(lsf_fn,V_φ);
φ = get_free_dof_values(φh)

## Interpolation and weak form
interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ))
I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

εᴹ = (TensorValue(1.,0.,0.,0.),     # ϵᵢⱼ⁽¹¹⁾≡ϵᵢⱼ⁽¹⁾
    TensorValue(0.,0.,0.,1.),     # ϵᵢⱼ⁽²²⁾≡ϵᵢⱼ⁽²⁾
    TensorValue(0.,1/2,1/2,0.))   # ϵᵢⱼ⁽¹²⁾≡ϵᵢⱼ⁽³⁾

a(u,v,φ,dΩ) = ∫((I ∘ φ)*sum(C ⊙ ε(u[i]) ⊙ ε(v[i]) for i = 1:length(u)))dΩ
l(v,φ,dΩ) = ∫(-(I ∘ φ)*sum(C ⊙ εᴹ[i] ⊙ ε(v[i]) for i ∈ 1:length(v)))dΩ;
res(u,v,φ,dΩ) = a(u,v,φ,dΩ) - l(v,φ,dΩ)

## Optimisation functionals
_C(C,ε_p,ε_q) = C ⊙ ε_p ⊙ ε_q;
_K(C,(u1,u2,u3),εᴹ) = (_C(C,ε(u1)+εᴹ[1],εᴹ[1]) + _C(C,ε(u2)+εᴹ[2],εᴹ[2]) + 2*_C(C,ε(u1)+εᴹ[1],εᴹ[2]))/4
_v_K(C,(u1,u2,u3),εᴹ) = (_C(C,ε(u1)+εᴹ[1],ε(u1)+εᴹ[1]) + _C(C,ε(u2)+εᴹ[2],ε(u2)+εᴹ[2]) + 2*_C(C,ε(u1)+εᴹ[1],ε(u2)+εᴹ[2]))/4    

J = (u,φ,dΩ) -> ∫(-(I ∘ φ)*_K(C,u,εᴹ))dΩ
dJ = (q,u,φ,dΩ) -> ∫(-_v_K(C,u,εᴹ)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;
Vol = (u,φ,dΩ) -> ∫(((ρ ∘ φ) - 0.5)/vol_D)dΩ;
dVol = (q,u,φ,dΩ) -> ∫(1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

## Finite difference solver and level set function
stencil = AdvectionStencil(FirstOrderStencil(2,Float64),model,V_φ,Δ./order,max_steps,tol);
reinit!(stencil,φ,γ_reinit)

## Special assembly option
using BlockArrays
using Gridap, Gridap.TensorValues, Gridap.Geometry, Gridap.FESpaces, 
  Gridap.Helpers, Gridap.ReferenceFEs, Gridap.Algebra,  Gridap.CellData, Gridap.FESpaces

import GridapDistributed: DistributedCellField, DistributedMultiFieldFEBasis
import Gridap.FESpaces: AffineFEOperator, assemble_matrix_and_vector, assemble_matrix!, assemble_matrix

Base.length(::DistributedCellField) = 1;
Base.length(a::DistributedMultiFieldFEBasis) = length(a.field_fe_basis);
Base.length(a::MultiFieldCellField) = length(a.single_fields);
Base.getindex(a::MultiFieldCellField,i::UnitRange) = a.single_fields[i]
Base.getindex(a::DistributedMultiFieldFEBasis,i::UnitRange) = a.field_fe_basis[i]

## Assumptions
# 1. Blocks down the diagonal are exactly the same
# 2. If a diagonal block is made of several blocks 
#     these must correspond to `diag_block_axes` and
#     be ordered in the manor they appear. E.g., ...
# 3. The block ordering must not change via `BlockMultiFieldStyle`
Base.@kwdef struct DiagonalBlockMatrixAssembler{A<:Assembler} <: SparseMatrixAssembler 
  assem::A
  diag_block_axes::UnitRange{Int64} = 1:1 # <- adjust so that user can't pass 'silly' unit ranges (e.g., starting from something other than zero) 
end

function  AffineFEOperator(
  a::Function,l::Function,trial::FESpace,test::FESpace,assem::DiagonalBlockMatrixAssembler)
  @assert ! isa(test,TrialFESpace) """\n
  It is not allowed to build an AffineFEOperator with a test space of type TrialFESpace.

  Make sure that you are writing first the trial space and then the test space when
  building an AffineFEOperator or a FEOperator.
  """
  A,b = assemble_matrix_and_vector(a,l,assem,trial,test)

  AffineFEOperator(trial,test,A,b)
end

function assemble_matrix_and_vector(a::Function,l::Function,assem::DiagonalBlockMatrixAssembler,U::FESpace,V::FESpace)
  diag_block_axes = assem.diag_block_axes
  _assem = assem.assem

  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  uhd = zero(U)
  matcontribs, veccontribs = a(u[diag_block_axes],v[diag_block_axes]),l(v)
  data = collect_cell_matrix_and_vector(U,V,matcontribs,veccontribs,uhd);
  A,b = assemble_matrix_and_vector(_assem,data)
  _identical_diag_block_assemble!(A,diag_block_axes)

  return A,b
end

function _assemble_matrix_and_vector!(a::Function,l::Function,A::AbstractMatrix,b::AbstractVector,assem::Assembler,U::FESpace,V::FESpace,uhd)
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  assemble_matrix_and_vector!(A,b,assem,collect_cell_matrix_and_vector(U,V,a(u,v),l(v),uhd))
end

function _assemble_matrix_and_vector!(a::Function,l::Function,A::AbstractMatrix,b::AbstractVector,assem::DiagonalBlockMatrixAssembler,U::FESpace,V::FESpace,uhd)
  diag_block_axes = assem.diag_block_axes
  _assem = assem.assem
  
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  matcontribs, veccontribs = a(u[diag_block_axes],v[diag_block_axes]),l(v)
  data = collect_cell_matrix_and_vector(U,V,matcontribs,veccontribs,uhd)
  assemble_matrix_and_vector!(A,b,_assem,data)
  _identical_diag_block_assemble!(A,diag_block_axes)
end

function assemble_matrix!(a::Function,A::AbstractMatrix,assem::DiagonalBlockMatrixAssembler,U::FESpace,V::FESpace)
  diag_block_axes = assem.diag_block_axes
  _assem = assem.assem
  
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  assemble_matrix!(A,_assem,collect_cell_matrix(U,V,a(u[diag_block_axes],v[diag_block_axes])))
  _identical_diag_block_assemble!(A,diag_block_axes)
end

function assemble_matrix(a::Function,assem::DiagonalBlockMatrixAssembler,U::FESpace,V::FESpace)
  diag_block_axes = assem.diag_block_axes
  _assem = assem.assem
  
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  A = assemble_matrix(_assem,collect_cell_matrix(U,V,a(u[diag_block_axes],v[diag_block_axes])))
  _identical_diag_block_assemble!(A,diag_block_axes)
  return A
end

function _identical_diag_block_assemble!(A::AbstractMatrix,diag_block_axes::UnitRange)
  @check typeof(A) <: BlockArrays.AbstractBlockArray "`DiagonalBlockMatrixAssembler` expects a block structure, recieved $(typeof(A))"
  blocks_size = size(A.blocks,1);
  block_iter = blocks_size % last(diag_block_axes)
  @check iszero(block_iter) "Inconsistant number of blocks to match `diag_block_axes`: 
      Expected to fit multiples of $diag_block_axes blocks into $(blocks_size)x$(blocks_size) block matrix."
  
  for i ∈ Iterators.partition(last(diag_block_axes)+1:blocks_size,last(diag_block_axes))
    A.blocks[i,i] = A.blocks[diag_block_axes,diag_block_axes]
  end
  return nothing
end

function BlockArrays.mortar(blocks::Matrix{<:PSparseMatrix})
  rows = map(b->axes(b,1),blocks[:,1])
  cols = map(b->axes(b,2),blocks[1,:])
  @show rows, cols

  function check_axes(a,r,c)
    A = PartitionedArrays.matching_local_indices(axes(a,1),r)
    B = PartitionedArrays.matching_local_indices(axes(a,2),c)
    return A & B
  end
  @show map(I -> check_axes(blocks[I],rows[I[1]],cols[I[2]]),CartesianIndices(size(blocks)))
  @check all(map(I -> check_axes(blocks[I],rows[I[1]],cols[I[2]]),CartesianIndices(size(blocks))))
  # Jordi the above doesn't like when the blocks are set to be equal at later stage.

  return GridapDistributed.BlockPMatrix(blocks,rows,cols)
end

K11 = K.blocks[1,1]
K22 = K.blocks[2,2]

K11_full = K_test.blocks[1,1]
K22_full = K_test.blocks[2,2]

_assemble_matrix_and_vector!((u,v) -> a(u,v,φh,dΩ),v -> l(v,φh,dΩ),K,b,assem,U,V,uhd)

BlockArrays.mortar(K.blocks)


#### New way
## Initialise op
uhd = zero(U);
assem = DiagonalBlockMatrixAssembler(assem=SparseMatrixAssembler(U,V));
@time op = AffineFEOperator((u,v) -> a(u,v,φh,dΩ),v -> l(v,φh,dΩ),U,V,assem);
K = get_matrix(op); b = get_vector(op); 
## Initialise adjoint
assem_adjoint = DiagonalBlockMatrixAssembler(assem=SparseMatrixAssembler(V,U));
adjoint_K = assemble_matrix((u,v) -> a(v,u,φh,dΩ),assem_adjoint,V,U);

## Update mat and vec
_assemble_matrix_and_vector!((u,v) -> a(u,v,φh,dΩ),v -> l(v,φh,dΩ),K,b,assem,U,V,uhd)
# numerical_setup!(...)

## Update adjoint
assemble_matrix!((u,v) -> a(v,u,φh,dΩ),adjoint_K,assem_adjoint,V,U)

### Test
@time op_test = AffineFEOperator((u,v) -> a(u,v,φh,dΩ),v -> l(v,φh,dΩ),U,V,SparseMatrixAssembler(U,V))
K_test = get_matrix(op_test);
@show Base.summarysize(K)
@show Base.summarysize(K_test)

if typeof(K_test) <: BlockArray
  @assert K_test == K
else
  @assert all(K.blocks[1,1].matrix_partition.items[:] .≈ K_test.blocks[1,1].matrix_partition.items[:])
  @assert all(K.blocks[2,2].matrix_partition.items[:] .≈ K_test.blocks[2,2].matrix_partition.items[:])
  @assert all(K.blocks[3,3].matrix_partition.items[:] .≈ K_test.blocks[3,3].matrix_partition.items[:])
end

# # OLD
# # # First iteration
# # full_assem = DiagonalBlockMatrixAssembler(assem=SparseMatrixAssembler(U,V));
# # full_du = get_trial_fe_basis(U);
# # full_dv = get_fe_basis(V);
# # full_matcontribs = a(full_du[diag_block_axes],full_dv[diag_block_axes],φh,dΩ)
# # full_data = Gridap.FESpaces.collect_cell_matrix(U,V,full_matcontribs);
# # @time full_A_cached = allocate_matrix(full_assem,full_data);
# # Base.summarysize(full_A_cached)

# # # nth iteration
# # matcontribs = a(full_du[diag_block_axes],full_dv[diag_block_axes],φh,dΩ)
# # data = Gridap.FESpaces.collect_cell_matrix(U,V,matcontribs);
# # assemble_matrix!(full_A_cached,full_assem,data)

# # blocks_size = size(full_A_cached.blocks,1);
# # block_iter = blocks_size % last(diag_block_axes)
# # @check iszero(block_iter) "Inconsistant number of blocks to match `diag_block_axes`: 
# #     Expected to fit multiples of $diag_block_axes blocks into $(blocks_size)x$(blocks_size) block matrix."
# # nothing

# # for i ∈ Iterators.partition(last(diag_block_axes)+1:blocks_size,last(diag_block_axes))
# #   full_A_cached.blocks[i,i] = full_A_cached.blocks[diag_block_axes,diag_block_axes]
# # end
# # Base.summarysize(full_A_cached)