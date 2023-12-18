"""
  DiagonalBlockMatrixAssembler

  Fields:
    assem::Assembler
    diag_block_axes::UnitRange{Int64}

  This assembler is designed for multifield FE problems where the resulting 
    block stiffness matrix possesses identical entries down the diagonal which
    have a subblock size of `diag_block_axes`. For example, consider elastic 
    inverse homogenisation. The weak form is of the form: Find u∈Uⁿ s.t.,
   
    ∫ Σ₁ⁿ(C ⊙ ε(u[i]) ⊙ ε(v[i]) dΩ = -∫ Σ₁ⁿ(C ⊙ εᴹ[i] ⊙ ε(v[i])) dΩ ∀v∈V

    Once discretised using finite elements this has the block form:

        ┌          ┐
        │ A      0 │ 
    K = │    B     │
        │ 0     ...│
        └          ┘

    where the diagonal entries [A,B,...] are equal. To avoid allocating or
    inplace calculation of all diagonal entries we use `DiagonalBlockMatrixAssembler`
    with `diag_block_axes=1:1` to tell the assembler that we have a special repeated
    structure with subblock size of 1. The assembler then only builds the matrix A 
    and points the other entries in the diagonal to this block. As a result only
    A is calculated.

  A more complicated example is when we have subblock structures down the diagonal.
    For example, in piezoelectric homogenisation the bilinear form has the structure:

    a(uφ,vq) = ∑₁ⁿ {aᵤᵤ(uφ[2i-1],vq[2i-1]) + aᵤᵩ(uφ[2i-1],vq[2i]) + 
                      aᵩᵤ(uφ[2i],vq[2i-1]) + aᵩᵩ(uφ[2i],vq[2i])} 

    where we assume the MultiFieldFESpace has ordering
      UΦ = [U1,Φ1,U2,Φ2,...]
      VQ = [V1,Q1,V2,Q2,...]

    Once discretised this has the form 

         ┌                           ┐
         │ A₁₁ A₁₂ │                 │
         │ A₂₁ A₁₂ │             0   │
         ├─────────┼─────────┐       │
         │         │ A₃₃ A₃₄ │       │
    K =  │         │ A₄₃ A₄₄ │       │
         │         └─────────┼───────┤
         │    0              │  ...  │
         │                   │       │
         └                           ┘ 
    
    where the diagonal subblocks are equal. In this case we take `diag_block_axes=1:2`

  Assumptions:

    1. If a diagonal block has a subblock structure 
        these must correspond to `diag_block_axes` and
        be ordered in the manor they appear. For example 
        in the piezoelectric problem above taking i = 1
        gives the first subblock in K. 

    2. The block ordering must not change via `BlockMultiFieldStyle`.
"""
struct DiagonalBlockMatrixAssembler{A<:Assembler} <: SparseMatrixAssembler 
  assem::A
  diag_block_axes::UnitRange{Int64}
  function DiagonalBlockMatrixAssembler(assem::A,diag_block_axes::UnitRange{Int64}=1:1) where A<:Assembler
    @assert isone(first(diag_block_axes)) && last(diag_block_axes) >= first(diag_block_axes) 
    new{A}(assem,diag_block_axes)
  end
end

function _identical_diag_block_assemble!(A::AbstractMatrix,diag_block_axes::UnitRange)
  @check typeof(A) <: BlockArrays.AbstractBlockArray "`DiagonalBlockMatrixAssembler`"*
    " expects a block structure\n\n   Recieved $(typeof(A)).\n\n   Check that you are using `BlockMultiFieldStyle`"
  blocks_size = size(A.blocks,1);
  block_iter = blocks_size % last(diag_block_axes)
  @check iszero(block_iter) "Inconsistant number of blocks to match `diag_block_axes`: 
      Expected to fit multiples of $diag_block_axes blocks into $(blocks_size)x$(blocks_size) block matrix."
  # Set diagonal
  for i ∈ Iterators.partition(last(diag_block_axes)+1:blocks_size,last(diag_block_axes))
    A.blocks[i,i] = A.blocks[diag_block_axes,diag_block_axes]
  end
  # Allocate empty blocks 
  _blocks = Iterators.partition(Base.OneTo(blocks_size),last(diag_block_axes))
  _non_empty_blocks = Iterators.flatten(Iterators.product.(_blocks,_blocks))
  for I ∈ CartesianIndices(A.blocks)
    I.I ∈ _non_empty_blocks && continue
    A.blocks[I] = zero_block(eltype(A.blocks),axes(A.blocks[I[1],I[1]],1),axes(A.blocks[I[2],I[2]],2))
  end
  # Checks that axes are consistant
  @check ~isnothing(mortar(A.blocks))
  return nothing
end

## AffineFEOperator & Assemblers
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

function assemble_matrix_and_vector(
    a::Function,l::Function,assem::DiagonalBlockMatrixAssembler,U::FESpace,V::FESpace)
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

function assemble_matrix!(
    a::Function,A::AbstractMatrix,assem::DiagonalBlockMatrixAssembler,U::FESpace,V::FESpace)
  diag_block_axes = assem.diag_block_axes
  _assem = assem.assem
  
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  assemble_matrix!(A,_assem,collect_cell_matrix(U,V,a(u[diag_block_axes],v[diag_block_axes])))
  _identical_diag_block_assemble!(A,diag_block_axes)
end

function assemble_matrix(
    a::Function,assem::DiagonalBlockMatrixAssembler,U::FESpace,V::FESpace)
  diag_block_axes = assem.diag_block_axes
  _assem = assem.assem
  
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  A = assemble_matrix(_assem,collect_cell_matrix(U,V,a(u[diag_block_axes],v[diag_block_axes])))
  _identical_diag_block_assemble!(A,diag_block_axes)
  return A
end

allocate_vector(a::DiagonalBlockMatrixAssembler,data) = allocate_vector(a.assem,data)
assemble_vector(a::DiagonalBlockMatrixAssembler,data) = assemble_vector(a.assem,data)
assemble_vector!(b,a::DiagonalBlockMatrixAssembler,data) = assemble_vector!(b,a.assem,data)

# Custom inplace assembler
function _assemble_matrix_and_vector!(
    a::Function,l::Function,A::AbstractMatrix,b::AbstractVector,assem::Assembler,U::FESpace,V::FESpace,uhd)
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  assemble_matrix_and_vector!(A,b,assem,collect_cell_matrix_and_vector(U,V,a(u,v),l(v),uhd))
end

function _assemble_matrix_and_vector!(
    a::Function,l::Function,A::AbstractMatrix,b::AbstractVector,assem::DiagonalBlockMatrixAssembler,U::FESpace,V::FESpace,uhd)
  diag_block_axes = assem.diag_block_axes
  _assem = assem.assem
  
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  matcontribs, veccontribs = a(u[diag_block_axes],v[diag_block_axes]),l(v)
  data = collect_cell_matrix_and_vector(U,V,matcontribs,veccontribs,uhd)
  assemble_matrix_and_vector!(A,b,_assem,data)
  _identical_diag_block_assemble!(A,diag_block_axes)
end

## Zero block assemble
function zero_block(::Type{SparseMatrixCSC{Tv,Ti}},rows,cols) where {Tv,Ti}
  m = length(rows)
  n = length(cols)
  return SparseMatrixCSC(m,n,fill(Ti(1),n+1),Ti[],Tv[])
end

function zero_block(::Type{SparseMatrixCSR{Bi,Tv,Ti}},rows,cols) where {Bi,Tv,Ti}
  SparseMatrixCSR{Bi}(transpose(zero_block(SparseMatrixCSC{Tv,Ti},cols,rows)))
end

function zero_block(::Type{<:PSparseMatrix{Tm}},rows,cols) where Tm
  mats = map(partition(rows),partition(cols)) do rows,cols
    zero_block(Tm,rows,cols)
  end
  return PSparseMatrix(mats,partition(rows),partition(cols))
end

## Indexing operations
Base.length(::DistributedCellField) = 1;
Base.length(a::DistributedMultiFieldFEBasis) = length(a.field_fe_basis);
Base.length(a::MultiFieldCellField) = length(a.single_fields);
Base.length(a::MultiFieldFEFunction) = length(a.single_fe_functions);
Base.getindex(a::MultiFieldCellField,i::UnitRange) = a.single_fields[i]
Base.getindex(a::DistributedMultiFieldFEBasis,i::UnitRange) = a.field_fe_basis[i]
Base.getindex(a::MultiFieldFEFunction,i::UnitRange) = a.single_fe_functions[i]