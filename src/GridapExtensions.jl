############################################################################################
###  These are things that should eventually be moved to official Gridap packages        ###
############################################################################################


################# Instantiate solvers #################

# Instantiate nonlinear solver caches (without actually doing the first iteration)
function instantiate_caches(x,nls::NLSolver,op::NonlinearOperator)
  Gridap.Algebra._new_nlsolve_cache(x,nls,op)
end

function instantiate_caches(x,nls::NewtonRaphsonSolver,op::NonlinearOperator)
  b = residual(op, x)
  A = jacobian(op, x)
  dx = similar(b)
  ss = symbolic_setup(nls.ls, A)
  ns = numerical_setup(ss,A)
  return Gridap.Algebra.NewtonRaphsonCache(A,b,dx,ns)
end

function instantiate_caches(x,nls::NewtonSolver,op::NonlinearOperator)
  b  = residual(op, x)
  A  = jacobian(op, x)
  dx = allocate_in_domain(A); fill!(dx,zero(eltype(dx)))
  ss = symbolic_setup(nls.ls, A)
  ns = numerical_setup(ss,A,x)
  return GridapSolvers.NonlinearSolvers.NewtonCache(A,b,dx,ns)
end

function instantiate_caches(x,nls::PETScNonlinearSolver,op::NonlinearOperator)
  return GridapPETSc._setup_cache(x,nls,op)
end

function GridapTopOpt.instantiate_caches(x,ls::LinearSolver,op::Gridap.Algebra.AffineOperator)
  numerical_setup(symbolic_setup(ls,get_matrix(op)),get_matrix(op))
end

function GridapTopOpt.instantiate_caches(x::PVector,ls::LinearSolver,op::Gridap.Algebra.AffineOperator)
  numerical_setup(symbolic_setup(ls,get_matrix(op)),get_matrix(op)), allocate_in_domain(get_matrix(op))
end

################# Assembly #################
# Transpose contributions before assembly
transpose_contributions(b::DistributedDomainContribution) =
  DistributedDomainContribution(map(transpose_contributions,local_views(b)))

function transpose_contributions(b::DomainContribution)
  c = DomainContribution()
  for (trian,array_old) in b.dict
    array_new = lazy_map(transpose,array_old)
    add_contribution!(c,trian,array_new)
  end
  return c
end

function assemble_adjoint_matrix(f::Function,a::Assembler,U::FESpace,V::FESpace)
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  contr = transpose_contributions(f(u,v))
  assemble_matrix(a,collect_cell_matrix(V,U,contr))
end

function assemble_adjoint_matrix!(f::Function,A::AbstractMatrix,a::Assembler,U::FESpace,V::FESpace)
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  contr = transpose_contributions(f(u,v))
  assemble_matrix!(A,a,collect_cell_matrix(V,U,contr))
end

# Assembly addons

get_local_matrix_type(a::Assembler) = get_matrix_type(a)
get_local_vector_type(a::Assembler) = get_vector_type(a)
get_local_assembly_strategy(a::Assembler) = get_assembly_strategy(a)

function get_local_matrix_type(a::GridapDistributed.DistributedSparseMatrixAssembler)
  return getany(map(get_matrix_type,a.assems))
end
function get_local_vector_type(a::GridapDistributed.DistributedSparseMatrixAssembler)
  return getany(map(get_vector_type,a.assems))
end
function get_local_assembly_strategy(a::GridapDistributed.DistributedSparseMatrixAssembler)
  return get_assembly_strategy(a)
end

function get_local_matrix_type(a::MultiField.BlockSparseMatrixAssembler)
  return get_local_matrix_type(first(a.block_assemblers))
end
function get_local_vector_type(a::MultiField.BlockSparseMatrixAssembler)
  return get_local_vector_type(first(a.block_assemblers))
end
function get_local_assembly_strategy(a::MultiField.BlockSparseMatrixAssembler)
  return get_local_assembly_strategy(first(a.block_assemblers))
end

################# Gridap convience #################
# Allow to swap test and trial in AffineFEOperator
# TODO: Don't do this, currently it will cause problems with AD!!
# function Gridap.FESpaces.AffineFEOperator(
#   weakform::Function,trial::FESpace,test::FESpace,assem::Assembler)
#   if isa(test,TrialFESpace)
#     @warn """\n
#     You are building an AffineFEOperator with a test space of type TrialFESpace.

#     This may result in unexpected behaviour.
#     """ maxlog=1
#   end

#   u = get_trial_fe_basis(trial)
#   v = get_fe_basis(test)

#   uhd = zero(trial)
#   matcontribs, veccontribs = weakform(u,v)
#   data = collect_cell_matrix_and_vector(trial,test,matcontribs,veccontribs,uhd)
#   A,b = assemble_matrix_and_vector(assem,data)
#   #GC.gc()

#   AffineFEOperator(trial,test,A,b)
# end

# Base.one for FESpace
function Base.one(f::FESpace)
  uh = zero(f)
  u = get_free_dof_values(uh)
  V = get_vector_type(f)
  fill!(u,one(eltype(V)))
  return uh
end

################# GridapDistributed #################

function GridapDistributed.to_parray_of_arrays(a::NTuple{N,T}) where {N,T<:DebugArray}
  indices = linear_indices(first(a))
  map(indices) do i
    map(a) do aj
      aj.items[i]
    end
  end
end

function GridapDistributed.to_parray_of_arrays(a::NTuple{N,T}) where {N,T<:MPIArray}
  indices = linear_indices(first(a))
  map(indices) do i
    map(a) do aj
      PartitionedArrays.getany(aj)
    end
  end
end

################# GridapSolvers #################
## Get solutions from vector of spaces
function GridapSolvers.BlockSolvers.get_solution(spaces::Vector{<:FESpace}, xh::MultiFieldFEFunction, k)
  r = MultiField.get_block_ranges(spaces)[k]
  if isone(length(r)) # SingleField
    xh_k = xh[r[1]]
  else # MultiField
    fv_k = blocks(get_free_dof_values(xh))[k]
    xh_k = MultiFieldFEFunction(fv_k, spaces[k], xh.single_fe_functions[r])
  end
  return xh_k
end

function GridapSolvers.BlockSolvers.get_solution(spaces::Vector{<:FESpace}, xh::DistributedMultiFieldFEFunction, k)
  r = MultiField.get_block_ranges(spaces)[k]
  if isone(length(r)) # SingleField
    xh_k = xh[r[1]]
  else # MultiField
    sf_k = xh.field_fe_fun[r]
    fv_k = blocks(get_free_dof_values(xh))[k]
    mf_k = map(local_views(spaces[k]),partition(fv_k),map(local_views,sf_k)...) do Vk, fv_k, sf_k...
      MultiFieldFEFunction(fv_k, Vk, [sf_k...])
    end
    xh_k = DistributedMultiFieldFEFunction(sf_k, mf_k, fv_k)
  end
  return xh_k
end

function MultiField.get_block_ranges(spaces::Vector{<:FESpace})
  NB = length(spaces)
  SB = Tuple(map(num_fields,spaces))
  MultiField.get_block_ranges(NB,SB,Tuple(1:sum(SB)))
end

# StaggeredAdjointAffineFEOperator
# TODO: Jordi, this was created because there is a problem with AD when the
#   trial and test spaces are swapped. The below fixes this because we assemble
#   the adjoint directly. We should look into this problem in future.

using GridapSolvers.BlockSolvers: BlockFESpaceTypes, BlockSparseMatrixAssembler

"""
    struct StaggeredAdjointAffineFEOperator{NB,SB} <: StaggeredFEOperator{NB,SB}
      ...
    end

Affine staggered operator, used to solve staggered problems
where the k-th equation is linear in `u_k` and the transpose of the stiffness
matrix is assembled.

Such a problem is formulated by a set of bilinear/linear form pairs:

    a_k((u_1,...,u_{k-1}),u_k,v_k) = ∫(...)
    l_k((u_1,...,u_{k-1}),v_k) = ∫(...)

than cam be assembled into a set of linear systems:

    A_kᵀ u_k = b_k

where `A_kᵀ` and `b_k` only depend on the previous variables `u_1,...,u_{k-1}` and
`A_kᵀ` is the tranpose of `A_k`.
"""
struct StaggeredAdjointAffineFEOperator{NB,SB} <: StaggeredFEOperator{NB,SB}
  biforms :: Vector{<:Function}
  liforms :: Vector{<:Function}
  trials  :: Vector{<:FESpace}
  tests   :: Vector{<:FESpace}
  assems  :: Vector{<:Assembler}
  trial   :: BlockFESpaceTypes{NB,SB}
  test    :: BlockFESpaceTypes{NB,SB}

  @doc """
    function StaggeredAdjointAffineFEOperator(
      biforms :: Vector{<:Function},
      liforms :: Vector{<:Function},
      trials  :: Vector{<:FESpace},
      tests   :: Vector{<:FESpace},
      [assems :: Vector{<:Assembler}]
    )

  Constructor for a `StaggeredAdjointAffineFEOperator` operator, taking in each
  equation as a pair of bilinear/linear forms and the corresponding trial/test spaces.
  The trial/test spaces can be single or multi-field spaces.
  """
  function StaggeredAdjointAffineFEOperator(
    biforms :: Vector{<:Function},
    liforms :: Vector{<:Function},
    trials  :: Vector{<:FESpace},
    tests   :: Vector{<:FESpace},
    assems  :: Vector{<:Assembler} = map(SparseMatrixAssembler,trials,tests)
  )
    @assert length(biforms) == length(liforms) == length(trials) == length(tests) == length(assems)
    trial = combine_fespaces(trials)
    test  = combine_fespaces(tests)
    NB, SB = length(trials), Tuple(map(num_fields,trials))
    new{NB,SB}(biforms,liforms,trials,tests,assems,trial,test)
  end

  @doc """
    function StaggeredAdjointAffineFEOperator(
      biforms :: Vector{<:Function},
      liforms :: Vector{<:Function},
      trial   :: BlockFESpaceTypes{NB,SB,P},
      test    :: BlockFESpaceTypes{NB,SB,P},
      [assem  :: BlockSparseMatrixAssembler{NB,NV,SB,P}]
    ) where {NB,NV,SB,P}

  Constructor for a `StaggeredAdjointAffineFEOperator` operator, taking in each
  equation as a pair of bilinear/linear forms and the global trial/test spaces.
  """
  function StaggeredAdjointAffineFEOperator(
    biforms :: Vector{<:Function},
    liforms :: Vector{<:Function},
    trial   :: BlockFESpaceTypes{NB,SB,P},
    test    :: BlockFESpaceTypes{NB,SB,P},
    assem   :: BlockSparseMatrixAssembler{NB,NV,SB,P} = SparseMatrixAssembler(trial,test)
  ) where {NB,NV,SB,P}
    @assert length(biforms) == length(liforms) == NB
    @assert P == Tuple(1:sum(SB)) "Permutations not supported"
    trials = blocks(trial)
    tests  = blocks(test)
    assems = diag(blocks(assem))
    new{NB,SB}(biforms,liforms,trials,tests,assems,trial,test)
  end
end

FESpaces.get_trial(op::StaggeredAdjointAffineFEOperator) = op.trial
FESpaces.get_test(op::StaggeredAdjointAffineFEOperator) = op.test

function GridapSolvers.BlockSolvers.get_operator(op::StaggeredAdjointAffineFEOperator{NB}, xhs, k) where NB
  @assert NB >= k
  a(uk,vk) = op.biforms[k](xhs,uk,vk)
  l(vk) = op.liforms[k](xhs,vk)
  A = assemble_adjoint_matrix(a,op.assems[k],op.trials[k],op.tests[k])
  b = assemble_vector(l,op.assems[k],op.tests[k])
  affine_op = AffineOperator(A,b)
  return AffineFEOperator(op.trials[k],op.tests[k],affine_op)
end

function GridapSolvers.BlockSolvers.get_operator!(op_k::AffineFEOperator, op::StaggeredAdjointAffineFEOperator{NB}, xhs, k) where NB
  @assert NB >= k
  A, b = get_matrix(op_k), get_vector(op_k)
  a(uk,vk) = op.biforms[k](xhs,uk,vk)
  l(vk) = op.liforms[k](xhs,vk)
  assemble_adjoint_matrix!(a,A,op.assems[k],op.trials[k],op.tests[k])
  assemble_vector!(l,b,op.assems[k],op.tests[k])
  return op_k
end

# function assemble_adjoint_matrix_and_vector(f::Function,b::Function,a::Assembler,U::FESpace,V::FESpace,uhd)
#   v = get_fe_basis(V)
#   u = get_trial_fe_basis(U)
#   fcontr = transpose_contributions(f(u,v))
#   assemble_matrix_and_vector(a,collect_cell_matrix_and_vector(V,U,fcontr,b(v),uhd))
# end

# function assemble_adjoint_matrix_and_vector!(f::Function,b::Function,A::AbstractMatrix,B::AbstractVector,
#     a::Assembler,U::FESpace,V::FESpace,uhd)
#   v = get_fe_basis(V)
#   u = get_trial_fe_basis(U)
#   fcontr = transpose_contributions(f(u,v))
#   assemble_matrix_and_vector!(A,B,a,collect_cell_matrix_and_vector(V,U,fcontr,b(v),uhd))
# end

# function GridapSolvers.BlockSolvers.get_operator(op::StaggeredAdjointAffineFEOperator{NB}, xhs, k) where NB
#   @assert NB >= k
#   a(uk,vk) = op.biforms[k](xhs,uk,vk)
#   l(vk) = op.liforms[k](xhs,vk)
#   A, b = assemble_adjoint_matrix_and_vector(a,l,op.assems[k],op.trials[k],op.tests[k],zero(op.tests[k]))
#   affine_op = AffineOperator(A,b)
#   return AffineFEOperator(op.trials[k],op.tests[k],affine_op)
# end

# function GridapSolvers.BlockSolvers.get_operator!(op_k::AffineFEOperator, op::StaggeredAdjointAffineFEOperator{NB}, xhs, k) where NB
#   @assert NB >= k
#   A, b = get_matrix(op_k), get_vector(op_k)
#   a(uk,vk) = op.biforms[k](xhs,uk,vk)
#   l(vk) = op.liforms[k](xhs,vk)
#   assemble_adjoint_matrix_and_vector!(a,l,A,b,op.assems[k],op.trials[k],op.tests[k],zero(op.tests[k]))
#   return op_k
# end

################# Embedded #################
# Fix for isbitstype bug in Gridap.Polynomials
function Arrays.return_cache(
  fg::Fields.FieldGradientArray{1,Polynomials.MonomialBasis{D,V}},
  x::AbstractVector{<:Point}) where {D,V}
  xi = testitem(x)
  T = gradient_type(V,xi)
  Polynomials._return_cache(fg,x,T,Val(false))
end

function Arrays.evaluate!(
  cache,
  fg::Fields.FieldGradientArray{1,Polynomials.MonomialBasis{D,V}},
  x::AbstractVector{<:Point}) where {D,V}
  Polynomials._evaluate!(cache,fg,x,Val(false))
end

# Fix for autodiff of CompositeTriangulations of Skeleton trians

function FESpaces._change_argument(
  op,f,trian::Geometry.CompositeTriangulation{Dc,Dp,A,<:SkeletonTriangulation},uh::SingleFieldFEFunction
) where {Dc,Dp,A}
  U = get_fe_space(uh)
  function g(cell_u)
    uh_dual = CellField(U,cell_u)
    scfp_plus = CellData.SkeletonCellFieldPair(uh_dual, uh)
    scfp_minus = CellData.SkeletonCellFieldPair(uh, uh_dual)
    cell_grad_plus = f(scfp_plus)
    cell_grad_minus = f(scfp_minus)
    CellData.get_contribution(cell_grad_plus,trian), CellData.get_contribution(cell_grad_minus,trian)
  end
  g
end

function FESpaces._compute_cell_ids(
  uh,ttrian::Geometry.CompositeTriangulation{Dc,Dp,A,<:SkeletonTriangulation}
) where {Dc,Dp,A}
  tcells_plus  = FESpaces._compute_cell_ids(uh,ttrian.dtrian.plus)
  tcells_minus = FESpaces._compute_cell_ids(uh,ttrian.dtrian.minus)
  CellData.SkeletonPair(tcells_plus,tcells_minus)
end

# # TODO: Below is dangerous, as it may break other Gridap methods,
# #   it is neccessary for now - see thermal_2d.jl problem
# function FESpaces._compute_cell_ids(uh,ttrian)
#   strian = get_triangulation(uh)
#   if strian === ttrian
#     return collect(IdentityVector(Int32(num_cells(strian))))
#   end
#   @check is_change_possible(strian,ttrian)
#   D = num_cell_dims(strian)
#   sglue = get_glue(strian,Val(D))
#   tglue = get_glue(ttrian,Val(D))
#   @notimplementedif !isa(sglue,FaceToFaceGlue)
#   @notimplementedif !isa(tglue,FaceToFaceGlue)
#   scells = IdentityVector(Int32(num_cells(strian)))
#   mcells = extend(scells,sglue.mface_to_tface)
#   tcells = lazy_map(Reindex(mcells),tglue.tface_to_mface)
#   # <-- Remove collect to keep PosNegReindex
#   # tcells = collect(tcells)
#   return tcells
# end

# # New dispatching
# function Arrays.lazy_map(k::Reindex,ids::Arrays.LazyArray{<:Fill{<:PosNegReindex}})
#   k_posneg = ids.maps.value
#   posneg_partition = ids.args[1]
#   pos_values = lazy_map(Reindex(k.values),k_posneg.values_pos)
#   pos_values, neg_values = Geometry.pos_neg_data(pos_values,posneg_partition)
#   # println("Byee ::: $(eltype(pos_values)) --- $(eltype(neg_values))")
#   lazy_map(PosNegReindex(pos_values,neg_values),posneg_partition)
# end

# function Arrays.lazy_map(k::Reindex,ids::Arrays.AppendedArray)
#   a = lazy_map(k,ids.a)
#   b = lazy_map(k,ids.b)
#   # println("Hello ::: $(eltype(a)) --- $(eltype(b))")
#   return lazy_append(a,b)
# end

# using ForwardDiff

# function Arrays.evaluate!(result,k::AutoDiffMap,ydual,x,cfg::ForwardDiff.GradientConfig{T}) where T
#   @notimplementedif ForwardDiff.chunksize(cfg) != length(x)
#   @notimplementedif length(result) != length(x)
#   !isempty(x) && ForwardDiff.extract_gradient!(T, result, ydual) # <-- Watch for empty cell contributions
#   return result
# end

################# MultiField #################
# TODO: Remove this once resolved in GridapDistributed#169
function test_triangulation(Ω1,Ω2)
  @assert typeof(Ω1.grid) == typeof(Ω2.grid)
  t = map(fieldnames(typeof(Ω1.grid))) do field
    getfield(Ω1.grid,field) == getfield(Ω2.grid,field)
  end
  all(t)
  a = Ω1.model === Ω2.model
  b = Ω1.tface_to_mface == Ω2.tface_to_mface
  a && b && all(t)
end

function CellData.get_triangulation(f::MultiFieldCellField)
  s1 = first(f.single_fields)
  trian = get_triangulation(s1)
  @check all(map(i->test_triangulation(trian,get_triangulation(i)),f.single_fields))
  trian
end