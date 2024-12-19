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
  @show typeof(x)
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
function Gridap.FESpaces.AffineFEOperator(
  weakform::Function,trial::FESpace,test::FESpace,assem::Assembler)
  if ! isa(test,TrialFESpace)
    @warn """\n
    You are building an AffineFEOperator with a test space of type TrialFESpace.

    This may result in unexpected behaviour.
    """ maxlog=1
  end

  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)

  uhd = zero(trial)
  matcontribs, veccontribs = weakform(u,v)
  data = collect_cell_matrix_and_vector(trial,test,matcontribs,veccontribs,uhd)
  A,b = assemble_matrix_and_vector(assem,data)
  #GC.gc()

  AffineFEOperator(trial,test,A,b)
end

# Base.one for FESpace
function Base.one(f::FESpace)
  uh = zero(f)
  u = get_free_dof_values(uh)
  V = get_vector_type(f)
  fill!(u,one(eltype(V)))
  return uh
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