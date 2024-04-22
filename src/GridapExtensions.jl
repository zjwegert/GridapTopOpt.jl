############################################################################################
###  These are things that should eventually be moved to official Gridap packages        ###
############################################################################################

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

function get_local_matrix_type(a::GridapDistributed.DistributedSparseMatrixAssembler)
  return getany(map(get_matrix_type,a.assems))
end

function get_local_vector_type(a::GridapDistributed.DistributedSparseMatrixAssembler)
  return getany(map(get_vector_type,a.assems))
end
