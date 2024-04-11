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

function Gridap.FESpaces.allocate_matrix(a::Function,assem::Assembler,U::FESpace,V::FESpace)
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  allocate_matrix(assem,collect_cell_matrix(U,V,a(u,v)))
end

function Gridap.FESpaces.allocate_vector(l::Function,assem::Assembler,V::FESpace)
  v = get_fe_basis(V)
  allocate_vector(assem,collect_cell_vector(V,l(v)))
end

function Gridap.FESpaces.assemble_matrix_and_vector!(
    a::Function,l::Function,A::AbstractMatrix,b::AbstractVector,assem::Assembler,U::FESpace,V::FESpace,uhd)
  v = get_fe_basis(V)
  u = get_trial_fe_basis(U)
  assemble_matrix_and_vector!(A,b,assem,collect_cell_matrix_and_vector(U,V,a(u,v),l(v),uhd))
end

# PETScNonlinearSolver override

function Gridap.Algebra.solve!(x::T,nls::PETScNonlinearSolver,op::Gridap.Algebra.NonlinearOperator,
    cache::GridapPETSc.PETScNonlinearSolverCache{<:T}) where T <: AbstractVector
  @check_error_code GridapPETSc.PETSC.SNESSolve(cache.snes[],C_NULL,cache.x_petsc.vec[])
  copy!(x,cache.x_petsc)
  cache
end

# Stuff from ODE refactor 

function (+)(a::Gridap.CellData.DomainContribution,b::GridapDistributed.DistributedDomainContribution)
  @assert iszero(Gridap.CellData.num_domains(a))
  return b
end

# GridapDistributed

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