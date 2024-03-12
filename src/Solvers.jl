## TODO: @Jordi, are these going into GridapSolvers or should I write documentation?

struct ElasticitySolver{A} <: LinearSolver
  space ::A
  rtol  ::PetscScalar
  maxits::PetscInt
  function ElasticitySolver(space::FESpace;
                            rtol=1.e-12,
                            maxits=200)
    A = typeof(space)
    new{A}(space,rtol,maxits)
  end
end

struct ElasticitySymbolicSetup{A} <: SymbolicSetup
  solver::A
end

function Gridap.Algebra.symbolic_setup(solver::ElasticitySolver,A::AbstractMatrix)
  ElasticitySymbolicSetup(solver)
end

function get_dof_coordinates(space::GridapDistributed.DistributedSingleFieldFESpace)
  coords  = map(local_views(space),partition(space.gids)) do space, dof_ids
    local_to_own_dofs = local_to_own(dof_ids)
    return get_dof_coordinates(space;perm=local_to_own_dofs)
  end

  ngdofs  = length(space.gids)
  indices = map(local_views(space.gids)) do dof_indices
    owner = part_id(dof_indices)
    own_indices   = OwnIndices(ngdofs,owner,own_to_global(dof_indices))
    ghost_indices = GhostIndices(ngdofs,Int64[],Int32[]) # We only consider owned dofs
    OwnAndGhostIndices(own_indices,ghost_indices)   
  end
  return PVector(coords,indices)
end

function get_dof_coordinates(space::FESpace;perm=Base.OneTo(num_free_dofs(space)))
  trian = get_triangulation(space)
  cell_dofs = get_fe_dof_basis(space)
  cell_ids  = get_cell_dof_ids(space)

  cell_ref_nodes = lazy_map(get_nodes,CellData.get_data(cell_dofs))
  cell_dof_to_node = lazy_map(get_dof_to_node,CellData.get_data(cell_dofs))
  cell_dof_to_comp = lazy_map(get_dof_to_comp,CellData.get_data(cell_dofs))

  cmaps = get_cell_map(trian)
  cell_phys_nodes = lazy_map(evaluate,cmaps,cell_ref_nodes)

  node_coords = Vector{Float64}(undef,maximum(perm))
  cache_nodes = array_cache(cell_phys_nodes)
  cache_ids = array_cache(cell_ids)
  cache_dof_to_node = array_cache(cell_dof_to_node)
  cache_dof_to_comp = array_cache(cell_dof_to_comp)
  for cell in 1:num_cells(trian)
    ids = getindex!(cache_ids,cell_ids,cell)
    nodes = getindex!(cache_nodes,cell_phys_nodes,cell)
    dof_to_comp = getindex!(cache_dof_to_comp,cell_dof_to_comp,cell)
    dof_to_node = getindex!(cache_dof_to_node,cell_dof_to_node,cell)
    for (dof,c,n) in zip(ids,dof_to_comp,dof_to_node)
      if (dof > 0) && (perm[dof] > 0)
        node_coords[perm[dof]] = nodes[n][c]
      end
    end
  end
  return node_coords
end

function elasticity_ksp_setup(ksp,rtol,maxits)
  rtol = PetscScalar(rtol)
  atol = GridapPETSc.PETSC.PETSC_DEFAULT
  dtol = GridapPETSc.PETSC.PETSC_DEFAULT
  maxits = PetscInt(maxits)

  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPCG)
  @check_error_code GridapPETSc.PETSC.KSPSetTolerances(ksp[], rtol, atol, dtol, maxits)

  pc = Ref{GridapPETSc.PETSC.PC}()
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCGAMG)

  @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
end

mutable struct ElasticityNumericalSetup <: NumericalSetup
  A::PETScMatrix
  X::PETScVector
  B::PETScVector
  ksp::Ref{GridapPETSc.PETSC.KSP}
  null::Ref{GridapPETSc.PETSC.MatNullSpace}
  initialized::Bool
  function ElasticityNumericalSetup(A::PETScMatrix,X::PETScVector,B::PETScVector)
    ksp  = Ref{GridapPETSc.PETSC.KSP}()
    null = Ref{GridapPETSc.PETSC.MatNullSpace}()
    new(A,X,B,ksp,null,false)
  end
end

function GridapPETSc.Init(a::ElasticityNumericalSetup)
  @assert Threads.threadid() == 1
  GridapPETSc._NREFS[] += 2
  a.initialized = true
  finalizer(GridapPETSc.Finalize,a)
end

function GridapPETSc.Finalize(ns::ElasticityNumericalSetup)
  if ns.initialized && GridapPETSc.Initialized()
    if ns.A.comm == MPI.COMM_SELF
      @check_error_code GridapPETSc.PETSC.KSPDestroy(ns.ksp)
      @check_error_code GridapPETSc.PETSC.MatNullSpaceDestroy(ns.null)
    else
      @check_error_code GridapPETSc.PETSC.PetscObjectRegisterDestroy(ns.ksp[].ptr)
      @check_error_code GridapPETSc.PETSC.PetscObjectRegisterDestroy(ns.null[].ptr)
    end
    ns.initialized = false
    @assert Threads.threadid() == 1
    GridapPETSc._NREFS[] -= 2
  end
  nothing
end

_num_dims(space::FESpace) = num_cell_dims(get_triangulation(space))
_num_dims(space::GridapDistributed.DistributedSingleFieldFESpace) = getany(map(_num_dims,local_views(space)))

function Gridap.Algebra.numerical_setup(ss::ElasticitySymbolicSetup,_A::PSparseMatrix)
  s = ss.solver

  # Create ns 
  A = convert(PETScMatrix,_A)
  X = convert(PETScVector,allocate_in_domain(_A))
  B = convert(PETScVector,allocate_in_domain(_A))
  ns = ElasticityNumericalSetup(A,X,B)

  # Compute  coordinates for owned dofs
  dof_coords = convert(PETScVector,get_dof_coordinates(s.space))
  @check_error_code GridapPETSc.PETSC.VecSetBlockSize(dof_coords.vec[],_num_dims(s.space))

  # Create matrix nullspace
  @check_error_code GridapPETSc.PETSC.MatNullSpaceCreateRigidBody(dof_coords.vec[],ns.null)
  @check_error_code GridapPETSc.PETSC.MatSetNearNullSpace(ns.A.mat[],ns.null[])
  
  # Setup solver and preconditioner
  @check_error_code GridapPETSc.PETSC.KSPCreate(ns.A.comm,ns.ksp)
  @check_error_code GridapPETSc.PETSC.KSPSetOperators(ns.ksp[],ns.A.mat[],ns.A.mat[])
  elasticity_ksp_setup(ns.ksp,s.rtol,s.maxits)
  @check_error_code GridapPETSc.PETSC.KSPSetUp(ns.ksp[])
  GridapPETSc.Init(ns)
end

function Gridap.Algebra.numerical_setup!(ns::ElasticityNumericalSetup,A::AbstractMatrix)
  ns.A = convert(PETScMatrix,A)
  @check_error_code GridapPETSc.PETSC.MatSetNearNullSpace(ns.A.mat[],ns.null[])
  @check_error_code GridapPETSc.PETSC.KSPSetOperators(ns.ksp[],ns.A.mat[],ns.A.mat[])
  @check_error_code GridapPETSc.PETSC.KSPSetUp(ns.ksp[])
  ns
end

function Algebra.solve!(x::AbstractVector{PetscScalar},ns::ElasticityNumericalSetup,b::AbstractVector{PetscScalar})
  X, B = ns.X, ns.B
  copy!(B,b)
  @check_error_code GridapPETSc.PETSC.KSPSolve(ns.ksp[],B.vec[],X.vec[])
  copy!(x,X)
  return x
end

# MUMPS solver

function mumps_ksp_setup(ksp)
  pc       = Ref{GridapPETSc.PETSC.PC}()
  mumpsmat = Ref{GridapPETSc.PETSC.Mat}()
  @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPPREONLY)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCLU)
  @check_error_code GridapPETSc.PETSC.PCFactorSetMatSolverType(pc[],GridapPETSc.PETSC.MATSOLVERMUMPS)
  @check_error_code GridapPETSc.PETSC.PCFactorSetUpMatSolverType(pc[])
  @check_error_code GridapPETSc.PETSC.PCFactorGetMatrix(pc[],mumpsmat)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  4, 1) # Level of printing
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  7, 0) # Perm type
# @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 14, 1000)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 28, 2) # Seq or parallel analysis
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 2) # Parallel ordering
  @check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[], 3, 1.0e-6) # Absolute pivoting threshold
end

function MUMPSSolver()
  return PETScLinearSolver(mumps_ksp_setup)
end

# Block diagonal preconditioner

struct BlockDiagonalPreconditioner{N,A} <: Gridap.Algebra.LinearSolver
  solvers :: A
  function BlockDiagonalPreconditioner(solvers::AbstractArray{<:Gridap.Algebra.LinearSolver})
    N = length(solvers)
    A = typeof(solvers)
    return new{N,A}(solvers)
  end
end

struct BlockDiagonalPreconditionerSS{A,B} <: Gridap.Algebra.SymbolicSetup
  solver   :: A
  block_ss :: B
end

function Gridap.Algebra.symbolic_setup(solver::BlockDiagonalPreconditioner,mat::AbstractBlockMatrix)
  mat_blocks = diag(blocks(mat))
  block_ss   = map(symbolic_setup,solver.solvers,mat_blocks)
  return BlockDiagonalPreconditionerSS(solver,block_ss)
end

struct BlockDiagonalPreconditionerNS{A,B} <: Gridap.Algebra.NumericalSetup
  solver   :: A
  block_ns :: B
end

function Gridap.Algebra.numerical_setup(ss::BlockDiagonalPreconditionerSS,mat::AbstractBlockMatrix)
  solver     = ss.solver
  mat_blocks = diag(blocks(mat))
  block_ns   = map(numerical_setup,ss.block_ss,mat_blocks)
  return BlockDiagonalPreconditionerNS(solver,block_ns)
end

function Gridap.Algebra.numerical_setup!(ns::BlockDiagonalPreconditionerNS,mat::AbstractBlockMatrix)
  mat_blocks = diag(blocks(mat))
  map(numerical_setup!,ns.block_ns,mat_blocks)
end

function Gridap.Algebra.solve!(x::AbstractBlockVector,ns::BlockDiagonalPreconditionerNS,b::AbstractBlockVector)
  @check blocklength(x) == blocklength(b) == length(ns.block_ns)
  for (iB,bns) in enumerate(ns.block_ns)
    xi = x[Block(iB)]
    bi = b[Block(iB)]
    solve!(xi,bns,bi)
  end
  return x
end

function LinearAlgebra.ldiv!(x,ns::BlockDiagonalPreconditionerNS,b)
  solve!(x,ns,b)
end
