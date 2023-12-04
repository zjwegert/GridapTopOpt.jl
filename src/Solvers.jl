struct ElasticitySolver{A,B} <: LinearSolver
  trian ::A
  space ::B
  rtol  ::PetscScalar
  maxits::PetscInt
  function ElasticitySolver(trian::DistributedTriangulation,
                            space::DistributedFESpace;
                            rtol=1.e-12,
                            maxits=100)
    A = typeof(trian)
    B = typeof(space)
    new{A,B}(trian,space,rtol,maxits)
  end
end

struct ElasticitySymbolicSetup{A} <: SymbolicSetup
  solver::A
end

function Gridap.Algebra.symbolic_setup(solver::ElasticitySolver,A::AbstractMatrix)
  ElasticitySymbolicSetup(solver)
end

function get_node_to_dof_glue(space::UnconstrainedFESpace{V,Nothing}) where V
  grid = get_triangulation(space)
  Dp = num_point_dims(grid)

  z = zero(VectorValue{Dp,get_dof_value_type(space)})
  node_to_tag = fill(Int8(0),num_nodes(grid))
  tag_to_mask = fill(Tuple(fill(false,Dp)),0)

  glue, _ = Gridap.FESpaces._generate_node_to_dof_glue_component_major(z,node_to_tag,tag_to_mask)
  return glue
end

function get_node_to_dof_glue(space::UnconstrainedFESpace{V,<:Gridap.FESpaces.NodeToDofGlue}) where V
  return space.metadata
end

function get_dof_coords(trian,space)
  coords = map(local_views(trian),local_views(space),partition(space.gids)) do trian, space, dof_indices
    node_coords = Gridap.Geometry.get_node_coordinates(trian)
    glue = get_node_to_dof_glue(space)
    dof_to_node = glue.free_dof_to_node
    dof_to_comp = glue.free_dof_to_comp

    o2l_dofs = own_to_local(dof_indices)
    coords = Vector{PetscScalar}(undef,length(o2l_dofs))
    for (i,dof) in enumerate(o2l_dofs)
      node = dof_to_node[dof]
      comp = dof_to_comp[dof]
      coords[i] = node_coords[node][comp]
    end
    return coords
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

function elasticity_ksp_setup(ksp,rtol,maxits)
  rtol = PetscScalar(rtol)
  atol = GridapPETSc.PETSC.PETSC_DEFAULT
  dtol = GridapPETSc.PETSC.PETSC_DEFAULT
  maxits = PetscInt(maxits)

  @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPGMRES)
  @check_error_code GridapPETSc.PETSC.KSPSetTolerances(ksp[], rtol, atol, dtol, maxits)

  pc = Ref{GridapPETSc.PETSC.PC}()
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCGAMG)
end

function Gridap.Algebra.numerical_setup(ss::ElasticitySymbolicSetup,A::PSparseMatrix)
  s = ss.solver; Dc = num_cell_dims(s.trian)

  # Compute  coordinates for owned dofs
  dof_coords = convert(PETScVector,get_dof_coords(s.trian,s.space))
  @check_error_code GridapPETSc.PETSC.VecSetBlockSize(dof_coords.vec[],Dc)

  # Create matrix nullspace
  B = convert(PETScMatrix,A)
  null = Ref{GridapPETSc.PETSC.MatNullSpace}()
  @check_error_code GridapPETSc.PETSC.MatNullSpaceCreateRigidBody(dof_coords.vec[],null)
  @check_error_code GridapPETSc.PETSC.MatSetNearNullSpace(B.mat[],null[])

  # Setup solver and preconditioner
  ns = GridapPETSc.PETScLinearSolverNS(A,B)
  @check_error_code GridapPETSc.PETSC.KSPCreate(B.comm,ns.ksp)
  @check_error_code GridapPETSc.PETSC.KSPSetOperators(ns.ksp[],ns.B.mat[],ns.B.mat[])
  elasticity_ksp_setup(ns.ksp,s.rtol,s.maxits)
  @check_error_code GridapPETSc.PETSC.KSPSetUp(ns.ksp[])
  GridapPETSc.Init(ns)
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