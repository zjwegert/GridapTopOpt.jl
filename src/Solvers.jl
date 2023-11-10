
struct ElasticitySolver{A,B} <: Gridap.Algebra.LinearSolver
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

struct ElasticitySymbolicSetup{A} <: Gridap.Algebra.SymbolicSetup
  solver::A
end

function Gridap.Algebra.symbolic_setup(solver::ElasticitySolver,A::AbstractMatrix)
  ElasticitySymbolicSetup(solver)
end

function get_dof_coords(trian,space)
  coords = map(local_views(trian),local_views(space),partition(space.gids)) do trian, space, dof_indices
    node_coords = Gridap.Geometry.get_node_coordinates(trian)
    dof_to_node = space.metadata.free_dof_to_node
    dof_to_comp = space.metadata.free_dof_to_comp

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
