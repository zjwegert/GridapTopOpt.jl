struct StaggeredAdjointAffineFEOperator{NB,SB} <: StaggeredFEOperator{NB,SB}
  biforms :: Vector{<:Function}
  liforms :: Vector{<:Function}
  trials  :: Vector{<:FESpace}
  tests   :: Vector{<:FESpace}
  assems  :: Vector{<:Assembler}
  trial   :: GridapSolvers.BlockSolvers.BlockFESpaceTypes{NB,SB}
  test    :: GridapSolvers.BlockSolvers.BlockFESpaceTypes{NB,SB}

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

  Constructor for a `StaggeredAffineFEOperator` operator, taking in each
  equation as a pair of bilinear/linear forms and the global trial/test spaces.
  """
  function StaggeredAdjointAffineFEOperator(
    biforms :: Vector{<:Function},
    liforms :: Vector{<:Function},
    trial   :: GridapSolvers.BlockSolvers.BlockFESpaceTypes{NB,SB,P},
    test    :: GridapSolvers.BlockSolvers.BlockFESpaceTypes{NB,SB,P},
    assem   :: GridapSolvers.BlockSolvers.BlockSparseMatrixAssembler{NB,NV,SB,P} = SparseMatrixAssembler(trial,test)
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
  A = GridapTopOpt.assemble_adjoint_matrix(a,op.assems[k],op.trials[k],op.tests[k])
  b = assemble_vector(l,op.assems[k],op.tests[k])
  return AffineFEOperator(op.trials[k],op.tests[k],AffineOperator(A,b))
end

function GridapSolvers.BlockSolvers.get_operator!(op_k::AffineFEOperator, op::StaggeredAdjointAffineFEOperator{NB}, xhs, k) where NB
  @assert NB >= k
  A, b = get_matrix(op_k), get_vector(op_k)
  a(uk,vk) = op.biforms[k](xhs,uk,vk)
  l(vk) = op.liforms[k](xhs,vk)
  assemble_matrix_and_vector!(a,l,A,b,op.assems[k],op.trials[k],op.tests[k],zero(op.tests[k]))
  return op_k
end

# Not currently required
function Algebra.solve!(xh, solver::StaggeredFESolver{NB}, op::StaggeredFEOperator{NB},l0::AbstractVector,::Nothing) where NB
  solvers = solver.solvers
  xhs, caches, operators = (), (), ()
  for k in 1:NB
    xh_k = get_solution(op,xh,k)
    op_k = get_operator(op,xhs,k)
    l = get_vector(op_k); l .+= l0
    xh_k, cache_k = solve!(xh_k,solvers[k],op_k,nothing)
    xhs, caches, operators = (xhs...,xh_k), (caches...,cache_k), (operators...,op_k)
  end
  return xh, (caches,operators)
end

function Algebra.solve!(xh, solver::StaggeredFESolver{NB}, op::StaggeredFEOperator{NB}, l0::AbstractVector,cache) where NB
  last_caches, last_operators = cache
  solvers = solver.solvers
  xhs, caches, operators = (), (), ()
  for k in 1:NB
    xh_k = get_solution(op,xh,k)
    op_k = get_operator!(last_operators[k],op,xhs,k)
    l = get_vector(op_k); l .+= l0
    xh_k, cache_k = solve!(xh_k,solvers[k],op_k,last_caches[k])
    xhs, caches, operators = (xhs...,xh_k), (caches...,cache_k), (operators...,op_k)
  end
  return xh, (caches,operators)
end
