"""
    struct AffineFEStateMap{A,B,C,D,E,F} <: AbstractFEStateMap

A structure to enable the forward problem and pullback for affine finite
element operators `AffineFEOperator`.

# Parameters

- `biform::A`: `Function` defining the bilinear form.
- `liform::B`: `Function` defining the linear form.
- `spaces::C`: `Tuple` of finite element spaces.
- `plb_caches::D`: A cache for the pullback operator.
- `fwd_caches::E`: A cache for the forward problem.
- `adj_caches::F`: A cache for the adjoint problem.
"""
struct AffineFEStateMap{A,B,C,D,E,F} <: AbstractFEStateMap
  biform     :: A
  liform     :: B
  spaces     :: C
  plb_caches :: D
  fwd_caches :: E
  adj_caches :: F

  @doc """
      AffineFEStateMap(
        a::Function,l::Function,
        U,V,V_φ,φh;
        assem_U = SparseMatrixAssembler(U,V),
        assem_adjoint = SparseMatrixAssembler(V,U),
        assem_deriv = SparseMatrixAssembler(V_φ,V_φ),
        ls::LinearSolver = LUSolver(),
        adjoint_ls::LinearSolver = LUSolver()
      )

  Create an instance of `AffineFEStateMap` given the bilinear form `a` and linear
  form `l` as `Function` types, trial and test spaces `U` and `V`, the FE space `V_φ`
  for `φh` and derivatives, and the measures as additional arguments.

  Optional arguments enable specification of assemblers and linear solvers.
  """
  function AffineFEStateMap(
      biform::Function,liform::Function,
      U,V,V_φ,φh;
      assem_U = SparseMatrixAssembler(U,V),
      assem_adjoint = SparseMatrixAssembler(V,U),
      assem_deriv = SparseMatrixAssembler(V_φ,V_φ),
      ls::LinearSolver = LUSolver(),
      adjoint_ls::LinearSolver = LUSolver()
    )
    # TODO: I really want to get rid of the φh argument...

    spaces = (U,V,V_φ)

    ## Pullback cache
    uhd = zero(U)
    # vecdata = collect_cell_vector(V_φ,∇(biform,[uhd,uhd,φh],3) - ∇(liform,[uhd,φh],2))
    # dudφ_vec = allocate_vector(assem_deriv,vecdata)
    dudφ_vec = get_free_dof_values(zero(V_φ))
    plb_caches = (dudφ_vec,assem_deriv)

    ## Forward cache
    op = AffineFEOperator((u,v)->biform(u,v,φh),v->liform(v,φh),U,V,assem_U)
    K, b = get_matrix(op), get_vector(op)
    x  = allocate_in_domain(K); fill!(x,zero(eltype(x)))
    ns = numerical_setup(symbolic_setup(ls,K),K)
    fwd_caches = (ns,K,b,x,uhd,assem_U,ls)

    ## Adjoint cache
    adjoint_K  = assemble_matrix((u,v)->biform(v,u,φh),assem_adjoint,V,U)
    adjoint_x  = allocate_in_domain(adjoint_K); fill!(adjoint_x,zero(eltype(adjoint_x)))
    adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
    adj_caches = (adjoint_ns,adjoint_K,adjoint_x,assem_adjoint,adjoint_ls)

    A,B,C = typeof(biform), typeof(liform), typeof(spaces)
    D,E,F = typeof(plb_caches),typeof(fwd_caches), typeof(adj_caches)
    return new{A,B,C,D,E,F}(biform,liform,spaces,plb_caches,fwd_caches,adj_caches)
  end
end

# Getters
get_state(m::AffineFEStateMap) = FEFunction(get_trial_space(m),m.fwd_caches[4])
get_spaces(m::AffineFEStateMap) = m.spaces
get_assemblers(m::AffineFEStateMap) = (m.fwd_caches[6],m.plb_caches[2],m.adj_caches[4])

function forward_solve!(φ_to_u::AffineFEStateMap,φh)
  biform, liform = φ_to_u.biform, φ_to_u.liform
  U, V, _ = φ_to_u.spaces
  ns, K, b, x, uhd, assem_U, _ = φ_to_u.fwd_caches

  a_fwd(u,v) = biform(u,v,φh)
  l_fwd(v)   = liform(v,φh)
  assemble_matrix_and_vector!(a_fwd,l_fwd,K,b,assem_U,U,V,uhd)
  numerical_setup!(ns,K)
  solve!(x,ns,b)
  return x
end

function forward_solve!(φ_to_u::AffineFEStateMap,φ::AbstractVector)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return forward_solve!(φ_to_u,φh)
end

function dRdφ(φ_to_u::AffineFEStateMap,uh,vh,φh)
  biform, liform = φ_to_u.biform, φ_to_u.liform
  return ∇(biform,[uh,vh,φh],3) - ∇(liform,[vh,φh],2)
end

function update_adjoint_caches!(φ_to_u::AffineFEStateMap,uh,φh)
  adjoint_ns, adjoint_K, _, assem_adjoint, _ = φ_to_u.adj_caches
  U, V, _ = φ_to_u.spaces
  assemble_matrix!((u,v) -> φ_to_u.biform(v,u,φh),adjoint_K,assem_adjoint,V,U)
  numerical_setup!(adjoint_ns,adjoint_K)
  return φ_to_u.adj_caches
end

function adjoint_solve!(φ_to_u::AffineFEStateMap,du::AbstractVector)
  adjoint_ns, _, adjoint_x, _, _ = φ_to_u.adj_caches
  solve!(adjoint_x,adjoint_ns,du)
  return adjoint_x
end