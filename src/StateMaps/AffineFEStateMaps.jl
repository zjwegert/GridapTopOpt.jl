"""
    struct AffineFEStateMap{A,B,C,D,E,F} <: AbstractFEStateMap

A structure to enable the forward problem and pullback for affine finite
element operators `AffineFEOperator`.

# Parameters

- `biform::A`: `Function` defining the bilinear form.
- `liform::B`: `Function` defining the linear form.
- `spaces::C`: `Tuple` of finite element spaces.
- `assems::D`: `Tuple` of assemblers
- `cache::E`: An AffineFEStateMapCache
"""
struct AffineFEStateMap{A,B,C,D,E} <: AbstractFEStateMap
  biform     :: A
  liform     :: B
  spaces     :: C
  assems     :: D
  cache      :: E

  @doc """
      AffineFEStateMap(
        a::Function,l::Function,U,V,V_φ;
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
      biform::Function,liform::Function,U,V,V_φ;
      assem_U = SparseMatrixAssembler(U,V),
      assem_adjoint = SparseMatrixAssembler(V,U),
      assem_deriv = SparseMatrixAssembler(V_φ,V_φ),
      ls::LinearSolver = LUSolver(),
      adjoint_ls::LinearSolver = LUSolver()
    )
    spaces = (U,V,V_φ)
    assems = (;assem_U,assem_deriv,assem_adjoint)
    cache = FEStateMapCache(ls,adjoint_ls)
    A,B,C,D,E = typeof(biform),typeof(liform),typeof(spaces),typeof(assems),typeof(cache)
    return new{A,B,C,D,E}(biform,liform,spaces,assems,cache)
  end
end

# Caching
function build_cache!(state_map::AffineFEStateMap,φh)
  assem_U, assem_deriv, assem_adjoint = state_map.assems
  U,V,V_φ = state_map.spaces
  biform, liform = state_map.biform, state_map.liform
  cache = state_map.cache
  ls, adjoint_ls = cache.solvers[1], cache.solvers[2]

  ## Pullback cache
  dudφ_vec = get_free_dof_values(zero(V_φ))
  cache.plb_cache = (dudφ_vec,assem_deriv)

  ## Forward cache
  uhd = zero(U)
  op = AffineFEOperator((u,v)->biform(u,v,φh),v->liform(v,φh),U,V,assem_U)
  K, b = get_matrix(op), get_vector(op)
  x  = allocate_in_domain(K); fill!(x,zero(eltype(x)))
  ns = numerical_setup(symbolic_setup(ls,K),K)
  cache.fwd_cache = (ns,K,b,x,uhd)

  ## Adjoint cache
  adjoint_K  = assemble_matrix((u,v)->biform(v,u,φh),assem_adjoint,V,U)
  adjoint_x  = allocate_in_domain(adjoint_K); fill!(adjoint_x,zero(eltype(adjoint_x)))
  adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
  cache.adj_cache = (adjoint_ns,adjoint_K,adjoint_x)

  ## Update cache status
  cache.cache_built = true

  return cache
end

# Getters
function get_state(m::AffineFEStateMap)
  @assert is_cache_built(m.cache) """
    You must build the cache before using get_state. This can be achieved by either
    solving your problem with my_state_map(φh) or by running build_cache!(my_state_map,φh)
  """
  FEFunction(get_trial_space(m),m.cache.fwd_cache[4])
end
get_plb_cache(m::AffineFEStateMap) = m.cache.plb_cache
get_spaces(m::AffineFEStateMap) = m.spaces
get_assemblers(m::AffineFEStateMap) = m.assems

function forward_solve!(φ_to_u::AffineFEStateMap,φh)
  biform, liform = φ_to_u.biform, φ_to_u.liform
  U, V, _ = φ_to_u.spaces
  assem_U = φ_to_u.assems.assem_U
  if !is_cache_built(φ_to_u.cache)
    build_cache!(φ_to_u,φh)
  end
  ns, K, b, x, uhd = φ_to_u.cache.fwd_cache

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
  if !is_cache_built(φ_to_u.cache)
    build_cache!(φ_to_u,φh)
  end
  assem_adjoint = φ_to_u.assems.assem_adjoint
  adjoint_ns, adjoint_K, _ = φ_to_u.cache.adj_cache
  U, V, _ = φ_to_u.spaces
  assemble_matrix!((u,v) -> φ_to_u.biform(v,u,φh),adjoint_K,assem_adjoint,V,U)
  numerical_setup!(adjoint_ns,adjoint_K)
  return φ_to_u.cache.adj_cache
end

function adjoint_solve!(φ_to_u::AffineFEStateMap,du::AbstractVector)
  adjoint_ns, _, adjoint_x = φ_to_u.cache.adj_cache
  solve!(adjoint_x,adjoint_ns,du)
  return adjoint_x
end

## Backwards compat
function AffineFEStateMap(biform::Function,liform::Function,U,V,V_φ,U_reg,φh; kwargs...)
  error(_msg_v0_3_0(AffineFEStateMap))
end
function AffineFEStateMap(biform::Function,liform::Function,U,V,V_φ,φh; kwargs...)
  error(_msg_v0_4_0(AffineFEStateMap))
end