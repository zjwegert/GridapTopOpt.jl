"""
    struct RepeatingAffineFEStateMap <: AbstractFEStateMap

A structure to enable the forward problem and pullback for affine finite
element operators `AffineFEOperator` with multiple linear forms but only
a single bilinear form.

# Parameters

- `biform`: A `Function` defining the bilinear form.
- `liform`: A vector of `Function` defining the linear forms.
- `spaces`: Repeated finite element spaces.
- `spaces_0`: Original finite element spaces that are being repeated.
- `plb_caches`: A cache for the pullback operator.
- `fwd_caches`: A cache for the forward problem.
- `adj_caches`: A cache for the adjoint problem.
"""
struct RepeatingAffineFEStateMap{A,B,C,D,E,F,G} <: AbstractFEStateMap
  biform     :: A
  liform     :: B
  spaces     :: C
  spaces_0   :: D
  plb_caches :: E
  fwd_caches :: F
  adj_caches :: G

  @doc """
      RepeatingAffineFEStateMap(
        nblocks::Int,biform::Function,liforms::Vector{<:Function},
        U0,V0,V_φ,φh;
        assem_U = SparseMatrixAssembler(U0,V0),
        assem_adjoint = SparseMatrixAssembler(V0,U0),
        assem_deriv = SparseMatrixAssembler(V_φ,V_φ),
        ls::LinearSolver = LUSolver(),
        adjoint_ls::LinearSolver = LUSolver()
      )

  Create an instance of `RepeatingAffineFEStateMap` given the number of blocks `nblocks`,
  a bilinear form `a`, a vector of linear form `l` as `Function` types, the trial and test
  spaces `U` and `V`, the FE space `V_φ` for `φh` and derivatives,
  and the measures as additional arguments.

  Optional arguments enable specification of assemblers and linear solvers.

  # Note

  - The resulting `FEFunction` will be a `MultiFieldFEFunction` (or GridapDistributed equivalent)
    where each field corresponds to an entry in the vector of linear forms
  """
  function RepeatingAffineFEStateMap(
    nblocks::Int,biform::Function,liforms::Vector{<:Function},
    U0,V0,V_φ,φh;
    assem_U = SparseMatrixAssembler(U0,V0),
    assem_adjoint = SparseMatrixAssembler(V0,U0),
    assem_deriv = SparseMatrixAssembler(V_φ,V_φ),
    ls::LinearSolver = LUSolver(),
    adjoint_ls::LinearSolver = LUSolver())
    @check nblocks == length(liforms)

    spaces_0 = (U0,V0)
    assem_U0 = assem_U

    U, V = repeat_spaces(nblocks,U0,V0)
    spaces = (U,V,V_φ)
    assem_U = SparseMatrixAssembler(
      get_local_matrix_type(assem_U0), get_local_vector_type(assem_U0),
      U, V, get_local_assembly_strategy(assem_U0)
    )

    ## Pullback cache
    uhd = zero(U0)
    dudφ_vec = get_free_dof_values(zero(V_φ))
    plb_caches = (dudφ_vec,assem_deriv)

    ## Forward cache
    K  = assemble_matrix((u,v) -> biform(u,v,φh),assem_U0,U0,V0)
    b  = allocate_in_range(K); fill!(b,zero(eltype(b)))
    b0 = allocate_in_range(K); fill!(b0,zero(eltype(b0)))
    x  = repeated_allocate_in_domain(nblocks,K); fill!(x,zero(eltype(x)))
    ns = numerical_setup(symbolic_setup(ls,K),K)
    fwd_caches = (ns,K,b,x,uhd,assem_U0,b0,assem_U,ls)

    ## Adjoint cache
    adjoint_K  = assemble_matrix((u,v)->biform(v,u,φh),assem_adjoint,V0,U0)
    adjoint_x  = repeated_allocate_in_domain(nblocks,adjoint_K); fill!(adjoint_x,zero(eltype(adjoint_x)))
    adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
    adj_caches = (adjoint_ns,adjoint_K,adjoint_x,assem_adjoint,adjoint_ls)

    A,B,C,D = typeof(biform), typeof(liforms), typeof(spaces), typeof(spaces_0)
    E,F,G = typeof(plb_caches), typeof(fwd_caches), typeof(adj_caches)
    return new{A,B,C,D,E,F,G}(biform,liforms,spaces,spaces_0,plb_caches,fwd_caches,adj_caches)
  end
end

const MultiFieldSpaceTypes = Union{<:MultiField.MultiFieldFESpace,<:GridapDistributed.DistributedMultiFieldFESpace}

function repeat_spaces(nblocks::Integer,U0::FESpace,V0::FESpace)
  U = MultiFieldFESpace([U0 for i in 1:nblocks];style=BlockMultiFieldStyle())
  V = MultiFieldFESpace([V0 for i in 1:nblocks];style=BlockMultiFieldStyle())
  return U,V
end

function repeat_spaces(nblocks::Integer,U0::MultiFieldSpaceTypes,V0::MultiFieldSpaceTypes)
  nfields = num_fields(U0)
  @assert nfields == num_fields(V0)
  @assert MultiFieldStyle(U0) === MultiFieldStyle(V0)

  mfs = repeat_mfs(nblocks,nfields,MultiFieldStyle(U0))
  U = MultiFieldFESpace(repeat([U0...],nblocks);style=mfs)
  V = MultiFieldFESpace(repeat([V0...],nblocks);style=mfs)
  return U,V
end

function repeat_mfs(nblocks::Integer,nfields::Integer,::ConsecutiveMultiFieldStyle)
  return BlockMultiFieldStyle(nblocks,Tuple(fill(nfields,nblocks)))
end

function repeat_mfs(nblocks::Integer,nfields::Integer,::BlockMultiFieldStyle{NB,SB,P}) where {NB,SB,P}
  @assert length(P) == nfields
  P_new = [P...]
  for iB in 2:nblocks
    P_new = vcat(P_new,(P .+ (iB-1)*nfields)...)
  end
  return BlockMultiFieldStyle(NB*nblocks, Tuple(repeat([SB...],nblocks)), Tuple(P_new))
end

repeated_blocks(V0::FESpace,x::AbstractBlockVector) = blocks(x)
repeated_blocks(V0::FESpace,xh) = xh

repeated_blocks(V0::MultiFieldSpaceTypes,x::AbstractBlockVector) = repeated_blocks(MultiFieldStyle(V0),V0,x)
repeated_blocks(::ConsecutiveMultiFieldStyle,V0,x::AbstractBlockVector) = blocks(x)

function repeated_blocks(::BlockMultiFieldStyle{NB},V0::MultiFieldSpaceTypes,x::AbstractBlockVector) where NB
  xb = blocks(x)
  @assert length(xb) % NB == 0

  nblocks = length(xb) ÷ NB
  rep_blocks = map(1:nblocks) do iB
    mortar(xb[(iB-1)*NB+1:iB*NB])
  end
  return rep_blocks
end

function repeated_blocks(V0::MultiFieldSpaceTypes,xh)
  x_blocks = repeated_blocks(MultiFieldStyle(V0),V0,get_free_dof_values(xh))
  rep_blocks = map(x_blocks) do xb
    FEFunction(V0,xb)
  end
  return rep_blocks
end

function repeated_allocate_in_domain(nblocks::Integer,M::AbstractMatrix)
  mortar(map(i -> allocate_in_domain(M), 1:nblocks))
end

function repeated_allocate_in_domain(nblocks::Integer,M::AbstractBlockMatrix)
  mortar(vcat(map(i -> blocks(allocate_in_domain(M)), 1:nblocks)...))
end

get_state(m::RepeatingAffineFEStateMap) = FEFunction(get_trial_space(m),m.fwd_caches[4])
get_spaces(m::RepeatingAffineFEStateMap) = m.spaces
get_assemblers(m::RepeatingAffineFEStateMap) = (m.fwd_caches[8],m.plb_caches[2],m.adj_caches[4])

function forward_solve!(φ_to_u::RepeatingAffineFEStateMap,φh)
  biform, liforms = φ_to_u.biform, φ_to_u.liform
  U0, V0 = φ_to_u.spaces_0
  ns, K, b, x, uhd, assem_U0, b0, _, _ = φ_to_u.fwd_caches

  a_fwd(u,v) = biform(u,v,φh)
  assemble_matrix!(a_fwd,K,assem_U0,U0,V0)
  numerical_setup!(ns,K)

  l0_fwd(v) = a_fwd(uhd,v)
  assemble_vector!(l0_fwd,b0,assem_U0,V0)
  rmul!(b0,-1)

  v = get_fe_basis(V0)
  map(repeated_blocks(U0,x),liforms) do xi, li
    copy!(b,b0)
    vecdata = collect_cell_vector(V0,li(v,φh))
    assemble_vector_add!(b,assem_U0,vecdata)
    solve!(xi,ns,b)
  end
  return x
end

function forward_solve!(φ_to_u::RepeatingAffineFEStateMap,φ::AbstractVector)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return forward_solve!(φ_to_u,φh)
end

function dRdφ(φ_to_u::RepeatingAffineFEStateMap,uh,vh,φh)
  biform, liforms = φ_to_u.biform, φ_to_u.liform
  U0, V0 = φ_to_u.spaces_0

  uh_blocks = repeated_blocks(U0,uh)
  vh_blocks = repeated_blocks(V0,vh)
  res = DomainContribution()
  for (liform,uhi,vhi) in zip(liforms,uh_blocks,vh_blocks)
    res = res + ∇(biform,[uhi,vhi,φh],3) - ∇(liform,[vhi,φh],2)
  end
  return res
end

function update_adjoint_caches!(φ_to_u::RepeatingAffineFEStateMap,uh,φh)
  adjoint_ns, adjoint_K, _, assem_adjoint, _ = φ_to_u.adj_caches
  U0, V0 = φ_to_u.spaces_0
  assemble_matrix!((u,v) -> φ_to_u.biform(v,u,φh),adjoint_K,assem_adjoint,V0,U0)
  numerical_setup!(adjoint_ns,adjoint_K)
  return φ_to_u.adj_caches
end

function adjoint_solve!(φ_to_u::RepeatingAffineFEStateMap,du::AbstractBlockVector)
  adjoint_ns, _, adjoint_x, _, _ = φ_to_u.adj_caches
  U0, V0 = φ_to_u.spaces_0
  map(repeated_blocks(U0,adjoint_x),repeated_blocks(U0,du)) do xi, dui
    solve!(xi,adjoint_ns,dui)
  end
  return adjoint_x
end

## Backwards compat
function RepeatingAffineFEStateMap(nblocks::Int,biform::Function,liforms::Vector{<:Function},
    U0,V0,V_φ,U_reg,φh;kwargs...)
  @warn _msg_v0_3_0 maxlog=1
  return RepeatingAffineFEStateMap(nblocks,biform,liforms,U0,V0,V_φ,φh;kwargs...)
end