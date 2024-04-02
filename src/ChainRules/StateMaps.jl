"""
    abstract type AbstractFEStateMap

Types inheriting from this abstract type should enable the evaluation and differentiation of 
the solution to an FE problem `u` that implicitly depends on an auxiliary parameter `φ`.
"""
abstract type AbstractFEStateMap end

get_state(::AbstractFEStateMap) = @abstractmethod
get_measure(::AbstractFEStateMap) = @abstractmethod
get_spaces(::AbstractFEStateMap) = @abstractmethod
get_assemblers(::AbstractFEStateMap) = @abstractmethod

get_trial_space(m::AbstractFEStateMap) = get_spaces(m)[1]
get_test_space(m::AbstractFEStateMap) = get_spaces(m)[2]
get_aux_space(m::AbstractFEStateMap) = get_spaces(m)[3]
get_pde_assembler(m::AbstractFEStateMap) = get_assemblers(m)[1]
get_aux_assembler(m::AbstractFEStateMap) = get_assemblers(m)[2]


@inline (φ_to_u::AbstractFEStateMap)(φh) = forward_solve!(φ_to_u,φh)

"""
    forward_solve!(φ_to_u::AbstractFEStateMap,φh)

Evaluate the forward problem `u` given `φ`. This should compute the
FE problem.
"""
function forward_solve!(φ_to_u::AbstractFEStateMap,φh)
  @abstractmethod
end

function forward_solve!(φ_to_u::AbstractFEStateMap,φ::AbstractVector)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return forward_solve!(φ_to_u,φh)
end

"""
    update_adjoint_caches!(φ_to_u::AbstractFEStateMap,uh,φh)

Update the cache for the adjoint problem. This is usually a tuple
of objects.
"""
function update_adjoint_caches!(φ_to_u::AbstractFEStateMap,uh,φh)
  @abstractmethod
end

function update_adjoint_caches!(φ_to_u::AbstractFEStateMap,u::AbstractVector,φ::AbstractVector)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return update_adjoint_caches!(φ_to_u,uh,φh)
end

"""
    adjoint_solve!(φ_to_u::AbstractFEStateMap,du::AbstractVector)

Evaluate the solution to the adjoint problem given a RHS vector `∂F∂u` denoted `du`.
This should solve the linear problem `dRduᵀ*λ = ∂F∂uᵀ`.
"""
function adjoint_solve!(φ_to_u::AbstractFEStateMap,du::AbstractVector)
  @abstractmethod
end 

"""
    dRdφ(φ_to_u::AbstractFEStateMap,uh,vh,φh)

Compute the derivative with respect to `φh` of the residual R.
"""
function dRdφ(φ_to_u::AbstractFEStateMap,uh,vh,φh)
  @abstractmethod
end

function dRdφ(φ_to_u::AbstractFEStateMap,u::AbstractVector,v::AbstractVector,φ::AbstractVector)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  vh = FEFunction(get_test_space(φ_to_u),v)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return dRdφ(φ_to_u,uh,vh,φh)
end

"""
    pullback(φ_to_u::AbstractFEStateMap,uh,φh,du;updated)

Compute `∂F∂u*dudφ` at `φh` and `uh` using the adjoint method. I.e., let 

`∂F∂u*dudφ = -λᵀ*dRdφ`

and solve the adjoint problem `dRduᵀ*λ = ∂F∂uᵀ` using [`adjoint_solve!`](@ref).
"""
function pullback(φ_to_u::AbstractFEStateMap,uh,φh,du;updated=false)
  ## Adjoint Solve
  if !updated
    update_adjoint_caches!(φ_to_u,uh,φh)
  end
  λ  = adjoint_solve!(φ_to_u,du)
  rmul!(λ,-1)
  
  λh = FEFunction(get_test_space(φ_to_u),λ)
  return dRdφ(φ_to_u,uh,λh,φh)
end

function pullback(φ_to_u::AbstractFEStateMap,u::AbstractVector,φ::AbstractVector,du::AbstractVector;updated=false)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return pullback(φ_to_u,uh,φh,du;updated=updated)
end

function rrule_pullback(φ_to_u::AbstractFEStateMap,uh,φh,du;updated=false)
  dudφ_vec, assem_φ = φ_to_u.plb_caches
  V_φ = get_aux_space(φ_to_u)

  ## Compute grad
  dudφ = pullback(φ_to_u,uh,φh,du;updated=updated)
  dudφ_vecdata = collect_cell_vector(V_φ,dudφ) 
  assemble_vector!(dudφ_vec,assem_φ,dudφ_vecdata)
  
  return (NoTangent(),dudφ_vec)
end

"""
    rrule(φ_to_u::AbstractFEStateMap,φh)

Return the evaluation of a `AbstractFEStateMap` and a 
a function for evaluating the pullback of `φ_to_u`. This enables
compatiblity with `ChainRules.jl`
"""
function ChainRulesCore.rrule(φ_to_u::AbstractFEStateMap,φh)
  u  = forward_solve!(φ_to_u,φh)
  uh = FEFunction(get_trial_space(φ_to_u),u)
  update_adjoint_caches!(φ_to_u,uh,φh)
  return u, du -> rrule_pullback(φ_to_u,uh,φh,du;updated=true)
end

function ChainRulesCore.rrule(φ_to_u::AbstractFEStateMap,φ::AbstractVector)
  φh = FEFunction(get_aux_space(φ_to_u),φ)
  return ChainRulesCore.rrule(φ_to_u,φh)
end

"""
    struct AffineFEStateMap{A,B,C,D,E,F} <: AbstractFEStateMap

A structure to enable the forward problem and pullback for affine finite
element operators `AffineFEOperator`.

# Parameters

- `biform::A`: `IntegrandWithMeasure` defining the bilinear form.
- `liform::B`: `IntegrandWithMeasure` defining the linear form.
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
        U,V,V_φ,φh,dΩ...;
        assem_U = SparseMatrixAssembler(U,V),
        assem_adjoint = SparseMatrixAssembler(V,U),
        assem_φ = SparseMatrixAssembler(V_φ,V_φ),
        ls::LinearSolver = LUSolver(),
        adjoint_ls::LinearSolver = LUSolver()
      )

  Create an instance of `AffineFEStateMap` given the bilinear form `a` and linear
  form `l` as `Function` types, trial and test spaces `U` and `V`, the FE space `V_φ`
  for `φh` and the measures as additional arguments. 

  Optional arguments enable specification of assemblers and linear solvers.
  """
  function AffineFEStateMap(
    a::Function,l::Function,
    U,V,V_φ,φh,dΩ...;
    assem_U = SparseMatrixAssembler(U,V),
    assem_adjoint = SparseMatrixAssembler(V,U),
    assem_φ = SparseMatrixAssembler(V_φ,V_φ),
    ls::LinearSolver = LUSolver(),
    adjoint_ls::LinearSolver = LUSolver()
  )
    # TODO: I really want to get rid of the φh argument...

    biform = IntegrandWithMeasure(a,dΩ)
    liform = IntegrandWithMeasure(l,dΩ)
    spaces = (U,V,V_φ)

    ## Pullback cache
    uhd = zero(U)
    vecdata = collect_cell_vector(V_φ,∇(biform,[uhd,uhd,φh],3) - ∇(liform,[uhd,φh],2))
    dudφ_vec = allocate_vector(assem_φ,vecdata)
    plb_caches = (dudφ_vec,assem_φ)

    ## Forward cache
    op = AffineFEOperator((u,v)->biform(u,v,φh),v->liform(v,φh),U,V,assem_U)
    K, b = get_matrix(op), get_vector(op)
    x  = allocate_in_domain(K); fill!(x,zero(eltype(x)))
    ns = numerical_setup(symbolic_setup(ls,K),K)
    fwd_caches = (ns,K,b,x,uhd,assem_U)

    ## Adjoint cache
    adjoint_K  = assemble_matrix((u,v)->biform(v,u,φh),assem_adjoint,V,U)
    adjoint_x  = allocate_in_domain(adjoint_K); fill!(adjoint_x,zero(eltype(adjoint_x)))
    adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
    adj_caches = (adjoint_ns,adjoint_K,adjoint_x,assem_adjoint)

    A,B,C = typeof(biform), typeof(liform), typeof(spaces)
    D,E,F = typeof(plb_caches),typeof(fwd_caches), typeof(adj_caches)
    return new{A,B,C,D,E,F}(biform,liform,spaces,plb_caches,fwd_caches,adj_caches)
  end
end

# Getters
get_state(m::AffineFEStateMap) = FEFunction(get_trial_space(m),m.fwd_caches[4])
get_measure(m::AffineFEStateMap) = m.biform.dΩ
get_spaces(m::AffineFEStateMap) = m.spaces
get_assemblers(m::AffineFEStateMap) = (m.fwd_caches[6],m.plb_caches[2],m.adj_caches[4])

function forward_solve!(φ_to_u::AffineFEStateMap,φh)
  biform, liform = φ_to_u.biform, φ_to_u.liform
  U, V, _ = φ_to_u.spaces
  ns, K, b, x, uhd, assem_U = φ_to_u.fwd_caches

  a_fwd(u,v) = biform(u,v,φh)
  l_fwd(v)   = liform(v,φh)
  assemble_matrix_and_vector!(a_fwd,l_fwd,K,b,assem_U,U,V,uhd)
  numerical_setup!(ns,K)
  solve!(x,ns,b)
  return x
end

function dRdφ(φ_to_u::AffineFEStateMap,uh,vh,φh)
  biform, liform = φ_to_u.biform, φ_to_u.liform
  return ∇(biform,[uh,vh,φh],3) - ∇(liform,[vh,φh],2)
end

function update_adjoint_caches!(φ_to_u::AffineFEStateMap,uh,φh)
  adjoint_ns, adjoint_K, _, assem_adjoint = φ_to_u.adj_caches
  U, V, _ = φ_to_u.spaces
  assemble_matrix!((u,v) -> φ_to_u.biform(v,u,φh),adjoint_K,assem_adjoint,V,U) # TODO: @Jordi, should this be `assemble_adjoint_matrix!`?
  numerical_setup!(adjoint_ns,adjoint_K)
  return φ_to_u.adj_caches
end

function adjoint_solve!(φ_to_u::AffineFEStateMap,du::AbstractVector)
  adjoint_ns, _, adjoint_x, _ = φ_to_u.adj_caches
  solve!(adjoint_x,adjoint_ns,du)
  return adjoint_x
end

"""
    struct NonlinearFEStateMap{A,B,C,D,E,F} <: AbstractFEStateMap

A structure to enable the forward problem and pullback for nonlinear finite
element operators.

# Parameters

- `res::A`: an `IntegrandWithMeasure` defining the residual of the problem.
- `jac::B`: a `Function` defining Jacobian of the residual.
- `spaces::C`: `Tuple` of finite element spaces.
- `plb_caches::D`: A cache for the pullback operator.
- `fwd_caches::E`: A cache for the forward problem.
- `adj_caches::F`: A cache for the adjoint problem.
"""
struct NonlinearFEStateMap{A,B,C,D,E,F} <: AbstractFEStateMap
  res        :: A
  jac        :: B
  spaces     :: C
  plb_caches :: D
  fwd_caches :: E
  adj_caches :: F

  @doc """
      NonlinearFEStateMap(
        res::Function,U,V,V_φ,φh,dΩ...;
        assem_U = SparseMatrixAssembler(U,V),
        assem_adjoint = SparseMatrixAssembler(V,U),
        assem_φ = SparseMatrixAssembler(V_φ,V_φ),
        nls::NonlinearSolver = NewtonSolver(LUSolver();maxiter=50,rtol=1.e-8,verbose=true),
        adjoint_ls::LinearSolver = LUSolver()
      )

  Create an instance of `NonlinearFEStateMap` given the residual `res` as a `Function` type, 
  trial and test spaces `U` and `V`, the FE space `V_φ` for `φh` and the measures as additional arguments. 

  Optional arguments enable specification of assemblers, nonlinear solver, and adjoint (linear) solver.
  """
  function NonlinearFEStateMap(
    res::Function,U,V,V_φ,φh,dΩ...;
    assem_U = SparseMatrixAssembler(U,V),
    assem_adjoint = SparseMatrixAssembler(V,U),
    assem_φ = SparseMatrixAssembler(V_φ,V_φ),
    nls::NonlinearSolver = NewtonSolver(LUSolver();maxiter=50,rtol=1.e-8,verbose=true),
    adjoint_ls::LinearSolver = LUSolver()
  )
    res = IntegrandWithMeasure(res,dΩ)
    jac = (u,du,dv,φh) -> jacobian(res,[u,dv,φh],1)
    spaces = (U,V,V_φ)

    ## Pullback cache
    uhd = zero(U)
    vecdata = collect_cell_vector(V_φ,∇(res,[uhd,uhd,φh],3))
    dudφ_vec = allocate_vector(assem_φ,vecdata)
    plb_caches = (dudφ_vec,assem_φ)

    ## Forward cache
    x = zero_free_values(U)
    _res(u,v) = res(u,v,φh)
    _jac(u,du,dv) = jac(u,du,dv,φh)
    op = get_algebraic_operator(FEOperator(_res,_jac,U,V,assem_U))
    nls_cache = instantiate_caches(x,nls,op)
    fwd_caches = (nls,nls_cache,x,assem_U)

    ## Adjoint cache
    _jac_adj(du,dv) = jac(uhd,du,dv,φh)
    adjoint_K  = assemble_adjoint_matrix(_jac_adj,assem_adjoint,U,V)
    adjoint_x  = allocate_in_domain(adjoint_K); fill!(adjoint_x,zero(eltype(adjoint_x)))
    adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
    adj_caches = (adjoint_ns,adjoint_K,adjoint_x,assem_adjoint)
    
    A, B, C = typeof(res), typeof(jac), typeof(spaces)
    D, E, F = typeof(plb_caches), typeof(fwd_caches), typeof(adj_caches)
    return new{A,B,C,D,E,F}(res,jac,spaces,plb_caches,fwd_caches,adj_caches)
  end
end

get_state(m::NonlinearFEStateMap) = FEFunction(get_trial_space(m),m.fwd_caches[3])
get_measure(m::NonlinearFEStateMap) = m.res.dΩ
get_spaces(m::NonlinearFEStateMap) = m.spaces
get_assemblers(m::NonlinearFEStateMap) = (m.fwd_caches[4],m.plb_caches[2],m.adj_caches[4])

function forward_solve!(φ_to_u::NonlinearFEStateMap,φh)
  U, V, _ = φ_to_u.spaces
  nls, nls_cache, x, assem_U = φ_to_u.fwd_caches

  res(u,v) = φ_to_u.res(u,v,φh)
  jac(u,du,dv) = φ_to_u.jac(u,du,dv,φh)
  op = get_algebraic_operator(FEOperator(res,jac,U,V,assem_U))
  solve!(x,nls,op,nls_cache)
  return x
end

function dRdφ(φ_to_u::NonlinearFEStateMap,uh,vh,φh)
  res = φ_to_u.res
  return ∇(res,[uh,vh,φh],3)
end

function update_adjoint_caches!(φ_to_u::NonlinearFEStateMap,uh,φh)
  adjoint_ns, adjoint_K, _, assem_adjoint = φ_to_u.adj_caches
  U, V, _ = φ_to_u.spaces
  jac(du,dv) =  φ_to_u.jac(uh,du,dv,φh)
  assemble_adjoint_matrix!(jac,adjoint_K,assem_adjoint,U,V)
  numerical_setup!(adjoint_ns,adjoint_K)
  return φ_to_u.adj_caches
end

function adjoint_solve!(φ_to_u::NonlinearFEStateMap,du::AbstractVector)
  adjoint_ns, _, adjoint_x, _ = φ_to_u.adj_caches
  solve!(adjoint_x,adjoint_ns,du)
  return adjoint_x
end

"""
    struct RepeatingAffineFEStateMap <: AbstractFEStateMap

A structure to enable the forward problem and pullback for affine finite
element operators `AffineFEOperator` with multiple linear forms but only
a single bilinear form.

# Parameters

- `biform`: `IntegrandWithMeasure` defining the bilinear form.
- `liform`: A vector of `IntegrandWithMeasure` defining the linear forms.
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
        nblocks::Int,a::Function,l::Vector{<:Function},
        U0,V0,V_φ,φh,dΩ...;
        assem_U = SparseMatrixAssembler(U0,V0),
        assem_adjoint = SparseMatrixAssembler(V0,U0),
        assem_φ = SparseMatrixAssembler(V_φ,V_φ),
        ls::LinearSolver = LUSolver(),
        adjoint_ls::LinearSolver = LUSolver()
      )

  Create an instance of `RepeatingAffineFEStateMap` given the number of blocks `nblocks`, 
  a bilinear form `a`, a vector of linear form `l` as `Function` types, the trial and test 
  spaces `U` and `V`, the FE space `V_φ` for `φh`, 
  and the measures as additional arguments. 

  Optional arguments enable specification of assemblers and linear solvers.

  # Note

  - The resulting `FEFunction` will be a `MultiFieldFEFunction` (or GridapDistributed equivalent) 
    where each field corresponds to an entry in the vector of linear forms
  """
    function RepeatingAffineFEStateMap(
    nblocks::Int,a::Function,l::Vector{<:Function},
    U0,V0,V_φ,φh,dΩ...;
    assem_U = SparseMatrixAssembler(U0,V0),
    assem_adjoint = SparseMatrixAssembler(V0,U0),
    assem_φ = SparseMatrixAssembler(V_φ,V_φ),
    ls::LinearSolver = LUSolver(),
    adjoint_ls::LinearSolver = LUSolver()
  )
    @check nblocks == length(l)

    spaces_0 = (U0,V0)
    assem_U0 = assem_U

    biform = IntegrandWithMeasure(a,dΩ)
    liforms = map(li -> IntegrandWithMeasure(li,dΩ),l)
    U, V = repeat_spaces(nblocks,U0,V0)
    spaces = (U,V,V_φ)
    assem_U = SparseMatrixAssembler(
      get_matrix_type(assem_U0),get_vector_type(assem_U0),U,V,FESpaces.get_assembly_strategy(assem_U0)
    )

    ## Pullback cache
    uhd = zero(U0)
    contr = nblocks * ∇(biform,[uhd,uhd,φh],3)
    for liform in liforms
      contr = contr - ∇(liform,[uhd,φh],2)
    end
    dudφ_vec = allocate_vector(assem_φ,collect_cell_vector(V_φ,contr))
    plb_caches = (dudφ_vec,assem_φ)

    ## Forward cache
    K  = assemble_matrix((u,v) -> biform(u,v,φh),assem_U0,U0,V0)
    b  = allocate_in_range(K); fill!(b,zero(eltype(b)))
    b0 = allocate_in_range(K); fill!(b0,zero(eltype(b0)))
    x  = mortar(map(i -> allocate_in_domain(K), 1:nblocks)); fill!(x,zero(eltype(x)))
    ns = numerical_setup(symbolic_setup(ls,K),K)
    fwd_caches = (ns,K,b,x,uhd,assem_U0,b0,assem_U)

    ## Adjoint cache
    adjoint_K  = assemble_matrix((u,v)->biform(v,u,φh),assem_adjoint,V0,U0)
    adjoint_x  = mortar(map(i -> allocate_in_domain(adjoint_K), 1:nblocks)); fill!(adjoint_x,zero(eltype(adjoint_x)))
    adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
    adj_caches = (adjoint_ns,adjoint_K,adjoint_x,assem_adjoint)

    A,B,C,D = typeof(biform), typeof(liforms), typeof(spaces), typeof(spaces_0)
    E,F,G = typeof(plb_caches), typeof(fwd_caches), typeof(adj_caches)
    return new{A,B,C,D,E,F,G}(biform,liforms,spaces,spaces_0,plb_caches,fwd_caches,adj_caches)
  end
end

function repeat_spaces(nblocks::Integer,U0::FESpace,V0::FESpace)
  U = MultiFieldFESpace([U0 for i in 1:nblocks];style=BlockMultiFieldStyle())
  V = MultiFieldFESpace([V0 for i in 1:nblocks];style=BlockMultiFieldStyle())
  return U,V
end

function repeat_spaces(
  nblocks::Integer,U0::T,V0::T
) where T <: Union{MultiField.MultiFieldFESpace,GridapDistributed.DistributedMultiFieldFESpace}
  nfields = num_fields(U0)
  @assert nfields == num_fields(V0)
  mfs = BlockMultiFieldStyle(nblocks,Tuple(fill(nfields,nblocks)))
  U = MultiFieldFESpace(repeat([U0...],nblocks);style=mfs)
  V = MultiFieldFESpace(repeat([V0...],nblocks);style=mfs)
  return U,V
end

get_state(m::RepeatingAffineFEStateMap) = FEFunction(get_trial_space(m),m.fwd_caches[4])
get_measure(m::RepeatingAffineFEStateMap) = m.biform.dΩ
get_spaces(m::RepeatingAffineFEStateMap) = m.spaces
get_assemblers(m::RepeatingAffineFEStateMap) = (m.fwd_caches[8],m.plb_caches[2],m.adj_caches[4])

function forward_solve!(φ_to_u::RepeatingAffineFEStateMap,φh)
  biform, liforms = φ_to_u.biform, φ_to_u.liform
  U0, V0 = φ_to_u.spaces_0
  ns, K, b, x, uhd, assem_U0, b0, _ = φ_to_u.fwd_caches

  a_fwd(u,v) = biform(u,v,φh)
  assemble_matrix!(a_fwd,K,assem_U0,U0,V0)
  numerical_setup!(ns,K)

  l0_fwd(v) = a_fwd(uhd,v)
  assemble_vector!(l0_fwd,b0,assem_U0,V0)
  rmul!(b0,-1)

  v = get_fe_basis(V0)
  map(blocks(x),liforms) do xi, li
    copy!(b,b0)
    vecdata = collect_cell_vector(V0,li(v,φh))
    assemble_vector_add!(b,assem_U0,vecdata)
    solve!(xi,ns,b)
  end
  return x
end

function dRdφ(φ_to_u::RepeatingAffineFEStateMap,uh,vh,φh)
  biform, liforms = φ_to_u.biform, φ_to_u.liform

  res = DomainContribution()
  for (liform,uhi,vhi) in zip(liforms,uh,vh)
    res = res + ∇(biform,[uhi,vhi,φh],3) - ∇(liform,[vhi,φh],2)
  end
  return res
end

function update_adjoint_caches!(φ_to_u::RepeatingAffineFEStateMap,uh,φh)
  adjoint_ns, adjoint_K, _, assem_adjoint = φ_to_u.adj_caches
  U0, V0 = φ_to_u.spaces_0
  assemble_matrix!((u,v) -> φ_to_u.biform(v,u,φh),adjoint_K,assem_adjoint,V0,U0)
  numerical_setup!(adjoint_ns,adjoint_K)
  return φ_to_u.adj_caches
end

function adjoint_solve!(φ_to_u::RepeatingAffineFEStateMap,du::AbstractBlockVector)
  adjoint_ns, _, adjoint_x, _ = φ_to_u.adj_caches
  map(blocks(adjoint_x),blocks(du)) do xi, dui
    solve!(xi,adjoint_ns,dui)
  end
  return adjoint_x
end