## Gridap
function Gridap.FESpaces.AffineFEOperator(
  weakform::Function,trial::FESpace,test::FESpace,assem::Assembler)
  if ! isa(test,TrialFESpace)
    @warn """\n
    You are building an AffineFEOperator with a test space of type TrialFESpace.

    This may result in unexpected behaviour.
    """ maxlog=1
  end

  u = get_trial_fe_basis(trial)
  v = get_fe_basis(test)

  uhd = zero(trial)
  matcontribs, veccontribs = weakform(u,v)
  data = collect_cell_matrix_and_vector(trial,test,matcontribs,veccontribs,uhd)
  A,b = assemble_matrix_and_vector(assem,data)
  #GC.gc()

  AffineFEOperator(trial,test,A,b)
end

function Base.one(f::FESpace)
  uh = zero(f)
  u = get_free_dof_values(uh)
  V = get_vector_type(f)
  fill!(u,one(eltype(V)))
  return uh
end

## Get solutions from vector of spaces
function GridapSolvers.BlockSolvers.get_solution(spaces::Vector{<:FESpace}, xh::MultiFieldFEFunction, k)
  r = MultiField.get_block_ranges(spaces)[k]
  if isone(length(r)) # SingleField
    xh_k = xh[r[1]]
  else # MultiField
    fv_k = blocks(get_free_dof_values(xh))[k]
    xh_k = MultiFieldFEFunction(fv_k, spaces[k], xh.single_fe_functions[r])
  end
  return xh_k
end

function MultiField.get_block_ranges(spaces::Vector{<:FESpace})
  NB = length(spaces)
  SB = Tuple(map(num_fields,spaces))
  MultiField.get_block_ranges(NB,SB,Tuple(1:sum(SB)))
end

## Get all solutions
function _get_solutions(op::StaggeredFEOperator{NB},xh) where NB
  map(i->get_solution(op,xh,i),Tuple(1:NB))
end

function _get_solutions(spaces,xh)
  map(i->get_solution(spaces,xh,i),Tuple(1:length(spaces)))
end

## Instantiation
function _instantiate_caches(xh,solver::StaggeredFESolver{NB},op::StaggeredFEOperator{NB}) where NB
  solvers = solver.solvers
  xhs, caches, operators = (), (), ()
  for k in 1:NB
    xh_k = GridapSolvers.BlockSolvers.get_solution(op,xh,k)
    x_k = get_free_dof_values(xh_k)
    op_k = GridapSolvers.BlockSolvers.get_operator(op,xhs,k)
    cache_k = GridapTopOpt.instantiate_caches(x_k,solvers[k],op_k)
    xhs, caches, operators = (xhs...,xh_k), (caches...,cache_k), (operators...,op_k)
  end
  return (caches,operators)
end

function GridapTopOpt.instantiate_caches(x,ls::LinearSolver,op::AffineFEOperator)
  numerical_setup(symbolic_setup(ls,get_matrix(op)),get_matrix(op))
end

#################
# StaggeredStateParamMap
struct StaggeredStateParamMap{A,B,C,D} <: GridapTopOpt.AbstractStateParamMap
  F       :: A
  spaces  :: B
  assems  :: C
  caches  :: D
end

function StaggeredStateParamMap(F::Function,φ_to_u::StaggeredAffineFEStateMap)
  Us = φ_to_u.spaces.trials
  V_φ = GridapTopOpt.get_aux_space(φ_to_u)
  U_reg = GridapTopOpt.get_deriv_space(φ_to_u)
  assem_deriv = GridapTopOpt.get_deriv_assembler(φ_to_u)
  assem_U = GridapTopOpt.get_pde_assembler(φ_to_u)
  StaggeredStateParamMap(F,Us,V_φ,U_reg,assem_U,assem_deriv)
end

function StaggeredStateParamMap(
  F,trials::Vector{<:FESpace},V_φ::FESpace,U_reg::FESpace,
  assem_U::Vector{<:Assembler},assem_deriv::Assembler
)
  φ₀, u₀s = interpolate(x->-sqrt((x[1]-1/2)^2+(x[2]-1/2)^2)+0.2,V_φ), zero.(trials)
  dFdxj(j,φh,xh_comb) = ∇((xj->F((xh_comb[1:j-1]...,xj,xh_comb[j+1:end]...),φh)))(xh_comb[j])

  ∂F∂φ_vecdata = collect_cell_vector(U_reg,∇((φ->F((u₀s...,),φ)))(φ₀))
  ∂F∂φ_vec = allocate_vector(assem_deriv,∂F∂φ_vecdata)
  assems = (assem_U,assem_deriv)
  spaces = (trials,combine_fespaces(trials),V_φ,U_reg)
  caches = (dFdxj,∂F∂φ_vec)
  return StaggeredStateParamMap(F,spaces,assems,caches)
end

function (u_to_j::StaggeredStateParamMap)(u::AbstractVector,φ::AbstractVector)
  _,trial,V_φ,_ = u_to_j.spaces
  uh = FEFunction(trial,u)
  φh = FEFunction(V_φ,φ)
  return u_to_j(uh,φh)
end

function (u_to_j::StaggeredStateParamMap)(uh,φh)
  trials,_,_,_ = u_to_j.spaces
  uh_comb = _get_solutions(trials,uh)
  sum(u_to_j.F(uh_comb,φh))
end

# The following is a hack to get this working in the current GridapTopOpt ChainRules API.
#   This will be refactored in the future
function ChainRulesCore.rrule(u_to_j::StaggeredStateParamMap,uh,φh)
  F = u_to_j.F
  trials,_,_,U_reg = u_to_j.spaces
  _,assem_deriv = u_to_j.assems
  dFdxj,∂F∂φ_vec = u_to_j.caches

  uh_comb = _get_solutions(trials,uh)

  function u_to_j_pullback(dj)
    ## Compute ∂F/∂uh(uh,φh) and ∂F/∂φh(uh,φh)
    ∂F∂φ = ∇((φ->F((uh_comb...,),φ)))(φh)
    ∂F∂φ_vecdata = collect_cell_vector(U_reg,∂F∂φ)
    assemble_vector!(∂F∂φ_vec,assem_deriv,∂F∂φ_vecdata)
    dj_dFdxj(x...) = dj*dFdxj(x...)
    ∂F∂φ_vec .*= dj
    (  NoTangent(), dFdxj, ∂F∂φ_vec)
    # As above, this is really bad as dFdxj is a function and ∂F∂φ_vec is a vector. This is temporary
  end
  return u_to_j(uh,φh), u_to_j_pullback
end

function ChainRulesCore.rrule(u_to_j::StaggeredStateParamMap,u::AbstractVector,φ::AbstractVector)
  _,trial,V_φ,_ = u_to_j.spaces
  uh = FEFunction(trial,u)
  φh = FEFunction(V_φ,φ)
  return ChainRulesCore.rrule(u_to_j,uh,φh)
end