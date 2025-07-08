"""
    struct FiniteDifferenceEvolver{O}

A standard forward Euler in time finite difference method for solving the
Hamilton-Jacobi evolution equation on order `O` finite elements in serial or parallel.

Based on the scheme by Osher and Fedkiw ([link](https://doi.org/10.1007/b98879)).

# Hamilton-Jacobi evolution equation
``\\frac{\\partial\\phi}{\\partial t} + V(\\boldsymbol{x})\\lVert\\boldsymbol{\\nabla}\\phi\\rVert = 0,``

with ``\\phi(0,\\boldsymbol{x})=\\phi_0(\\boldsymbol{x})`` and ``\\boldsymbol{x}\\in D,~t\\in(0,T)``.

# Parameters

- `stencil::Stencil`: Spatial finite difference stencil for a single step HJ
  equation and reinitialisation equation.
- `model`: A `CartesianDiscreteModel`.
- `space`: FE space for level-set function
- `perm`: A permutation vector
- `params`: Tuple of additional params
"""
struct FiniteDifferenceEvolver{O} <: LevelSetEvolution
  stencil :: Stencil
  model
  space
  perm
  params
end

"""
    FiniteDifferenceEvolver(stencil::Stencil,model,space,max_steps)

Create an instance of `FiniteDifferenceEvolver` given a stencil, model, FE space, and
additional optional arguments. This automatically creates the DoF permutation
to handle high-order finite elements.

# Optional Arguments
- `max_steps`: number of timesteps
- `correct_ls`: Boolean for whether or not to ensure LS DOFs aren't zero
  this MUST be true for differentiation in unfitted methods.
"""
function FiniteDifferenceEvolver(
  stencil::Stencil,
  model,
  space;
  max_steps=100,
  correct_ls = false
)
  @check isa(model,CartesianDiscreteModel) || isa(model,DistributedDiscreteModel) """
    We expect `model` to be a `CartesianDiscreteModel` or `DistributedDiscreteModel`
    for the current implementation of finite differencing on arbitrary order Lagrangian
    finite elements.
  """

  # Parameters
  order, isperiodic, Δ, ndof = get_stencil_params(model,space)
  params = (;isperiodic,Δ,ndof,max_steps,correct_ls)

  # Check that we have sufficient order
  check_order(stencil,order)

  # Dof permutation
  perm = create_dof_permutation(model,space,order)
  return FiniteDifferenceEvolver{order}(stencil,model,space,perm,params)
end

get_min_dof_spacing(m::FiniteDifferenceEvolver) = m.params.Δ
get_ls_space(m::FiniteDifferenceEvolver) = m.space

# Compute the time step for the `FiniteDifferenceEvolver`.
function compute_Δt(::FiniteDifferenceEvolver,Δ,γ,φ,vel)
  T = eltype(γ)
  v_norm = maximum(abs,vel)
  return γ * min(Δ...) / (eps(T)^2 + v_norm)
end

function evolve!(s::FiniteDifferenceEvolver{O},φ::AbstractVector,vel::AbstractVector,γ) where O
  # Create caches
  stencil,perm,ndof=s.stencil,s.perm,s.params.ndof
  cache  = allocate_caches(stencil,φ,vel,perm,O,ndof)
  # Evolve
  evolve!(s,φ,vel,γ,cache)
end

function evolve!(s::FiniteDifferenceEvolver{O},φ::PVector,vel::PVector,γ,cache) where O
  _, _, perm_caches, stencil_cache = cache
  Δ, isperiodic, correct_ls = s.params.Δ, s.params.isperiodic, s.params.correct_ls
  ndof, max_steps = s.params.ndof, s.params.max_steps

  _φ   = (O >= 2) ? permute!(perm_caches[1],φ,s.perm) : φ
  _vel = (O >= 2) ? permute!(perm_caches[2],vel,s.perm) : vel

  ## CFL Condition (requires γ≤1.0)
  @check γ <= 1 "The CFL condition for the FiniteDifferenceReinitialiser requires γ≤1"
  Δt = compute_Δt(s,Δ,γ,φ,vel)
  for _ in 1:max_steps
    # Apply operations across partitions
    map(local_views(_φ),local_views(_vel),stencil_cache,ndof) do _φ,_vel,stencil_cache,S
      φ_mat   = reshape(_φ,S)
      vel_mat = reshape(_vel,S)
      evolve!(s.stencil,φ_mat,vel_mat,Δt,Δ,isperiodic,stencil_cache)
    end
    # Update ghost nodes
    consistent!(_φ) |> fetch
  end
  φ = (O >= 2) ? permute_inv!(φ,_φ,s.perm) : _φ
  correct_ls && correct_ls!(φ)
  return φ
end

function evolve!(s::FiniteDifferenceEvolver{O},φ::Vector,vel::Vector,γ,cache) where O
  _, _, perm_caches, stencil_cache = cache
  Δ, isperiodic, correct_ls = s.params.Δ, s.params.isperiodic, s.params.correct_ls
  ndof, max_steps = s.params.ndof, s.params.max_steps

  _φ   = (O >= 2) ? permute!(perm_caches[1],φ,s.perm) : φ
  _vel = (O >= 2) ? permute!(perm_caches[2],vel,s.perm) : vel

  ## CFL Condition (requires γ≤1.0)
  @check γ <= 1 "The CFL condition for the FiniteDifferenceReinitialiser requires γ≤1"
  Δt = compute_Δt(s,Δ,γ,φ,vel)
  for _ in 1:max_steps
    φ_mat   = reshape(_φ,ndof)
    vel_mat = reshape(_vel,ndof)
    evolve!(s.stencil,φ_mat,vel_mat,Δt,Δ,isperiodic,stencil_cache)
  end
  φ = (O >= 2) ? permute_inv!(φ,_φ,s.perm) : _φ
  correct_ls && correct_ls!(φ)
  return φ
end

function evolve!(s::FiniteDifferenceEvolver,φh,velh,args...)
  evolve!(s,get_free_dof_values(φh),get_free_dof_values(velh),args...)
end