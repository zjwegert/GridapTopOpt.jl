"""
    struct FiniteDifferenceReinitialiser{O}

A standard forward Euler in time finite difference method for solving the
reinitialisation equation on order `O` finite elements in serial or parallel.

Based on the scheme by Osher and Fedkiw ([link](https://doi.org/10.1007/b98879)).

# Reinitialisation equation
``\\frac{\\partial\\phi}{\\partial t} + \\mathrm{sign}(\\phi_0)(\\lVert\\boldsymbol{\\nabla}\\phi\\rVert-1) = 0,``

with ``\\phi(0,\\boldsymbol{x})=\\phi_0(\\boldsymbol{x})`` and ``\\boldsymbol{x}\\in D,~t\\in(0,T)``.

# Parameters

- `stencil::Stencil`: Spatial finite difference stencil for a single step HJ
  equation and reinitialisation equation.
- `model`: A `CartesianDiscreteModel`.
- `space`: FE space for level-set function
- `perm`: A permutation vector
- `params`: Tuple of additional params
"""
struct FiniteDifferenceReinitialiser{O} <: Reinitialiser
  stencil :: Stencil
  model
  space
  perm
  params
end

"""
    FiniteDifferenceReinitialiser(stencil::Stencil,model,space;γ_reinit,tol,max_steps)

Create an instance of `FiniteDifferenceReinitialiser` given a stencil, model, FE space, and
additional optional arguments. This automatically creates the DoF permutation
to handle high-order finite elements.

# Optional Arguments
- `γ_reinit`: coeffient on the time step size.
- `tol`: stopping tolerance for reinitialiser
- `max_steps`: number of timesteps
- `correct_ls`: Boolean for whether or not to ensure LS DOFs aren't zero
  this MUST be true for differentiation in unfitted methods.
"""
function FiniteDifferenceReinitialiser(
  stencil::Stencil,
  model,
  space;
  γ_reinit=0.5,
  tol=1.e-3,
  max_steps=2000,
  correct_ls = false
)
  @check isa(model,CartesianDiscreteModel) || isa(model,DistributedDiscreteModel) """
    We expect `model` to be a `CartesianDiscreteModel` or `DistributedDiscreteModel`
    for the current implementation of finite differencing on arbitrary order Lagrangian
    finite elements.
  """
  @check γ_reinit <= 0.5 "The CFL condition for the FiniteDifferenceReinitialiser requires γ≤0.5"

  # Parameters
  order, isperiodic, Δ, ndof = get_stencil_params(model,space)
  params = (;isperiodic,Δ,ndof,max_steps,tol,γ_reinit,correct_ls)

  # Check that we have sufficient order
  check_order(stencil,order)

  # Dof permutation
  perm = create_dof_permutation(model,space,order)
  return FiniteDifferenceReinitialiser{order}(stencil,model,space,perm,params)
end

# Compute the time step for the `FiniteDifferenceReinitialiser`.
function compute_Δt(::FiniteDifferenceReinitialiser,Δ,γ_reinit,φ,vel)
  T = eltype(γ_reinit)
  v_norm = maximum(abs,vel)
  return γ_reinit * min(Δ...) / (eps(T)^2 + v_norm)
end

function reinit!(s::FiniteDifferenceReinitialiser{O},φ::AbstractVector) where O
  # Create caches
  stencil,perm,ndof=s.stencil,s.perm,s.params.ndof
  cache  = allocate_caches(stencil,φ,vel,perm,O,ndof)
  # Evolve
  reinit!(s,φ,vel,cache)
end

function reinit!(s::FiniteDifferenceReinitialiser{O},φ::PVector) where O
  φ_tmp, vel_tmp, perm_caches, stencil_cache = s.cache
  γ_reinit, Δ, isperiodic, ndof  = s.params.γ_reinit,s.params.Δ,s.params.isperiodic,s.params.ndof
  tol, max_steps, correct_ls = s.params.tol, s.params.max_steps_reinit, s.params.correct_ls

  _φ = (O >= 2) ? permute!(perm_caches[1],φ,s.perm) : φ

  ## CFL Condition (requires γ≤0.5). Note infnorm(S) = 1.0
  Δt = compute_Δt(s,Δ,γ_reinit,_φ,1.0)

  # Apply operations across partitions
  step = 1; err = maximum(abs,φ); fill!(φ_tmp,0.0)
  while (err > tol) && (step <= max_steps)
    # Step of 1st order upwind reinitialisation equation
    map(local_views(φ_tmp),local_views(_φ),local_views(vel_tmp),stencil_cache,ndof) do φ_tmp,_φ,vel_tmp,stencil_cache,S
      φ_tmp_mat   = reshape(φ_tmp,S)
      φ_mat       = reshape(_φ,S)
      vel_tmp_mat = reshape(vel_tmp,S)
      reinit!(s.stencil,φ_tmp_mat,φ_mat,vel_tmp_mat,Δt,Δ,isperiodic,stencil_cache)
    end

    # Compute error
    _φ .-= φ_tmp # φ - φ_tmp
    err = maximum(abs,_φ)
    step += 1

    # Update φ
    copy!(_φ,φ_tmp)
    consistent!(_φ) |> fetch # We exchange ghosts here!
  end
  φ = (O >= 2) ? permute_inv!(φ,_φ,s.perm) : _φ
  correct_ls && correct_ls!(φ)
  return φ
end

function reinit!(s::FiniteDifferenceReinitialiser{O},φ::Vector) where O
  φ_tmp, vel_tmp, perm_caches, stencil_cache = s.cache
  γ_reinit, Δ, isperiodic, ndof  = s.params.γ_reinit,s.params.Δ,s.params.isperiodic,s.params.ndof
  tol, max_steps, correct_ls = s.params.tol, s.params.max_steps_reinit, s.params.correct_ls

  _φ = (O >= 2) ? permute!(perm_caches[1],φ,s.perm) : φ

  ## CFL Condition (requires γ≤0.5)
  Δt = compute_Δt(s,Δ,γ_reinit,_φ,1.0)

  # Apply operations across partitions
  step = 1; err = maximum(abs,φ); fill!(φ_tmp,0.0)
  while (err > tol) && (step <= max_steps)
    # Step of 1st order upwind reinitialisation equation
    φ_tmp_mat   = reshape(φ_tmp,ndof)
    φ_mat       = reshape(_φ,ndof)
    vel_tmp_mat = reshape(vel_tmp,ndof)
    reinit!(s.stencil,φ_tmp_mat,φ_mat,vel_tmp_mat,Δt,Δ,isperiodic,stencil_cache)

    # Compute error
    _φ .-= φ_tmp
    err = maximum(abs,_φ)
    step += 1

    # Update φ
    copy!(_φ,φ_tmp)
  end
  φ = (O >= 2) ? permute_inv!(φ,_φ,s.perm) : _φ
  correct_ls && correct_ls!(φ)
  return φ
end

function reinit!(s::FiniteDifferenceReinitialiser,φh,args...)
  reinit!(s,get_free_dof_values(φh),args...)
end