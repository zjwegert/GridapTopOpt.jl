abstract type StabilisationMethod end

"""
    struct StabilisedReinitialiser{V,M} <: Reinitialiser

Stabilised FE method for level-set reinitialisation. Artificial viscosity
approach (`ArtificialViscosity`) based on Mallon et al. (2023). DOI: `10.1016/j.cma.2025.118203`.
Interior jump penalty approach (`InteriorPenalty`) adapted from that work and replaces
the artifical viscosity term with an interior jump penalty term.

# Parameters
- `nls::NonlinearSolver`: Nonlinear solver for solving the reinitialisation equation
- `stabilisation_method::A`: A `StabilisationMethod` method for stabilising the problem
- `Ωs::B`: `EmbeddedCollection` holding updatable triangulation and measures from GridapEmbedded
- `dΩ_bg::D`: Background mesh measure
- `space::C`: FE space for level-set function
- `assembler::Assembler`: Assembler for LS FE space
- `params::E`: Tuple of Nitsche parameter `γd` and mesh size `h`

# Note
- We expect the EmbeddedCollection `Ωs` to contain `:dΓ`. If this is not
  available we add it to the recipe list in `Ωs` and a warning will appear.
"""
struct StabilisedReinitialiser{A,B,C,D} <: Reinitialiser
  nls::NonlinearSolver
  stabilisation_method::A
  Ωs::EmbeddedCollection
  dΩ_bg::B
  space::C
  assembler::Assembler
  params::D
  @doc """
      StabilisedReinitialiser(V_φ::C,Ωs::EmbeddedCollection,dΩ_bg::B,h;
        γd = 20.0,
        nls = NewtonSolver(LUSolver();maxiter=20,rtol=1.e-14,verbose=true),
        assembler=SparseMatrixAssembler(V_φ,V_φ),
        stabilisation_method::A = InteriorPenalty(V_φ)
      ) where {A,B,C}

  Create an instance of `StabilisedReinitialiser` with the space for the level-set `V_φ`,
  the `EmbeddedCollection` `Ωs` for the triangulation and measures, the measure
  `dΩ_bg` for the background mesh, and the mesh size `h`. The mesh size `h` can
  either be a scalar or a `CellField` object.

  The optional arguments are:
  - `correct_ls`: Boolean for whether or not to ensure LS DOFs aren't zero
    this MUST be true for differentiation in unfitted methods.
  - `γd`: Interface penalty parameter for the reinitialisation equation.
  - `nls`: Nonlinear solver for solving the reinitialisation equation.
  - `assembler`: Assembler for the finite element space.
  - `stabilisation_method`: A `StabilisationMethod` method for stabilising the problem.
  """
  function StabilisedReinitialiser(V_φ::C,Ωs::EmbeddedCollection,dΩ_bg::B,h;
      correct_ls = true,
      γd = 20.0,
      nls = NewtonSolver(LUSolver();maxiter=20,rtol=1.e-14,verbose=true),
      assembler=SparseMatrixAssembler(V_φ,V_φ),
      stabilisation_method::A = InteriorPenalty(V_φ)) where {A,B,C}

    if !(:dΓ ∈ keys(Ωs.objects))
      @warn "Expected measure ':dΓ' not found in the
      EmbeddedCollection. This has been added to the recipe list.

      Ensure that you are not using ':dΓ' under a different
      name to avoid additional computation for cutting."
      function dΓ_recipe(cutgeo)
        Γ = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
        (;
          :dΓ => Measure(Γ,2get_order(V_φ))
        )
      end
      add_recipe!(Ωs,dΓ_recipe)
    end
    params = (;γd,h,correct_ls)
    new{A,B,C,typeof(params)}(nls,stabilisation_method,Ωs,dΩ_bg,V_φ,assembler,params)
  end
end

function reinit!(s::StabilisedReinitialiser,φh)
  nls, V_φ, assembler, = s.nls, s.space, s.assembler
  correct_ls = s.params.correct_ls
  # Temp solution for later reinitialisation
  φ = get_free_dof_values(φh);
  φ_tmp = copy(φ)
  # Weak form
  res, jac = get_residual_and_jacobian(s,φh,FEFunction(V_φ,φ_tmp))
  # Operator and cache
  op = get_algebraic_operator(FEOperator(res,jac,V_φ,V_φ,assembler))
  nls_cache = instantiate_caches(get_free_dof_values(φh),nls,op)
  cache = (;nls_cache,φ_tmp)
  # Solve
  solve!(φ_tmp,nls,op,nls_cache)
  if _get_solver_flag(nls.log) ∈ (SOLVER_CONVERGED_ATOL,SOLVER_CONVERGED_RTOL)
    copy!(get_free_dof_values(φh),φ_tmp)
    # Check LS
    correct_ls && correct_ls!(φh)
    update_collection!(s.Ωs,φh)
  end
  return get_free_dof_values(φh),cache
end

function reinit!(s::StabilisedReinitialiser,φh,cache)
  nls, V_φ, assembler, = s.nls, s.space, s.assembler
  correct_ls = s.params.correct_ls
  nls_cache, φ_tmp = cache
  # Update φ_tmp
  copy!(φ_tmp,get_free_dof_values(φh))
  # Weak form
  res, jac = get_residual_and_jacobian(s,φh,FEFunction(V_φ,φ_tmp))
  # Operator and cache
  op = get_algebraic_operator(FEOperator(res,jac,V_φ,V_φ,assembler))
  # Solve
  solve!(φ_tmp,nls,op,nls_cache)
  if _get_solver_flag(nls.log) ∈ (SOLVER_CONVERGED_ATOL,SOLVER_CONVERGED_RTOL)
    copy!(get_free_dof_values(φh),φ_tmp)
    # Check LS
    correct_ls && correct_ls!(φh)
    update_collection!(s.Ωs,φh)
  end
  return get_free_dof_values(φh),cache
end

function reinit!(s::StabilisedReinitialiser,φ::AbstractVector,args...)
  φh = FEFunction(get_ls_space(s),φ)
  reinit!(s,φh,args...)
end

struct ArtificialViscosity <: StabilisationMethod
  stabilisation_coefficent::Number
end

function get_residual_and_jacobian(s::StabilisedReinitialiser{ArtificialViscosity},φh,φh0)
  Ωs, dΩ_bg, = s.Ωs, s.dΩ_bg
  γd, h, _ = s.params
  ca = s.stabilisation_method.stabilisation_coefficent
  ϵ = 1e-20

  dΓ = Ωs.dΓ

  S(φ,∇φ,h) = φ/sqrt(φ^2 + h^2*(∇φ ⋅ ∇φ))
  S(φ,∇φ) = φ/sqrt(φ^2 + h^2*(∇φ ⋅ ∇φ))
  _sign(h::CellField) = S ∘ (φh0,∇(φh0),h)
  _sign(h::Real) = S ∘ (φh0,∇(φh0))

  W(∇u) = ∇u / (ϵ + norm(∇u))
  V(w) = ca*h*(sqrt ∘ ( w ⋅ w ))
  a(w,u,v) = ∫(v*(_sign(h)*(W ∘ ∇(w))) ⋅ ∇(u) + V((sign ∘ u)*(W ∘ ∇(w)))*∇(u) ⋅ ∇(v))dΩ_bg + ∫((γd/h)*v*u)dΓ
  b(w,v) = ∫((_sign(h))*v)dΩ_bg
  res(u,v) = a(u,u,v) - b(u,v)
  jac(u,du,v) = a(u,du,v)
  return res,jac
end

struct InteriorPenalty <: StabilisationMethod
  dΛ
  γg
end
function InteriorPenalty(V_φ::FESpace;γg=1.0)
  model = get_background_model(get_triangulation(V_φ))
  Λ = SkeletonTriangulation(model)
  dΛ = Measure(Λ,2get_order(V_φ))
  return InteriorPenalty(dΛ,γg)
end

function get_residual_and_jacobian(s::StabilisedReinitialiser{InteriorPenalty},φh,φh0)
  Ωs, dΩ_bg, = s.Ωs, s.dΩ_bg
  γd, h, _ = s.params
  ϵ = 1e-20
  dΛ = s.stabilisation_method.dΛ
  γg = s.stabilisation_method.γg

  dΓ = Ωs.dΓ
  γ(h) = γg*h^2

  S(φ,∇φ,h) = φ/sqrt(φ^2 + h^2*(∇φ ⋅ ∇φ))
  S(φ,∇φ) = φ/sqrt(φ^2 + h^2*(∇φ ⋅ ∇φ))
  _sign(h::CellField) = S ∘ (φh0,∇(φh0),h)
  _sign(h::Real) = S ∘ (φh0,∇(φh0))

  aₛ(u,v,h::CellField) = ∫(mean(γ ∘ h)*jump(∇(u)) ⋅ jump(∇(v)))dΛ
  aₛ(u,v,h::Real) = ∫(γ(h)*jump(∇(u)) ⋅ jump(∇(v)))dΛ

  W(∇u) = ∇u / (ϵ + norm(∇u))
  V(w) = ca*h*(sqrt ∘ ( w ⋅ w ))
  a(w,u,v) = ∫(v*(_sign(h)*(W ∘ ∇(w))) ⋅ ∇(u))dΩ_bg + aₛ(u,v,h) + ∫((γd/h)*v*u)dΓ
  b(w,v) = ∫((_sign(h))*v)dΩ_bg
  res(u,v) = a(u,u,v) - b(u,v)
  jac(u,du,v) = a(u,du,v)
  return res,jac
end

struct MultiStageStabilisedReinitialiser <: Reinitialiser
  stages::Vector{StabilisedReinitialiser}
end

function reinit!(s::MultiStageStabilisedReinitialiser,φh)
  wrapped_cache = ()
  for stage in s.stages
    _, cache = reinit!(stage,φh)
    wrapped_cache = (wrapped_cache...,cache)
  end
  return get_free_dof_values(φh),wrapped_cache
end

function reinit!(s::MultiStageStabilisedReinitialiser,φh,caches)
  wrapped_cache = ()
  for (stage,cache) in zip(s.stages,caches)
    _, cache = reinit!(stage,φh,cache)
    wrapped_cache = (wrapped_cache...,cache)
  end
  return get_free_dof_values(φh),wrapped_cache
end

## Helper
function _get_solver_flag(log::ConvergenceLog)
  r_abs = log.residuals[log.num_iters+1]
  r_rel = r_abs / log.residuals[1]
  flag  = finished_flag(log.tols,log.num_iters,r_abs,r_rel)
  return flag
end