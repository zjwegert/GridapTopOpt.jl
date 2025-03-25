abstract type StabilisationMethod end

"""
    mutable struct StabilisedReinit{V,M} <: Reinitialiser

Stabilised FE method for level-set reinitialisation. Artificial viscosity
approach (`ArtificialViscosity`) based on Mallon et al. (2023). DOI: `10.48550/arXiv.2303.13672`.
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
- `cache`: Cache for reinitialiser, initially `nothing`.

# Note
- We expect the EmbeddedCollection `Ωs` to contain `:dΓ`. If this is not
  available we add it to the recipe list in `Ωs` and a warning will appear.
"""
mutable struct StabilisedReinit{A,B,C,D} <: Reinitialiser
  nls::NonlinearSolver
  stabilisation_method::A
  Ωs::EmbeddedCollection
  dΩ_bg::B
  space::C
  assembler::Assembler
  params::D
  cache
  function StabilisedReinit(V_φ::C,Ωs::EmbeddedCollection,dΩ_bg::B,h;
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
    params = (;γd,h)
    new{A,B,C,typeof(params)}(nls,stabilisation_method,Ωs,
    dΩ_bg,V_φ,assembler,params,nothing)
  end
end

get_nls(s::StabilisedReinit) = s.nls
get_stabilisation_method(s::StabilisedReinit) = s.stabilisation_method
get_assembler(s::StabilisedReinit) = s.assembler
get_space(s::StabilisedReinit) = s.space
get_embedded_collection(s::StabilisedReinit) = s.Ωs
get_measure(s::StabilisedReinit) = s.dΩ_bg
get_params(s::StabilisedReinit) = s.params
get_element_diameters(s::StabilisedReinit) = s.params.h
get_cache(s::StabilisedReinit) = s.cache

function solve!(s::StabilisedReinit,φh,cache::Nothing)
  nls, V_φ, assembler, = s.nls, s.space, s.assembler
  # Temp solution for later reinitialisation
  φ = get_free_dof_values(φh);
  φ_tmp = copy(φ)
  # Weak form
  res, jac = get_residual_and_jacobian(s,φh,FEFunction(V_φ,φ_tmp))
  # Operator and cache
  op = get_algebraic_operator(FEOperator(res,jac,V_φ,V_φ,assembler))
  nls_cache = instantiate_caches(get_free_dof_values(φh),nls,op)
  s.cache = (;nls_cache,φ_tmp)
  # Solve
  solve!(φ_tmp,nls,op,nls_cache)
  if _get_solver_flag(nls.log) ∈ (SOLVER_CONVERGED_ATOL,SOLVER_CONVERGED_RTOL)
    copy!(get_free_dof_values(φh),φ_tmp)
    update_collection!(s.Ωs,φh)
  end
  # Check LS
  correct_ls!(φh)
  return φh
end

function solve!(s::StabilisedReinit,φh,cache)
  nls, V_φ, assembler, = s.nls, s.space, s.assembler
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
    update_collection!(s.Ωs,φh)
  end
  # Check LS
  correct_ls!(φh)
  return φh
end

struct ArtificialViscosity <: StabilisationMethod
  stabilisation_coefficent::Number
end

function get_residual_and_jacobian(s::StabilisedReinit{ArtificialViscosity},φh,φh0)
  Ωs, dΩ_bg, = s.Ωs, s.dΩ_bg
  γd, h = s.params
  ca = s.stabilisation_method.stabilisation_coefficent
  ϵ = 1e-20

  # correct_ls!(φh)
  # update_collection!(Ωs,φh)
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
  # jac(u,du,v) = jacobian(res,[u,v],1)
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

function get_residual_and_jacobian(s::StabilisedReinit{InteriorPenalty},φh,φh0)
  Ωs, dΩ_bg, = s.Ωs, s.dΩ_bg
  γd, h = s.params
  ϵ = 1e-20
  dΛ = s.stabilisation_method.dΛ
  γg = s.stabilisation_method.γg

  # correct_ls!(φh)
  # update_collection!(Ωs,φh)
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
  # jac(u,du,v) = jacobian(res,[u,v],1)
  return res,jac
end

struct MultiStageStabilisedReinit <: Reinitialiser
  stages::Vector{StabilisedReinit}
  function MultiStageStabilisedReinit(stages::Vector{<:StabilisedReinit})
    h = get_element_diameters(first(stages))
    V_φ = get_space(first(stages))
    for stage in stages
      @check h === get_element_diameters(stage)
      @check V_φ === get_space(stage)
    end
    new(stages)
  end
end

get_cache(s::MultiStageStabilisedReinit) = get_cache.(s.stages)
get_element_diameters(s::MultiStageStabilisedReinit) = get_element_diameters(first(s.stages))
get_space(s::MultiStageStabilisedReinit) = get_space(first(s.stages))

function solve!(s::MultiStageStabilisedReinit,φh,caches)
  for (stage,cache) in zip(s.stages,caches)
    solve!(stage,φh,cache)
  end
  return φh
end

## Helper
function _get_solver_flag(log::ConvergenceLog)
  r_abs = log.residuals[log.num_iters+1]
  r_rel = r_abs / log.residuals[1]
  flag  = finished_flag(log.tols,log.num_iters,r_abs,r_rel)
  return flag
end