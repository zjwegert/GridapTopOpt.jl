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
- `cache`: Cache for evolver, initially `nothing`.

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
  function StabilisedReinit(V_φ::C,Ωs::EmbeddedCollection,dΩ_bg::B,h::Real;
      γd = 20.0,
      nls = NewtonSolver(LUSolver();maxiter=50,rtol=1.e-14,verbose=true),
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
get_cache(s::StabilisedReinit) = s.cache

function solve!(s::StabilisedReinit,φh,nls_cache::Nothing)
  nls, V_φ, assembler, = s.nls, s.space, s.assembler
  # Weak form
  res, jac = get_residual_and_jacobian(s,φh)
  # Operator and cache
  op = get_algebraic_operator(FEOperator(res,jac,V_φ,V_φ,assembler))
  nls_cache = instantiate_caches(get_free_dof_values(φh),nls,op)
  s.cache = nls_cache
  # Solve
  solve!(get_free_dof_values(φh),nls,op,nls_cache)
  update_collection!(s.Ωs,φh) # TODO: remove?
  return φh
end

function solve!(s::StabilisedReinit,φh,nls_cache)
  nls, V_φ, assembler, = s.nls, s.space, s.assembler
  # Weak form
  res, jac = get_residual_and_jacobian(s,φh)
  # Operator and cache
  op = get_algebraic_operator(FEOperator(res,jac,V_φ,V_φ,assembler))
  # Solve
  solve!(get_free_dof_values(φh),nls,op,nls_cache)
  update_collection!(s.Ωs,φh) # TODO: remove?
  return φh
end

struct ArtificialViscosity <: StabilisationMethod
  stabilisation_coefficent::Number
end

function get_residual_and_jacobian(s::StabilisedReinit{ArtificialViscosity},φh)
  Ωs, dΩ_bg, = s.Ωs, s.dΩ_bg
  γd, h = s.params
  ca = s.stabilisation_method.stabilisation_coefficent
  ϵ = 1e-20

  update_collection!(Ωs,φh)
  dΓ = Ωs.dΓ

  W(u,∇u) = sign(u) * ∇u / (ϵ + norm(∇u))
  V(w) = ca*h*(sqrt ∘ ( w ⋅ w ))
  a(w,u,v) = ∫(v*(W ∘ (w,∇(w))) ⋅ ∇(u) + V(W ∘ (w,∇(w)))*∇(u) ⋅ ∇(v))dΩ_bg + ∫((γd/h)*v*u)dΓ
  b(w,v) = ∫((sign ∘ w)*v)dΩ_bg
  res(u,v) = a(u,u,v) - b(u,v)
  jac(u,du,v) = a(u,du,v)
  # jac(u,du,v) = jacobian(res,[u,v],1)
  return res,jac
end

struct InteriorPenalty <: StabilisationMethod
  dΛ
end
function InteriorPenalty(V_φ::FESpace)
  Λ = SkeletonTriangulation(get_triangulation(V_φ))
  dΛ = Measure(Λ,2get_order(V_φ))
  return InteriorPenalty(dΛ)
end

function get_residual_and_jacobian(s::StabilisedReinit{InteriorPenalty},φh)
  Ωs, dΩ_bg, = s.Ωs, s.dΩ_bg
  γd, h = s.params
  ϵ = 1e-20
  dΛ = s.stabilisation_method.dΛ

  update_collection!(Ωs,φh)
  dΓ = Ωs.dΓ

  W(u,∇u) = sign(u) * ∇u / (ϵ + norm(∇u))
  a(w,u,v) = ∫(v*(W ∘ (w,∇(w))) ⋅ ∇(u))dΩ_bg + ∫(h^2*jump(∇(u)) ⋅ jump(∇(v)))dΛ + ∫((γd/h)*v*u)dΓ
  b(w,v) = ∫((sign ∘ w)*v)dΩ_bg
  res(u,v) = a(u,u,v) - b(u,v)
  jac(u,du,v) = a(u,du,v)
  # jac(u,du,v) = jacobian(res,[u,v],1)
  return res,jac
end