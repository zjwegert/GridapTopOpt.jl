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
- `model::B`: FE model
- `space::C`: FE space for level-set function
- `dΩ::D`: Background mesh measure
- `assembler::Assembler`: Assembler for LS FE space 
- `params::E`: Tuple of Nitsche parameter `γd`, mesh size `h`, 
  and FE space `order`
- `cache`: Cache for evolver, initially `nothing`.

"""
mutable struct StabilisedReinit{A,B,C,D,E} <: Reinitialiser
  nls::NonlinearSolver
  stabilisation_method::A
  model::B
  space::C
  dΩ::D
  assembler::Assembler
  params::E
  cache
  function StabilisedReinit(model::B,V_φ::C,dΩ::D,h::Real;
      γd = 20.0,
      nls = NewtonSolver(LUSolver();maxiter=50,rtol=1.e-14,verbose=true),
      assembler=SparseMatrixAssembler(V_φ,V_φ),
      stabilisation_method::A = InteriorPenalty(V_φ)) where {A,B,C,D}
    params = (;γd,h,order=get_order(V_φ))
    new{A,B,C,D,typeof(params)}(nls,stabilisation_method,model,
      V_φ,dΩ,assembler,params,nothing)
  end
end

get_nls(s::StabilisedReinit) = s.nls
get_stabilisation_method(s::StabilisedReinit) = s.stabilisation_method
get_assembler(s::StabilisedReinit) = s.assembler
get_space(s::StabilisedReinit) = s.space
get_model(s::StabilisedReinit) = s.model
get_measure(s::StabilisedReinit) = s.dΩ
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
  return φh
end

struct ArtificialViscosity <: StabilisationMethod 
  stabilisation_coefficent::Number
end

function get_residual_and_jacobian(s::StabilisedReinit{ArtificialViscosity},φh)
  model, dΩ, = s.model, s.dΩ
  γd, h, order = s.params
  ca = s.stabilisation_method.stabilisation_coefficent
  ϵ = 1e-20

  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)
  Γ = EmbeddedBoundary(cutgeo)
  dΓ = Measure(Γ,2*order)

  W(u,∇u) = sign(u) * ∇u / (ϵ + norm(∇u))
  V(w) = ca*h*(sqrt ∘ ( w ⋅ w ))
  a(w,u,v) = ∫(v*(W ∘ (w,∇(w))) ⋅ ∇(u) + V(W ∘ (w,∇(w)))*∇(u) ⋅ ∇(v))dΩ + ∫((γd/h)*v*u)dΓ
  b(w,v) = ∫((sign ∘ w)*v)dΩ
  res(u,v) = a(u,u,v) - b(u,v)
  jac(u,du,v) = a(u,du,v)

  return res,jac
end

struct InteriorPenalty <: StabilisationMethod
  dΛ::Measure
end
function InteriorPenalty(V_φ::FESpace)
  Λ = SkeletonTriangulation(get_triangulation(V_φ))
  dΛ = Measure(Λ,2get_order(V_φ))
  return InteriorPenalty(dΛ)
end

function get_residual_and_jacobian(s::StabilisedReinit{InteriorPenalty},φh)
  model, dΩ, = s.model, s.dΩ
  γd, h, order = s.params
  ϵ = 1e-20
  dΛ = s.stabilisation_method.dΛ

  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)
  Γ = EmbeddedBoundary(cutgeo)
  dΓ = Measure(Γ,2*order)

  W(u,∇u) = sign(u) * ∇u / (ϵ + norm(∇u))
  V(w) = ca*h*(sqrt ∘ ( w ⋅ w ))
  a(w,u,v) = ∫(v*(W ∘ (w,∇(w))) ⋅ ∇(u))dΩ + ∫(h^2*jump(∇(u)) ⋅ jump(∇(v)))dΛ + ∫((γd/h)*v*u)dΓ
  b(w,v) = ∫((sign ∘ w)*v)dΩ
  res(u,v) = a(u,u,v) - b(u,v)
  jac(u,du,v) = a(u,du,v)

  return res,jac
end