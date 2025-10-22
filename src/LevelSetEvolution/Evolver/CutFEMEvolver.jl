"""
    struct CutFEMEvolver{V,M} <: Evolver

CutFEM method for solving level-set evolution based on method developed by
- Burman et al. (2018). DOI: `10.1016/j.cma.2017.09.005`.
- Burman et al. (2017). DOI: `10.1016/j.cma.2016.12.021`.
- Burman and Fernández (2009). DOI: `10.1016/j.cma.2009.02.011`
This solves the transport equation
``
\\frac{\\partial\\phi(t,\boldsymbol{x})}{\\partial t}+\\boldsymbol{\\beta}\\cdot\\boldsymbol{\\nabla}\\phi(t,\\boldsymbol{x})=0,
``
with ``boldsymbol{\\beta}=\\boldsymbol{n}v_h``, ``\\phi(0,\\boldsymbol{x})=\\phi_0(\\boldsymbol{x}),`` and ``\\quad\\boldsymbol{x}\\in D,~t\\in(0,T)``.

The method uses a Crank-Nicolson time-stepping scheme.

# Parameters
- `ode_ls::LinearSolver`: Linear solver for the Crank-Nicolson time-stepping
- `dΩ_bg::C`: Measure for integration
- `space::B`: Level-set FE space
- `assembler::Assembler`: FE assembler
- `params::D`: Tuple of stabilisation parameter `γg`, mesh sizes `h`, and
  max steps `max_steps`, and background mesh skeleton parameters
"""
struct CutFEMEvolver{A,B,C} <: Evolver
  ode_ls::LinearSolver
  dΩ_bg::A
  space::B
  assembler::Assembler
  params::C

  @doc """
      CutFEMEvolver(V_φ::B,dΩ_bg::A,h;
        correct_ls = true,
        max_steps=10,
        γg = 0.1,
        ode_ls = LUSolver(),
        assembler=SparseMatrixAssembler(V_φ,V_φ)
      ) where {A,B}

  Create an instance of `CutFEMEvolver` with the space for the level-set `V_φ`,
  the measure `dΩ_bg` for the background mesh, and the mesh size `h`.
  The mesh size `h` can either be a scalar or a `CellField` object.

  The optional arguments are:
  - `correct_ls`: Boolean for whether or not to ensure LS DOFs aren't zero
    this MUST be true for differentiation in unfitted methods.
  - `max_steps`: Maximum number of steps for the ODE solver.
  - `γg`: Stabilisation parameter for the continuous interior penalty term.
  - `ode_ls`: Linear solver for the ODE solver.
  - `assembler`: Assembler for the finite element space, default is `SparseMatrixAssembler(V_φ,V_φ)`.
  """
  function CutFEMEvolver(V_φ::B,dΩ_bg::A,h;
      correct_ls = true,
      max_steps = 10,
      γg = 0.1,
      ode_ls = LUSolver(),
      assembler = SparseMatrixAssembler(V_φ,V_φ)) where {A,B}
    model = get_background_model(get_triangulation(V_φ))
    Γg = SkeletonTriangulation(model)
    dΓg = Measure(Γg,2get_order(V_φ))
    n_Γg = get_normal_vector(Γg)
    hmin = minimum(get_element_diameters(model))
    uhd = zero(V_φ)
    params = (;γg,h,hmin,max_steps,dΓg,n_Γg,correct_ls,uhd)
    new{A,B,typeof(params)}(ode_ls,dΩ_bg,V_φ,assembler,params)
  end
end

function get_min_dof_spacing(s::CutFEMEvolver)
  V_φ = get_ls_space(s)
  hmin = s.params.hmin
  return hmin/get_order(V_φ)
end

function get_ls_space(s::CutFEMEvolver)
  s.space
end

function evolve!(s::CutFEMEvolver,φh,velh,γ)
  ls, V_φ, assem, params = s.ode_ls, s.space, s.assembler, s.params
  hmin, max_steps, correct_ls, uhd = params.hmin, params.max_steps, params.correct_ls, params.uhd
  V_φ = get_ls_space(s)

  # Weak form
  Δt = γ*hmin
  tF = Δt*max_steps
  φ_tmp = copy(get_free_dof_values(φh))
  φh_tmp = FEFunction(V_φ,φ_tmp)
  A, B = get_weak_form(φh_tmp,velh,Δt,s)

  # Setup
  φ = get_free_dof_values(φh)
  op = AffineFEOperator(A,v->B(v,φh),V_φ,V_φ,assem)
  K, b = get_matrix(op), get_vector(op)
  ns = numerical_setup(symbolic_setup(ls,K),K)

  # March
  ti = 0.0
  while ti <= tF - Gridap.ODEs.ε
    solve!(φ,ns,b)
    assemble_vector!(v -> B(v,φh) - A(uhd,v),b,assem,V_φ)
    ti += Δt
  end

  correct_ls && correct_ls!(φh)
  cache = (;op,ns,φh_tmp)
  return get_free_dof_values(φh), cache
end

function evolve!(s::CutFEMEvolver,φh,velh,γ,cache)
  V_φ, assem, params = s.space, s.assembler, s.params
  hmin, max_steps, correct_ls, uhd = params.hmin, params.max_steps, params.correct_ls, params.uhd
  op, ns, φh_tmp = cache

  # Weak form
  Δt = γ*hmin
  tF = Δt*max_steps
  copyto!(get_free_dof_values(φh_tmp),get_free_dof_values(φh))
  A, B = get_weak_form(φh_tmp,velh,Δt,s)

  # Setup
  φ = get_free_dof_values(φh)
  K, b = get_matrix(op), get_vector(op)
  assemble_matrix!(A,K,assem,V_φ,V_φ)
  assemble_vector!(v->B(v,φh) - A(uhd,v),b,assem,V_φ)
  numerical_setup!(ns,K)

  # March
  ti = 0.0
  while ti <= tF - Gridap.ODEs.ε
    solve!(φ,ns,b)
    assemble_vector!(v -> B(v,φh) - A(uhd,v),b,assem,V_φ)
    ti += Δt
  end
  correct_ls && correct_ls!(φh)
  return get_free_dof_values(φh), cache
end

function get_weak_form(φh,velh,Δt,s::CutFEMEvolver)
  dΩ_bg, params = s.dΩ_bg, s.params
  γg, h, dΓg, n_Γg = params.γg, params.h, params.dΓg, params.n_Γg
  ϵ = 1e-20

  v_norm = maximum(abs,get_free_dof_values(velh))
  β(vh,∇φ) = vh/(ϵ + v_norm) * ∇φ/(ϵ + norm(∇φ))
  γ(h) = γg*h^2
  βh = β ∘ (velh,∇(φh))
  βh_n_Γg = abs ∘ (βh.plus ⋅ n_Γg.plus)

  aₛ(u,v,h::CellField) = ∫(mean(γ ∘ h)*βh_n_Γg*jump(∇(u) ⋅ n_Γg)*jump(∇(v) ⋅ n_Γg))dΓg
  aₛ(u,v,h::Real) = ∫(γ(h)*βh_n_Γg*jump(∇(u) ⋅ n_Γg)*jump(∇(v) ⋅ n_Γg))dΓg

  a(u,v) = ∫((βh ⋅ ∇(u)) * v)dΩ_bg + aₛ(u,v,h)
  m(∂ₜu, v) = ∫(∂ₜu * v)dΩ_bg

  A(uₙ,v) = m(uₙ, v) + Δt/2*a(uₙ, v)
  B(v,uₙ₋₁) = m(uₙ₋₁, v) - Δt/2*a(uₙ₋₁,v)

  return A, B
end

# Avoid ambiguities
function evolve!(s::CutFEMEvolver,φh,velh,γ,::Nothing)
  evolve!(s,φh,velh,γ)
end
function evolve!(s::CutFEMEvolver,φ::AbstractVector,vel::AbstractVector,γ,::Nothing)
  φh = FEFunction(get_ls_space(s),φ)
  velh = FEFunction(get_ls_space(s),vel)
  evolve!(s,φh,velh,γ)
end
function evolve!(s::CutFEMEvolver,φ::AbstractVector,vel::AbstractVector,args...)
  φh = FEFunction(get_ls_space(s),φ)
  velh = FEFunction(get_ls_space(s),vel)
  evolve!(s,φh,velh,args...)
end

# Old constructor
function CutFEMEvolver(V_φ::B,Ωs::EmbeddedCollection,dΩ_bg::A,h;
    correct_ls = true,
    max_steps=10,
    γg = 0.1,
    ode_ls = LUSolver(),
    ode_nl = ode_ls,
    ode_solver = MutableRungeKutta(ode_nl, ode_ls, 0.1, :DIRK_CrankNicolson_2_2),
    assembler=SparseMatrixAssembler(V_φ,V_φ)) where {A,B}
  @warn """
  This constructor will be deprecated in future releases. Please use the new CutFEMEvolver
  constructor that does not require an EmbeddedCollection:
    CutFEMEvolver(V_φ,dΩ_bg,h;kwargs...)
  """ maxlog=1
  CutFEMEvolver(V_φ,dΩ_bg,h;correct_ls,max_steps,γg,ode_ls,assembler)
end