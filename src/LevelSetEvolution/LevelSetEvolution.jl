abstract type AbstractLevelSetEvolution end

function evolve!(::AbstractLevelSetEvolution,φ,args...)
  @abstractmethod
end
function reinit!(::AbstractLevelSetEvolution,φ,args...)
  @abstractmethod
end
function get_evolver(::AbstractLevelSetEvolution)
  @abstractmethod
end
function get_reinitialiser(::AbstractLevelSetEvolution)
  @abstractmethod
end
function get_min_dof_spacing(::AbstractLevelSetEvolution)
  @abstractmethod
end
function get_ls_space(::AbstractLevelSetEvolution)
  @abstractmethod
end

#### LevelSetEvolution
"""
    struct LevelSetEvolution{A,B} <: AbstractLevelSetEvolution


"""
struct LevelSetEvolution{A,B} <: AbstractLevelSetEvolution
  evolver       :: A
  reinitialiser :: B
end

"""
    evolve!(s::LevelSetEvolution,φ,args...)

Evolve the level set function φ using the evolver in LevelSetEvolution.
"""
function evolve!(s::LevelSetEvolution,φ,args...)
  evolve!(get_evolver(s),φ,args...)
end

"""
    reinit!(::LevelSetEvolution,φ,args...)

Reinitialise the level set function φ using the reinitialiser in LevelSetEvolution.
"""
function reinit!(s::LevelSetEvolution,φ,args...)
  reinit!(get_reinitialiser(s),φ,args...)
end

"""
    get_evolver(s::LevelSetEvolution)

Return the level-set function evolver.
"""
function get_evolver(s::LevelSetEvolution)
  s.evolver
end

"""
    reinitialiser(s::LevelSetEvolution)

Return the level-set function reinitialiser.
"""
function get_reinitialiser(s::LevelSetEvolution)
  s.reinitialiser
end

"""
    get_min_dof_spacing(s::LevelSetEvolution)

Return the minimum spacing of DOFs for the level-set function.
"""
function get_min_dof_spacing(s::LevelSetEvolution)
  get_min_dof_spacing(get_evolver(s))
end

"""
    get_ls_space(s::LevelSetEvolution)

Return the finite element space used for the level-set function.
"""
function get_ls_space(s::LevelSetEvolution)
  get_ls_space(get_evolver(s))
end

include("Utilities/Helpers.jl")
include("Stencil/Stencil.jl")
include("Evolver/Evolver.jl")
include("Reinitialiser/Reinitialiser.jl")

# Backwards compat/errors
function UnfittedFEEvolution(args...)
  error(
    """
    As of v0.4.0, UnfittedFEEvolution has been deprecated in favour of LevelSetEvolution.
    Please replace UnfittedFEEvolution with LevelSetEvolution in your script.

    This method and error will be removed in a future release.
    """
  )
end

function HamiltonJacobiEvolution(args...)
  error(
    """
    As of v0.4.0, HamiltonJacobiEvolution has been deprecated in favour of splitting
    the evolver and reinitialiser into FiniteDifferenceEvolver and FiniteDifferenceReinitialiser
    respectively.

    In your script, please replace

      ls_evo = HamiltonJacobiEvolution(FirstOrderStencil(D,T),model,V_φ,tol,max_steps)

    with

      evo = FiniteDifferenceEvolver(FirstOrderStencil(D,T),model,V_φ;max_steps)
      reinit = FiniteDifferenceReinitialiser(FirstOrderStencil(D,T),model,V_φ;γ_reinit)
      ls_evo = LevelSetEvolution(evo,reinit)

    This method and error will be removed in a future release.
    """
  )
end

function CutFEMEvolve(args...)
  error(
    """
    As of v0.4.0, CutFEMEvolve has been renamed to CutFEMEvolver to conform with
    the refactor.
    """
  )
end

function StabilisedReinit(args...)
  error(
    """
    As of v0.4.0, StabilisedReinit has been renamed to StabilisedReinitialiser to
    conform with the refactor.
    """
  )
end