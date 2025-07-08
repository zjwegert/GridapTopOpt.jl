"""
    abstract type Evolver

Your own level-set evolution method can be created by implementing
concrete functionality for `solve!`, `get_dof_spacing`, and `get_ls_space`.
"""
abstract type Evolver end

"""
    get_min_dof_spacing(m::Evolver)

Return the minimum spacing of DOFs for the level-set function.
"""
function get_min_dof_spacing(::Evolver)
  @abstractmethod
end

"""
    get_ls_space(m::Evolver)

Return the finite element space used for the level-set function.
"""
function get_ls_space(::Evolver)
  @abstractmethod
end

"""
    evolve!(::Evolver,φ::AbstractVector,vel::AbstractVector,γ)

Evolve the level-set function `φ` using a velocity field `vel` and parameter
γ according to an Evolution method.

Returns
- φ: The updated level-set function as an AbstractVector
- cache: The cache for the evolver
"""
function evolve!(::Evolver,φ::AbstractVector,vel::AbstractVector,γ)
  @abstractmethod
end

"""
    evolve!(::Evolver,φh,velh,γ)

Evolve the level-set function `φh` using a velocity field `velh` and parameter
γ according to an Evolution method.

φh and velh should be `FEFunction`s.

Returns
- φ: The updated level-set function as an AbstractVector
- cache: The cache for the evolver
"""
function evolve!(::Evolver,φh,velh,γ)
  @abstractmethod
end

"""
    evolve!(::Evolver,φ::AbstractVector,vel::AbstractVector,γ,cache)

Evolve the level-set function `φ` using a velocity field `vel` and parameter
γ according to an Evolution method. Reuse the supplied cache.

Returns
- φ: The updated level-set function as an AbstractVector
- cache: The cache for the evolver
"""
function evolve!(::Evolver,φ::AbstractVector,vel::AbstractVector,γ,cache)
  @abstractmethod
end

"""
    evolve!(::Evolver,φh,velh,γ,cache)

Evolve the level-set function `φh` using a velocity field `velh` and parameter
γ according to an Evolution method. Reuse the supplied cache.

φh and velh should be `FEFunction`s.

Returns
- φ: The updated level-set function as an AbstractVector
- cache: The cache for the evolver
"""
function evolve!(::Evolver,φh,velh,γ,cache)
  @abstractmethod
end

include("FiniteDifferenceEvolver.jl")
include("CutFEMEvolver.jl")