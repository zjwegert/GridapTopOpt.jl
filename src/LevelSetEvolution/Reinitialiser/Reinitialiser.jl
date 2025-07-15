"""
    abstract type Reinitialiser

Your own level-set reinitialisation method can be created by implementing
concrete functionality for `solve!`.
"""
abstract type Reinitialiser end

"""
    reinit!(::Reinitialiser,φ::AbstractVector)

Reinitialise the level-set function `φ` according to an Reinitialiser method.

Returns
- φ: The updated level-set function as an AbstractVector
- cache: The cache for the reinitialiser
"""
function reinit!(::Reinitialiser,φ::AbstractVector)
  @abstractmethod
end

"""
    reinit!(::Reinitialiser,φh)

Reinitialise the level-set function `φh` according to an Reinitialiser method.

φh should be an `FEFunction`.

Returns
- φ: The updated level-set function as an AbstractVector
- cache: The cache for the reinitialiser
"""
function reinit!(::Reinitialiser,φh)
  @abstractmethod
end

"""
    reinit!(::Reinitialiser,φ::AbstractVector,cache)

Reinitialise the level-set function `φ` according to an Reinitialiser method.
Reuse the supplied cache.

Returns
- φ: The updated level-set function as an AbstractVector
- cache: The cache for the reinitialiser
"""
function reinit!(::Reinitialiser,φ::AbstractVector,cache)
  @abstractmethod
end

"""
    reinit!(::Reinitialiser,φh,cache)

Reinitialise the level-set function `φh` according to an Reinitialiser method.
Reuse the supplied cache.

φh should be an `FEFunction`.

Returns
- φ: The updated level-set function as an AbstractVector
- cache: The cache for the reinitialiser
"""
function reinit!(::Reinitialiser,φh,cache)
  @abstractmethod
end

include("IdentityReinitialiser.jl")
include("FiniteDifferenceReinitialiser.jl")
include("StabilisedReinitialiser.jl")
include("HeatReinitialiser.jl")