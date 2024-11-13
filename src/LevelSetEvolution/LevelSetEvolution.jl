"""
    abstract type LevelSetEvolution

Your own evolution method can be implemented by implementing 
concrete functionality of the below.
"""
abstract type LevelSetEvolution end

"""
    evolve!(::LevelSetEvolution,φ,args...)

Evolve the level set function φ according to an evolution
method LevelSetEvolution.
"""
function evolve!(::LevelSetEvolution,φ,args...)
  @abstractmethod
end

"""
    reinit!(::LevelSetEvolution,φ,args...)

Reinitialise the level set function φ according to an
evolution method LevelSetEvolution.
"""
function reinit!(::LevelSetEvolution,φ,args...)
  @abstractmethod
end

"""
    get_dof_Δ(::LevelSetEvolution)

Return the distance betweem degree of freedom
"""
function get_dof_Δ(::LevelSetEvolution)
  @abstractmethod
end

include("Stencil.jl")
include("HamiltonJacobiEvolution.jl")
include("UnfittedEvolution/UnfittedEvolution.jl")