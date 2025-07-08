"""
    struct IdentityReinitialiser <: Reinitialiser end

A level-set function reinitialiser that does nothing.
"""
struct IdentityReinitialiser <: Reinitialiser end

function reinit!(::Reinitialiser,φ::AbstractVector)
  φ,nothing
end

function reinit!(::Reinitialiser,φh)
  get_free_dof_values(φh),nothing
end

function reinit!(::Reinitialiser,φ::AbstractVector,cache)
  φ,nothing
end

function reinit!(::Reinitialiser,φh,cache)
  get_free_dof_values(φh),nothing
end