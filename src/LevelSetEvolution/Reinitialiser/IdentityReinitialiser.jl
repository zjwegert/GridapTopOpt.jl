"""
    struct IdentityReinitialiser <: Reinitialiser end

A level-set function reinitialiser that does nothing.
"""
struct IdentityReinitialiser <: Reinitialiser end

function reinit!(::IdentityReinitialiser,φ::AbstractVector)
  φ,nothing
end

function reinit!(::IdentityReinitialiser,φh)
  get_free_dof_values(φh),nothing
end

function reinit!(::IdentityReinitialiser,φ::AbstractVector,cache)
  φ,nothing
end

function reinit!(::IdentityReinitialiser,φh,cache)
  get_free_dof_values(φh),nothing
end