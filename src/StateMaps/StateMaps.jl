include("FEStateMaps.jl")
include("StateParamMaps.jl")
include("AffineFEStateMaps.jl")
include("NonlinearFEStateMaps.jl")
include("RepeatingAffineFEStateMaps.jl")
include("StaggeredFEStateMaps.jl")
include("PDEConstrainedFunctionals.jl")

"""
    Gridap.gradient(F,uh::Vector{<:CellField},K::Int)

Given a function `F` that returns a DomainContribution when called, and a vector of
`FEFunctions` `uh`, evaluate the partial derivative of `F` with respect to `uh[K]`.

# Example

Suppose `uh` and `φh` are FEFunctions with measures `dΩ` and `dΓ_N`.
Then the partial derivative of a function `J` wrt to `φh` is computed via
````
J(u,φ) = ∫(f(u,φ))dΩ + ∫(g(u,φ))dΓ_N
∂J∂φh = ∇(J,[uh,φh],2)
````
where `f` and `g` are user defined.
"""
function Gridap.gradient(F,uh::Vector{<:CellField},K::Int)
  @check 0 < K <= length(uh)
  _f(uk) = F(uh[1:K-1]...,uk,uh[K+1:end]...)
  return Gridap.gradient(_f,uh[K])
end

"""
    Gridap.jacobian(F,uh::Vector{<:CellField},K::Int)

Given a function `F` that returns a DomainContribution when called, and a
vector of `FEFunctions` or `CellField` `uh`, evaluate the Jacobian
`F` with respect to `uh[K]`.
"""
function Gridap.jacobian(F,uh::Vector{<:CellField},K::Int)
  @check 0 < K <= length(uh)
  _f(uk) = F(uh[1:K-1]...,uk,uh[K+1:end]...)
  return Gridap.jacobian(_f,uh[K])
end

# Backwards compat msgs
_msg_v0_3_0 = """
  Inclusion of `U_reg` in the arguments of this constructor has been deprecated
  in v0.3.0 and derivatives are now on the correct tangent space (V_φ). See
  patch notes for futher information.

  This method will be removed in a future release.
"""