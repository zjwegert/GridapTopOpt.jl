"""
    abstract type AbstractVelocityExtension end

An abstract type for velocity extension structures. Structs that inherit from
  this type must implement the `project!` method.
"""
abstract type AbstractVelocityExtension end

function project!(::AbstractVelocityExtension,::AbstractVector)
  @abstractmethod
end

function Base.show(io::IO,object::AbstractVelocityExtension)
  print(io,"$(nameof(typeof(object)))")
end

"""
    struct IdentityVelocityExtension <: AbstractVelocityExtension

A velocity-extension method that does nothing.
"""
struct IdentityVelocityExtension <: AbstractVelocityExtension end

project!(::IdentityVelocityExtension,dF) = dF
project!(::IdentityVelocityExtension,dF_vec::Vector{<:AbstractVector}) = dF_vec

"""
    struct VelocityExtension{A,B} <: AbstractVelocityExtension

Wrapper to hold a stiffness matrix and a cache for
the Hilbertian extension-regularisation. See Allaire et al. 2022
([link](https://doi.org/10.1016/bs.hna.2020.10.004)).

The Hilbertian extension-regularisation method involves solving an
identification problem over a Hilbert space ``H`` on ``D`` with
inner product ``\\langle\\cdot,\\cdot\\rangle_H``:
*Find* ``g_\\Omega\\in H`` *such that* ``\\langle g_\\Omega,w\\rangle_H
=-J^{\\prime}(\\Omega)(w\\boldsymbol{n})~
\\forall w\\in H.``

This provides two benefits:
 1) It naturally extends the shape sensitivity from ``\\partial\\Omega``
    onto the bounding domain ``D``; and
 2) ensures a descent direction for ``J(\\Omega)`` with additional regularity
    (i.e., ``H`` as opposed to ``L^2(\\partial\\Omega)``)

# Properties

- `K::A`: The discretised inner product over ``H``.
- `cache::B`: Cached objects used for [`project!`](@ref)
"""
struct VelocityExtension{A,B} <: AbstractVelocityExtension
  K     :: A
  cache :: B
end

"""
    VelocityExtension(biform,U_reg,V_reg;assem,ls)

Create an instance of `VelocityExtension` given a bilinear form `biform`,
trial space `U_reg`, and test space `V_reg`.

# Optional

- `assem`: A matrix assembler
- `ls::LinearSolver`: A linear solver
"""
function VelocityExtension(
    biform::Function,
    U_reg::FESpace,
    V_reg::FESpace;
    assem = SparseMatrixAssembler(U_reg,V_reg),
    ls::LinearSolver = LUSolver())
  ## Assembly
  K  = assemble_matrix(biform,assem,U_reg,V_reg)
  ns = numerical_setup(symbolic_setup(ls,K),K)
  x  = allocate_in_domain(K)
  cache = (ns,x)
  return VelocityExtension(K,cache)
end

"""
    project!(vel_ext::VelocityExtension,dF::AbstractVector) -> dF

Project shape derivative `dF` onto a function space described
by the `vel_ext`.
"""
function project!(vel_ext::VelocityExtension,dF::AbstractVector)
  ns, x = vel_ext.cache
  fill!(x,zero(eltype(x)))
  @show length(dF), length(x)
  solve!(x,ns,dF)
  copy!(dF,x)
  return dF
end

project!(vel_ext::VelocityExtension,dF_vec::Vector{<:AbstractVector}) = broadcast(dF -> project!(vel_ext,dF),dF_vec)