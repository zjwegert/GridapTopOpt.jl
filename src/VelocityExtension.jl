"""
    abstract type AbstractVelocityExtension end

An abstract type for velocity extension structures. Structs that inherit from
  this type must implement the `project!` method.
"""
abstract type AbstractVelocityExtension end

function project!(::AbstractVelocityExtension,::CellField,V_φ)
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

project!(::IdentityVelocityExtension,dFh::CellField,V_φ) = dFh
project!(::IdentityVelocityExtension,dFh_vec::Vector{<:CellField},V_φ) = dFh_vec

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
- `U_reg::B`: The trial space used for the Hilbertian extension-regularisation.
- `cache::C`: Cached objects used for [`project!`](@ref)
"""
struct VelocityExtension{A,B,C} <: AbstractVelocityExtension
  K     :: A
  U_reg :: B
  cache :: C
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
  b = allocate_in_range(K)
  cache = (ns,x,b)
  return VelocityExtension(K,U_reg,cache)
end

"""
    project!(vel_ext::VelocityExtension,dFh::CellField,V_φ) -> dFh_reg

Project `dFh` onto a function space described by the `vel_ext`.
"""
function project!(vel_ext::VelocityExtension,dFh::CellField,V_φ)
  ns, x, b = vel_ext.cache
  U_reg = vel_ext.U_reg
  interpolate!(dFh,b,U_reg)
  solve!(x,ns,b)
  interpolate!(FEFunction(U_reg,x),get_free_dof_values(dFh),V_φ)
  return dFh
end

project!(vel_ext::VelocityExtension,dFh_vec::Vector{<:CellField},V_φ) = broadcast(dFh -> project!(vel_ext,dFh,V_φ),dFh_vec)