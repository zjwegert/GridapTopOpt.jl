"""
    abstract type AbstractVelocityExtension end

An abstract type for velocity extension structures. Structs that inherit from
  this type must implement the `project!` method.
"""
abstract type AbstractVelocityExtension end

function project!(::AbstractVelocityExtension,::AbstractVector,V_φ,uhd)
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

project!(::IdentityVelocityExtension,dF::AbstractVector,V_φ,uhd) = dF
project!(::IdentityVelocityExtension,dF_vec::Vector{<:AbstractVector},V_φ,uhd) = dF_vec

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

## TODO: In future, I want the cache to be an output of project!, like
#        the rest of Gridap.

"""
    VelocityExtension(biform,U_reg,V_reg;assem,ls)

Create an instance of `VelocityExtension` given a bilinear form `biform`,
trial space `U_reg`, and test space `V_reg`.

# Optional

- `assem`: A matrix assembler
- `ls::LinearSolver`: A linear solver
"""
function VelocityExtension(
    biform :: Function,
    U_reg  :: FESpace,
    V_reg  :: FESpace;
    assem = SparseMatrixAssembler(U_reg,V_reg),
    ls     :: LinearSolver = LUSolver())
  ## Assembly
  K      = assemble_matrix(biform,assem,U_reg,V_reg)
  ns     = numerical_setup(symbolic_setup(ls,K),K)
  x      = allocate_in_domain(K)
  b      = allocate_in_range(K)
  bh_tmp = zero(U_reg)
  cache  = (ns,x,b,bh_tmp)
  return VelocityExtension(K,U_reg,cache)
end

"""
    project!(vel_ext::VelocityExtension,dF::AbstractVector,V_φ,uhd) -> dF_reg::Vector

Project `dFh` onto a function space described by the `vel_ext`.

Note:
- We expect that `dF` is a vector resulting from assembly on V_φ.
- uhd should be an FEFunction on V_φ.
"""
function project!(vel_ext::VelocityExtension,dF::AbstractVector,V_φ,uhd)
  ns, x, b, bh_tmp = vel_ext.cache
  U_reg = vel_ext.U_reg

  _interpolate_onto_rhs!(b,U_reg,bh_tmp,dF,V_φ)
  solve!(x,ns,b)
  _interpolate_onto_rhs!(dF,V_φ,uhd,x,U_reg)

  return dF
end

project!(vel_ext::VelocityExtension,dFh_vec::Vector{<:AbstractVector},V_φ,uhd) = broadcast(dFh -> project!(vel_ext,dFh,V_φ,uhd),dFh_vec)

### Functionality for interpolating between
# - RHS vectors -> RHS vectors
# - DOF vectors -> RHS vectors

function _map_rhs_to_dofs!(x,V_dof_indies,rhs,rhs_indices)
  rhs_to_dof = GridapDistributed.find_local_to_local_map(rhs_indices,V_dof_indies)
  x[rhs_to_dof] = rhs
end

function _map_dofs_to_rhs!(rhs,rhs_indices,x,V_dof_indies)
  rhs_to_dof = GridapDistributed.find_local_to_local_map(rhs_indices,V_dof_indies)
  copyto!(rhs,x[rhs_to_dof])
end

# In-place interpolation of source RHS vector (`src`) on an FESpace (`Q`) onto a
# destination RHS vector (`dst`) on another FESpace (`V`). `uhdV` and `uhdQ` should
# be "true" FEFunctions (with the correct ghosts) on `V` and `Q` respectively.
#
# This is required as the interpolate functionality in GridapDistributed
# expects ghosts to be present. However, RHS vectors do not have ghosts.
#
# To solve this problem, we map `src` onto an FEFunction `uhdQ`
# that has the correct ghosts. We then interpolate this onto
# another FEFunction `uhdV` that has the correct ghosts for `V`.
# Finally, we map the result back onto `dst`.
function _interpolate_rhs_onto_rhs!(dst::PVector,V,uhdV,src::PVector,Q,uhdQ)
  # Map src onto uhdQ, then interpolate uhdQ onto uhdV
  Q_gids  = get_free_dof_ids(Q)
  src_gids = src.index_partition
  src_dofed = get_free_dof_values(uhdQ)
  map(_map_rhs_to_dofs!,local_views(src_dofed),local_views(Q_gids),local_views(src),src_gids)
  consistent!(src_dofed) |> fetch
  interpolate!(uhdQ,get_free_dof_values(uhdV),V)
  # Map uhdV onto dst
  V_gids  = get_free_dof_ids(V)
  dst_gids = dst.index_partition
  dst_dofed = get_free_dof_values(uhdV)
  map(_map_dofs_to_rhs!,local_views(dst),dst_gids,local_views(dst_dofed),local_views(V_gids))

  return dst
end

# Serial implementation - already works in Gridap
_interpolate_rhs_onto_rhs!(dst,V,uhdV,src,Q,uhdQ) = interpolate!(FEFunction(Q,src),dst,V)

# Similar to the above except now we are interpolating a DOF vector (`src`)
# (correct ghost information) on an FESpace (`Q`) onto a RHS vector (`dst`)
# on another FESpace (`V`).
function _interpolate_onto_rhs!(dst::PVector,V,uhdV,src::PVector,Q)
  # Interpolate src onto uhdV, we can do this because src has ghosts
  V_gids  = get_free_dof_ids(V)
  dst_gids = dst.index_partition
  dst_dofed = get_free_dof_values(uhdV)
  interpolate!(FEFunction(Q,src),dst_dofed,V)
  # Map dst_dofed onto dst (rhs)
  map(_map_dofs_to_rhs!,local_views(dst),local_views(dst_gids),local_views(dst_dofed),local_views(V_gids))
end

# Serial implementation - already works in Gridap
_interpolate_onto_rhs!(dst,V,uhdV,src,Q) = interpolate!(FEFunction(Q,src),dst,V)