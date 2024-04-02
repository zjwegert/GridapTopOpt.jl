"""
    struct VelocityExtension{A,B}

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
struct VelocityExtension{A,B,C,D}
  K     :: A
  U_reg :: B
  U_φ   :: C
  rhs   :: D
  cache :: E
end

"""
    VelocityExtension(lhs,rhs,U_reg,V_reg,U_φ;assem,ls)
  
# Arguments
- `lhs`: The bilinear form of the identification problem.
- `rhs`: The right-hand side of the identification problem.
- `U_reg`: The Hilbertian extension-regularisation spaces.
- `U_φ`: The non-regularised shape sensitivity space.

# Optional

- `assem`: A matrix assembler
- `ls::LinearSolver`: A linear solver
"""
function VelocityExtension(
    lhs::Function,
    rhs::Function,
    U_reg::FESpace,
    U_φ::FESpace;
    assem = SparseMatrixAssembler(U_reg,U_reg),
    ls::LinearSolver = LUSolver()
)
  ## Assembly
  K  = assemble_matrix(lhs,assem,U_reg,U_reg)
  cache = velocity_extension_caches(K,ls)
  return VelocityExtension(K,U_reg,U_φ,rhs,cache)
end

function velocity_extension_caches(K::AbstractMatrix,ls)
  ns = numerical_setup(symbolic_setup(ls,K),K)
  b  = allocate_in_range(K); fill!(b,zero(eltype(b)))
  return (ns,b)
end

function velocity_extension_caches(K::PSparseMatrix,ls)
  ns = numerical_setup(symbolic_setup(ls,K),K)
  x  = allocate_in_domain(K); fill!(x,zero(eltype(x)))
  b  = allocate_in_range(K); fill!(b,zero(eltype(b)))
  return (ns,x,b)
end

function Base.show(io::IO,object::VelocityExtension)
  print(io,"$(nameof(typeof(object)))")
end

"""
    project!(φh_reg::FEFunction,P::VelocityExtension,φh::FEFunction)
    project!(φ_reg::AbstractVector,P::VelocityExtension,φ::AbstractVector)
  
  Projects the shape sensitivity `φh` into the Hilbertian extension-regularisation
  space `U_reg`. Stores the result in `φh_reg`.
"""
function project!(φh_reg,P::VelocityExtension,φh)
  φ_reg = get_free_dof_values(φh_reg)
  project!(φ_reg,P,φh)
  return φh_reg
end

function project!(φ_reg::AbstractVector,P::VelocityExtension,φ::AbstractVector)
  φh = FEFunction(P.U_φ,φ)
  project!(φ_reg,P,φh)
  return φ_reg
end

function project!(φ_reg::AbstractVector,P::VelocityExtension{<:AbstractMatrix},φh)
  ns, b = P.cache
  assemble_vector!(v->P.rhs(φh,v),b,assem,V_reg)
  solve!(φ_reg,ns,b)
  return φ_reg
end

function project!(φ_reg::PVector,P::VelocityExtension{<:PsparseMatrix},φh)
  ns, x, b = P.cache
  assemble_vector!(v->P.rhs(φh,v),b,assem,V_reg)
  solve!(x,ns,b)
  copy!(φ_reg,x)
  consistent!(φ_reg)
  return φ_reg
end

"""
    project(P::VelocityExtension,φh::FEFunction) -> FEFunction
    project(P::VelocityExtension,φ::AbstractVector) -> AbstractVector
  
  Projects the shape sensitivity `φh` into the Hilbertian extension-regularisation
  space `U_reg`. Allocates output.
"""
function project(P::VelocityExtension,φh)
  φh_reg = zero(P.U_reg)
  return project!(φh_reg,P,φh)
end

function project(P::VelocityExtension,φ::AbstractVector)
  φ_reg = allocate_in_domain(P.K); fill!(φ_reg,zero(eltype(φ_reg)))
  return project!(φ_reg,P,φ)
end