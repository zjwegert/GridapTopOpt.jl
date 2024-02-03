"""
  struct VelocityExtension{A,B}

Wrapper to hold a stiffness matrix and a cache for
the Hilbertian extension-regularisation. See Allaire et al.
(10.1016/bs.hna.2020.10.004_978-0-444-64305-6_2021).

# Properties

- `K::A`
- `cache::B`
"""
struct VelocityExtension{A,B}
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
  solve!(x,ns,dF)
  copy!(dF,x)
  return dF
end

project!(vel_ext::VelocityExtension,dF_vec::Vector{<:AbstractVector}) = broadcast(dF -> project!(vel_ext,dF),dF_vec)