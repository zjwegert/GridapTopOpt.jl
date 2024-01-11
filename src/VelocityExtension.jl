"""
  struct VelocityExtension{A,B} end

Wrapper to hold stiffness matrix and cache for
  Hilbertian extension-regularisation procedure.
"""
struct VelocityExtension{A,B}
  K     :: A
  cache :: B
end

function VelocityExtension(
    biform,
    U_reg,
    V_reg;
    assem = SparseMatrixAssembler(U_reg,V_reg),
    ls = LUSolver())   
  ## Assembly
  K  = assemble_matrix(biform,assem,U_reg,V_reg)
  ns = numerical_setup(symbolic_setup(ls,K),K)
  x  = allocate_in_domain(K)
  cache = (ns,x)
  return VelocityExtension(K,cache)
end

"""
  project!(vel_ext::VelocityExtension,dF::AbstractVector) -> dF

Apply Hilbertian extension-regularisation to dF to project
 gradient into a space with additional regularity over the 
 bounding domain. See Allaire et al.
  (10.1016/bs.hna.2020.10.004_978-0-444-64305-6_2021).
"""
function project!(vel_ext::VelocityExtension,dF::AbstractVector)
  ns, x = vel_ext.cache
  fill!(x,zero(eltype(x)))
  solve!(x,ns,dF)
  copy!(dF,x)
  return dF
end

project!(vel_ext::VelocityExtension,dF_vec::Vector{<:AbstractVector}) = map(dF -> project!(vel_ext,dF),dF_vec)