struct VelocityExtension{A}
  cache :: A
end

function VelocityExtension(
    biform,
    U_reg,
    V_reg,
    dΩ;
    assem = SparseMatrixAssembler(U_reg,V_reg),
    ls = LUSolver())   
  ## Assembly
  K  = assemble_matrix((U,V) -> biform(U,V,dΩ),assem,U_reg,V_reg)
  ns = numerical_setup(symbolic_setup(ls,K),K)
  x  = allocate_in_domain(K)
  cache = (ns,x)
  return VelocityExtension(cache)
end

function project!(vel_ext::VelocityExtension,dF::AbstractVector)
  ns, x = vel_ext.cache
  solve!(x,ns,dF)
  copy!(dF,x)
  return dF
end

project!(vel_ext::VelocityExtension,dF_vec::Vector{<:AbstractVector}) = map(dF -> project!(vel_ext,dF),dF_vec)