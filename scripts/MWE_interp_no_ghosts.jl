using Gridap, Gridap.FESpaces, Gridap.Geometry, Gridap.CellData, Gridap.Fields
using GridapDistributed, PartitionedArrays

using GridapDistributed: allocate_in_domain, allocate_in_range

mesh_parts = (2,2)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(mesh_parts),)))
end

model = CartesianDiscreteModel(ranks,mesh_parts,(0,1,0,1),(10,10))
Ω = Triangulation(model)
dΩ = Measure(Ω,2)

V_φ = TestFESpace(model,ReferenceFE(lagrangian,Float64,1))
φh = interpolate(x->x[1],V_φ)
V_reg = TestFESpace(model,ReferenceFE(lagrangian,Float64,1);dirichlet_tags=["boundary"])
U_reg = TrialFESpace(V_reg)

K = assemble_matrix((u,v)->∫(∇(u)⋅∇(v)+u*v)dΩ,U_reg,V_reg)
ns = numerical_setup(symbolic_setup(LUSolver(),K),K)
x = allocate_in_domain(K)

b = allocate_in_range(K)
dF = assemble_vector(∇(φ->∫(φ)dΩ,φh),V_φ)
dFh = FEFunction(V_φ,dF)

# interpolate!(dFh,b,U_reg)
# solve!(x,ns,b)
# interpolate!(FEFunction(U_reg,x),get_free_dof_values(dFh),V_φ)

dFh_Ureg = interpolate(FEFunction(V_φ,get_free_dof_values(dFh)),U_reg)
copy!(b,get_free_dof_values(dFh_Ureg))
solve!(x,ns,b)
xh_V_φ = interpolate(FEFunction(U_reg,x),V_φ)
copy!(get_free_dof_values(dFh),get_free_dof_values(xh_V_φ))

writevtk(Ω,"results/MWE_interp_no_ghosts",cellfields=["xh_V_φ"=>xh_V_φ,"dFh"=>dFh])