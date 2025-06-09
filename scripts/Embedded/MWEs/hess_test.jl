using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt
using FiniteDiff

using GridapDistributed,PartitionedArrays,GridapPETSc
using GridapDistributed: DistributedCellField

n = 3
domain = (0,1,0,1)
cell_partition = (n,n)
base_model = UnstructuredDiscreteModel(CartesianDiscreteModel(domain,cell_partition))
ref_model = refine(base_model, refinement_method = "barycentric")
model = Adaptivity.get_model(ref_model)
Ω_bg = Triangulation(model)

# Cut the background model
reffe_scalar = ReferenceFE(lagrangian,Float64,1)
V_φ = TestFESpace(model,reffe_scalar)

lsf(x) = sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-0.35
φh = interpolate(lsf,V_φ)

# Cut
order = 1
degree = 2*(order+1)

Ω_data = EmbeddedCollection(model,φh) do cutgeo,_,_
  Ω_IN = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
  Ω_OUT = Triangulation(cutgeo,PHYSICAL_OUT)
  Ω_CUTIN = Triangulation(cutgeo,CUT_IN)
  Ω_act_IN = Triangulation(cutgeo,ACTIVE)
  (;
    :Ω_IN => Ω_IN,
    :dΩ_IN => Measure(Ω_IN,degree),
    :Ω_OUT => Ω_OUT,
    :Ω_CUTIN => Ω_CUTIN,
    :Ω_act_IN => Ω_act_IN
  )
end

mkpath("./results/2ndOrderDerivs")
writevtk(Ω_data.Ω_IN,"./results/2ndOrderDerivs/Omega_IN")
writevtk(Ω_data.Ω_CUTIN,"./results/2ndOrderDerivs/Omega_CUTIN")
writevtk(Ω_data.Ω_OUT,"./results/2ndOrderDerivs/Omega_OUT")

J(φ) = ∫(1)Ω_data.dΩ_IN

function _J(φ)
  φh = FEFunction(V_φ,φ)
  update_collection!(Ω_data,φh)
  sum(J(φ))
end

grad = gradient(J,φh)
dJ = assemble_vector(grad,V_φ)
hess = hessian(J,φh)
d²J = assemble_matrix(hess,V_φ,V_φ)

x = get_free_dof_values(φh)

dJ_FD = FiniteDiff.finite_difference_gradient(_J,x)
d²J_FD = FiniteDiff.finite_difference_hessian(_J,x)

norm(dJ,Inf)
norm(Matrix(d²J),Inf)

norm(dJ_FD - dJ,Inf)/norm(dJ,Inf)
norm(d²J_FD - Matrix(d²J),Inf)/norm(Matrix(d²J),Inf)