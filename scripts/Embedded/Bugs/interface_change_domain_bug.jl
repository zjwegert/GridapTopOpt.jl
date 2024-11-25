using Test

using GridapTopOpt
using Gridap

using GridapDistributed, PartitionedArrays

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity

using GridapTopOpt: get_subfacet_normal_vector, get_ghost_normal_vector
using GridapTopOpt: get_conormal_vector, get_tangent_vector

using GridapDistributed: DistributedTriangulation, DistributedDomainContribution

order = 1
n = 16

parts = (2,2)
ranks = DebugArray(LinearIndices((prod(parts),)))

# _model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),(n,n))
_model = CartesianDiscreteModel((0,1,0,1),(n,n))

base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = Gridap.Adaptivity.get_model(ref_model)

reffe = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe)

φh = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-0.5223,V_φ) # Circle
fh = interpolate(x->cos(x[1]*x[2]),V_φ)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

Γ = EmbeddedBoundary(cutgeo)
Γ_AD = DifferentiableTriangulation(Γ,V_φ)
dΓ_AD = Measure(Γ_AD,2*order)

J_int2(φ) = ∫(g(fh))dΓ_AD
dJ_int_AD2 = gradient(J_int2,φh)
dJ_int_AD_vec2 = assemble_vector(dJ_int_AD2,V_φ)