using GridapTopOpt
using Gridap

using GridapDistributed, PartitionedArrays

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity

using GridapTopOpt: get_subfacet_normal_vector, get_ghost_normal_vector
using GridapTopOpt: get_conormal_vector, get_tangent_vector

order = 1
n = 16

parts = (2,2)
ranks = DebugArray(LinearIndices((prod(parts),)))

_model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),(n,n))
# _model = CartesianDiscreteModel((0,1,0,1),(n,n))

base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = Gridap.Adaptivity.get_model(ref_model)

reffe = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe)

φh = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-0.5223,V_φ) # Circle
fh = interpolate(x->cos(x[1]*x[2]),V_φ)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

Ωin = Triangulation(cutgeo,PHYSICAL_IN)
Ω_AD = DifferentiableTriangulation(Ωin,V_φ)
dΩ = Measure(Ω_AD,2*order)

Γ = EmbeddedBoundary(cutgeo)
dΓ = Measure(Γ,2*order)

J_bulk(φ) = ∫(fh)dΩ
dJ_bulk_AD = gradient(J_bulk,φh)
dJ_bulk_AD_vec = assemble_vector(dJ_bulk_AD,V_φ)

dJ_bulk_exact(q) = ∫(-fh*q/(norm ∘ (∇(φh))))dΓ
dJ_bulk_exact_vec = assemble_vector(dJ_bulk_exact,V_φ)

norm(dJ_bulk_AD_vec - dJ_bulk_exact_vec) < 1e-10