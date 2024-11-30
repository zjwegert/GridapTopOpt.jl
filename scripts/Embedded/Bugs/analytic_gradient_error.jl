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

g(fh) = ∇(fh)⋅∇(fh)

J_int2(φ) = ∫(g(fh))dΓ_AD
dJ_int_AD2 = gradient(J_int2,φh)
dJ_int_AD_vec2 = assemble_vector(dJ_int_AD2,V_φ)

Λ = Skeleton(Γ)
Σ = Boundary(Γ)
dΓ = Measure(Γ,2*order)
dΛ = Measure(Λ,2*order)
dΣ = Measure(Σ,2*order)

n_Γ = get_normal_vector(Γ)
n_S_Λ = get_normal_vector(Λ)
m_k_Λ = get_conormal_vector(Λ)
∇ˢφ_Λ = Operation(abs)(n_S_Λ ⋅ ∇(φh).plus)
n_S_Σ = get_normal_vector(Σ)
m_k_Σ = get_conormal_vector(Σ)
∇ˢφ_Σ = Operation(abs)(n_S_Σ ⋅ ∇(φh))

# TODO: This currently fails with
# `ERROR: This function belongs to an interface definition and cannot be used.`
dJ_int_exact2(w) = ∫((-n_Γ⋅∇(g(fh)))*w/(norm ∘ (∇(φh))))dΓ +
                  ∫(-n_S_Λ ⋅ (jump(g(fh)*m_k_Λ) * mean(w) / ∇ˢφ_Λ))dΛ +
                  ∫(-n_S_Σ ⋅ (g(fh)*m_k_Σ * w / ∇ˢφ_Σ))dΣ
dJ_int_exact_vec2 = assemble_vector(dJ_int_exact2,V_φ)

Ω = Triangulation(model)
x = get_cell_points(Ω)

using Gridap.CellData: get_data

∇g(fh) = ∇∇(fh)⋅∇(fh) + ∇(fh)⋅∇∇(fh)
∇g2(∇∇f,∇f) = ∇∇f⋅∇f + ∇f⋅∇∇f

h = g(fh)
∇h = ∇(h)
∇h_exact = ∇g(fh)
∇h_exact2 = ∇g2∘(∇∇(fh),∇(fh))

h(x)
∇h(x)
∇h_exact(x)
∇h_exact2(x)
