using GridapTopOpt
using Gridap

using GridapDistributed, PartitionedArrays

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using GridapEmbedded.Distributed
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity

using GridapTopOpt: get_subfacet_normal_vector, get_ghost_normal_vector
using GridapTopOpt: get_conormal_vector, get_tangent_vector

order = 1
n = 8
N = 8

parts = (2,2)
ranks = DebugArray(LinearIndices((prod(parts),)))

_model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),(n,n))
#_model = CartesianDiscreteModel((0,1,0,1,0,1),(n,n,n))

base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = Gridap.Adaptivity.get_model(ref_model)

Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)

reffe = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe)

φh = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-0.5223,V_φ) # Circle
fh = interpolate(x->cos(x[1]*x[2]),V_φ)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

Γ = EmbeddedBoundary(cutgeo)
Λ = Skeleton(Γ)
Σ = Boundary(Γ)
Γg = GhostSkeleton(cutgeo)

n_Γ = get_normal_vector(Γ);

n_S_Λ = get_normal_vector(Λ);
m_k_Λ = get_conormal_vector(Λ);
∇ˢφ_Λ = Operation(abs)(n_S_Λ ⋅ ∇(φh).plus);

n_S_Σ = get_normal_vector(Σ);
m_k_Σ = get_conormal_vector(Σ);
∇ˢφ_Σ = Operation(abs)(n_S_Σ ⋅ ∇(φh));

############################################################################################

writevtk(
  Ω,"results/background",
  cellfields=[
    "φh"=>φh,
  ],
  append=false
)
writevtk(
  Triangulation(cutgeo,PHYSICAL_IN),"results/trian_in";
  append=false
)
writevtk(
  Triangulation(cutgeo,PHYSICAL_OUT),"results/trian_out";
  append=false
)
writevtk(
  Γ,"results/gamma";
  append=false
)
writevtk(
  Λ,
  "results/lambda",
  append=false
)
writevtk(
  Σ,
  "results/sigma",
  append=false
)
