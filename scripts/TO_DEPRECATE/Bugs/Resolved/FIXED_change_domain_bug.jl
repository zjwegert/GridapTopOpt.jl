using Gridap,GridapTopOpt
using Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters

n = 10
order = 1
_model = CartesianDiscreteModel((0,1,0,1),(n,n))
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = ref_model.model

Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)
V_φ = TestFESpace(model,ReferenceFE(lagrangian,Float64,order))
φh = interpolate(x->-cos(4π*x[1])*cos(4π*x[2])-0.2,V_φ)
vh = zero(V_φ)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

Γ = EmbeddedBoundary(cutgeo)
dΓ = Measure(Γ,2order)
∫(vh)dΓ
∫(2vh)dΓ


_Γ = EmbeddedBoundary(cutgeo)
Γ = DifferentiableTriangulation(_Γ,V_φ)
dΓ = Measure(Γ,2order)
∫(vh)dΓ
∫(2vh)dΓ

cf = CellField(1,_Γ)
change_domain(cf,Γ,ReferenceDomain())
