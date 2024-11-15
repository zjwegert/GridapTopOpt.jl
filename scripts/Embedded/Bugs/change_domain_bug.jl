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
Λ = SkeletonTriangulation(Ω)
dΛ = Measure(Λ,2order)

φh = interpolate(x->-cos(4π*x[1])*cos(4π*x[2])-0.2,V_φ)
x_φ = get_free_dof_values(φh)
idx = findall(isapprox(0.0;atol=10^-10),x_φ)
@assert isempty(idx)

Ωs = EmbeddedCollection(model,φh) do cutgeo
  Γ = DifferentiableTriangulation(EmbeddedBoundary(cutgeo))
  (; 
    :Γ    => Γ,
    :dΓ   => Measure(Γ,2*order)
  )
end

vh = zero(V_φ)

∫(vh)Ωs.dΓ
∫(vh*vh)Ωs.dΓ