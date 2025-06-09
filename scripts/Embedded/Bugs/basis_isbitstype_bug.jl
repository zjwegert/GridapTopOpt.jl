using Test

using GridapTopOpt
using Gridap, Gridap.Geometry, Gridap.Adaptivity
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using Gridap.Arrays, Gridap.Polynomials, Gridap.Fields, Gridap.TensorValues

using Gridap.Arrays: Operation
using GridapTopOpt: get_conormal_vector,get_subfacet_normal_vector,get_ghost_normal_vector

function generate_model(D,n)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  base_model = UnstructuredDiscreteModel((CartesianDiscreteModel(domain,cell_partition)))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = ref_model.model
  return model
end

D = 2
n = 10
model = generate_model(D,n)
φ = x -> sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-0.5223
f = x -> 1.0
order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe)

φh = interpolate(φ,V_φ)
fh = interpolate(f,V_φ)

# Correction if level set is on top of a node
x_φ = get_free_dof_values(φh)
idx = findall(isapprox(0.0;atol=10^-10),x_φ)
!isempty(idx) && @info "Correcting level values!"
x_φ[idx] .+= 10*eps(eltype(x_φ))

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

# A.1) Volume integral

Ω = Triangulation(cutgeo,PHYSICAL_IN)
Ω_AD = DifferentiableTriangulation(Ω,V_φ)
dΩ = Measure(Ω_AD,2*order)

Γ = EmbeddedBoundary(cutgeo)
dΓ = Measure(Γ,2*order)

J_bulk(φ) = ∫(fh)dΩ
dJ_bulk_AD = gradient(J_bulk,φh)
dJ_bulk_AD_vec = assemble_vector(dJ_bulk_AD,V_φ)

dJ_bulk_exact(q) = ∫(-fh*q/(norm ∘ (∇(φh))))dΓ
dJ_bulk_exact_vec = assemble_vector(dJ_bulk_exact,V_φ)

@test norm(dJ_bulk_AD_vec - dJ_bulk_exact_vec) < 1e-10

# A.2) Volume integral

function Arrays.return_cache(
  fg::Fields.FieldGradientArray{1,Polynomials.MonomialBasis{D,V}},
  x::AbstractVector{<:Point}) where {D,V}
  xi = testitem(x)
  T = gradient_type(V,xi)
  Polynomials._return_cache(fg,x,T,Val(false))
end

function Arrays.evaluate!(
  cache,
  fg::Fields.FieldGradientArray{1,Polynomials.MonomialBasis{D,V}},
  x::AbstractVector{<:Point}) where {D,V}
  Polynomials._evaluate!(cache,fg,x,Val(false))
end

fh = interpolate(x -> x[1]+x[2],V_φ)
g(fh) = ∇(fh)⋅∇(fh)

J_bulk(φ) = ∫(g(fh))dΩ
#J_bulk(φ) = ∫(g(φ))dΩ
dJ_bulk_AD = gradient(J_bulk,φh)
dJ_bulk_AD_vec = assemble_vector(dJ_bulk_AD,V_φ)

dJ_bulk_exact(q) = ∫(-g(fh)*q/(norm ∘ (∇(φh))))dΓ
dJ_bulk_exact_vec = assemble_vector(dJ_bulk_exact,V_φ)

@test norm(dJ_bulk_AD_vec - dJ_bulk_exact_vec) < 1e-10