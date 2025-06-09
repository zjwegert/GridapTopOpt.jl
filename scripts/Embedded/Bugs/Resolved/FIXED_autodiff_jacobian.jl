using GridapTopOpt
using Gridap, Gridap.Geometry, Gridap.Adaptivity
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using Gridap.Arrays, Gridap.Polynomials, Gridap.Fields, Gridap.TensorValues

using Gridap.Arrays: Operation
using GridapTopOpt: get_conormal_vector,get_subfacet_normal_vector,get_ghost_normal_vector

function generate_model(D,n)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  cart_mod = CartesianDiscreteModel(domain,cell_partition)
  base_model = UnstructuredDiscreteModel(cart_mod)
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = ref_model.model
  desc = get_cartesian_descriptor(cart_mod)
  return model,maximum(desc.sizes)
end

D = 2
n = 10
model,h = generate_model(D,n)
φ = x -> sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-0.5223
order = 1

reffe = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe)

φh = interpolate(φ,V_φ)

# Correction if level set is on top of a node
x_φ = get_free_dof_values(φh)
idx = findall(isapprox(0.0;atol=10^-10),x_φ)
!isempty(idx) && @info "Correcting level values!"
x_φ[idx] .+= 10*eps(eltype(x_φ))

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

# Error 1
Γ = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
dΓ = Measure(Γ,2get_order(V_φ))
dj(u,du,v) = jacobian(u->∫(u*v)dΓ,u)
dj(φh, get_trial_fe_basis(V_φ), get_fe_basis(V_φ))

# Error 2
Λ = SkeletonTriangulation(Triangulation(model))
dΛ = Measure(Λ,2*order)
dj(u,du,v) = jacobian(u->∫(jump(∇(u)) ⋅ jump(∇(v)))dΛ,u)
dj(φh, get_trial_fe_basis(V_φ), get_fe_basis(V_φ))
