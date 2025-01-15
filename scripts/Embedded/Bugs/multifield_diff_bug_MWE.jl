using Gridap

order = 1
degree = 2order

model = CartesianDiscreteModel((0,1,0,1),(10,10))
V_φ = TestFESpace(model,ReferenceFE(lagrangian,Float64,1))
φf(x) = sqrt((x[1]-0.5)^2 + (x[2]-0.5)^2) - 0.3
φh = interpolate(φf,V_φ)

V = TestFESpace(model,ReferenceFE(lagrangian,VectorValue{2,Float64},order,space=:P))
Q = TestFESpace(model,ReferenceFE(lagrangian,Float64,order,space=:P))
VQ = MultiFieldFESpace([V,Q])

Λ = SkeletonTriangulation(model)
dΛ = Measure(Λ,degree)
n_Λ = get_normal_vector(Λ)
dΩ = Measure(get_triangulation(model),degree)

res((u,p),(v,q),φ) = ∫(u⋅v + φ)dΩ + ∫(jump(n_Λ ⋅ ∇(u)) ⋅ jump(n_Λ ⋅ ∇(v)))dΛ
# res((u,p),(v,q),φ) = ∫(u⋅v + φ)dΩ + ∫(jump(n_Λ ⋅ ∇(u)) ⋅ jump(n_Λ ⋅ ∇(v)) + 0mean(φ))dΛ # workaround

∇(φ->res(zero(VQ),zero(VQ),φ),φh)