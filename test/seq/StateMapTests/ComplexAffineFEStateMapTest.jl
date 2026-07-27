module ComplexAffineFEStateMapTest

using GridapTopOpt
using Gridap, Gridap.MultiField
using FiniteDiff
using Test

model = CartesianDiscreteModel((0,1,0,1),(8,8))
order = 2
reffe = ReferenceFE(lagrangian,Float64,order)
Ω = Triangulation(model)

V_φ = TestFESpace(model,reffe, vector_type=Vector{ComplexF64})
φf(x) = x[1]*x[2] + im
φh = interpolate(φf,V_φ)

V = FESpace(model,reffe;dirichlet_tags="boundary", vector_type=Vector{ComplexF64})
U = TrialFESpace(V,0)

# Define weakforms
dΩ = Measure(Ω,3*order)

a1(u1,v1,φ) = ∫(∇(u1) ⋅ ∇(v1) + im*u1*v1)dΩ
l1(v1,φ) = ∫(φ* φ * v1 + im*φ*v1)dΩ

# Create operator from components
op = AffineFEStateMap(a1,l1,U,V,V_φ)
uh = FEFunction(U,op(φh))

# Compute gradient
f(u1,φ) = ∫(∇(u1)' ⋅ ∇(u1) + φ)dΩ
F = StateParamMap(f,op)

# Check gradient against fdm
using Random
φ = get_free_dof_values(φh)
u = get_free_dof_values(uh)
function g(φ)
  u = op(φ)
  real(F(u,φ))
end
v = randn(MersenneTwister(1),ComplexF64,  length(φ))
h = 1e-6
dfv_fd = real(g(φ + h*v) - g(φ - h*v))/(2h)
dj_ad = GridapTopOpt.val_and_gradient(g, φ)[2][1]
@test dfv_fd ≈ real(dj_ad ⋅ v) # Test directional derivative

fdm_grad = FiniteDiff.finite_difference_gradient(g, get_free_dof_values(φh))
rel_error = norm(dj_ad - fdm_grad, Inf)/norm(fdm_grad,Inf)
@test rel_error < 1e-8

end