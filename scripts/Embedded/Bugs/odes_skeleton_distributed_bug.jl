using Test

using GridapTopOpt
using Gridap, Gridap.Geometry, Gridap.Adaptivity
using GridapEmbedded

using GridapDistributed, PartitionedArrays

parts = (2,2)
ranks = DebugArray(LinearIndices((prod(parts),)))

n = 200
order = 1

_model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),(n,n))
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = Adaptivity.get_model(ref_model)
el_Δ = get_el_Δ(_model)
h = maximum(el_Δ)
h_refine = maximum(el_Δ)/2

Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe_scalar)
Γg = SkeletonTriangulation(get_triangulation(V_φ))
dΓg = Measure(Γg,2order)
n_Γg = get_normal_vector(Γg)

φh = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25,V_φ)

velh = interpolate(x->-1,V_φ)

γg = 0.1
dt = GridapTopOpt._compute_Δt(h,0.1,get_free_dof_values(velh))
ϵ = 1e-20

ode_ls = LUSolver()
ode_nl = NLSolver(ode_ls, show_trace=false, method=:newton, iterations=10)
oode_solver = RungeKutta(ode_nl, ode_ls, 0.1, :DIRK_CrankNicolson_2_2)

β = velh*∇(φh)/(ϵ + norm ∘ ∇(φh))
stiffness(t,u,v) = ∫((β ⋅ ∇(u)) * v)dΩ + ∫(γg*h^2*jump(∇(u) ⋅ n_Γg)*jump(∇(v) ⋅ n_Γg))dΓg
mass(t, ∂ₜu, v) = ∫(∂ₜu * v)dΩ
forcing(t,v) = ∫(0v)dΩ
Ut_φ = TransientTrialFESpace(V_φ)
ode_op = TransientLinearFEOperator((stiffness,mass),forcing,Ut_φ,V_φ;
  constant_forms=(false,true))
ode_sol = solve(oode_solver,ode_op,0.0,dt*10,φh)

march = Base.iterate(ode_sol)