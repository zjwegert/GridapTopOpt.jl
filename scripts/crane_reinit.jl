using Gridap, Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt, GridapSolvers

_model = CartesianDiscreteModel((0,1,0,1),(50,50))
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
ref_model = refine(ref_model)
ref_model = refine(ref_model)
model = get_model(ref_model)
Ω = Triangulation(model)

order = 1
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe_scalar)

## Level-set function
f1((x,y)) = -cos(4π*x)*cos(4π*y)-0.5-1e-5
f2((x,y)) = (x-0.5)^2+(y-0.5)^2-0.135^2-1e-5
φh = interpolate(x->max(f1(x),-f2(x)),V_φ)

h = maximum(get_element_diameters(model))
dΩ = Measure(Ω,2order)
∂Ω = BoundaryTriangulation(model,tags="boundary")
d∂Ω = Measure(∂Ω,2order)
n = get_normal_vector(∂Ω)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

Γ = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
Γ_n = get_normal_vector(Γ)
dΓ = Measure(Γ,2order)

# Step 1
Vx = TestFESpace(model,ReferenceFE(lagrangian,VectorValue{2,Float64},order))
t = h^2
A(u,v,φ) = ∫(u ⋅ v + t*(∇(u) ⊙ ∇(v)))dΩ
L(v,φ) = ∫(Γ_n ⋅ v)dΓ
φ_to_x = AffineFEStateMap(A,L,Vx,Vx,V_φ)

# Step 2
_y(x) = -x/norm(x)
x_to_y = AffineFEStateMap((u,v,x)->∫(u⋅v)dΩ,(v,x)->∫(v⋅(_y∘x))dΩ,Vx,Vx,Vx)

# Step 3
γd = 10.0
A2(u,v,y) = ∫(∇(u)⋅∇(v))dΩ + ∫((γd/h)*v*u)dΓ
L2(v,y) = ∫(v*(∇ ⋅ y))dΩ + ∫(-(n⋅y)*v)d∂Ω
y_to_sdf = AffineFEStateMap(A2,L2,V_φ,V_φ,Vx)

function φ_to_sdf(φ)
  x   = φ_to_x(φ)
  y   = x_to_y(x)
  return y_to_sdf(y)
end

sdfh = FEFunction(V_φ,φ_to_sdf(get_free_dof_values(φh)))

dmin = 0.1
_op(sdf) = -sdf^2*max(sdf+dmin/2,0)^2
_E(sdf,φ) = ∫(_op ∘ sdf)dΩ
E = GridapTopOpt.StateParamMap(_E,V_φ,V_φ,SparseMatrixAssembler(V_φ,V_φ),SparseMatrixAssembler(V_φ,V_φ))

function φ_to_E(φ)
  sdf = φ_to_sdf(φ)
  E(sdf,φ)
end

val, grad = GridapTopOpt.val_and_jacobian(φ_to_E,get_free_dof_values(φh))

writevtk(Ω,"results/crane_sdf",cellfields=["φh"=>φh,"sdfh"=>sdfh,"|∇(sdfh)|"=>(norm ∘ ∇(sdfh)),"grad"=>FEFunction(V_φ,grad[1][1])])