using GridapTopOpt
using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField,
  Gridap.TensorValues, Gridap.FESpaces, Gridap.Arrays, Gridap.Helpers
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapDistributed,PartitionedArrays,GridapPETSc

np = (2,1)
ranks = DebugArray([1,2])

model = UnstructuredDiscreteModel(CartesianDiscreteModel(ranks,np,(0,1,0,1),(4,4)))

reffe_m = ReferenceFE(lagrangian,Float64,1)
M = FESpace(model,reffe_m)

ls(x) = ifelse(x[1] > 0.8,-1.0,1.0)
φh = interpolate(ls,M)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)
cutgeo_facets = cut_facets(model,geo)

order = 1
degree = 2order
Ω_act = Triangulation(model)
dΩ_act = Measure(Ω_act,degree)
# Ω = EmbeddedCollection(model,φh) do cutgeo,cutgeo_facets,_φh
#   Ωs = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
#   Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
#   Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
#   Γg = GhostSkeleton(cutgeo)
#   Ω_act_s = Triangulation(cutgeo,ACTIVE)
#   Ω_act_f = Triangulation(cutgeo,ACTIVE_OUT)
#   Γi = SkeletonTriangulation(cutgeo_facets,ACTIVE_OUT)
#   (;
#     :Ωs       => Ωs,
#     :dΩs      => Measure(Ωs,degree),
#     :Ωf       => Ωf,
#     :dΩf      => Measure(Ωf,degree),
#     :Γg       => Γg,
#     :dΓg      => Measure(Γg,degree),
#     :n_Γg     => get_normal_vector(Γg),
#     :Γ        => Γ,
#     :dΓ       => Measure(Γ,degree),
#     :n_Γ      => get_normal_vector(Γ),
#     :Ω_act_s  => Ω_act_s,
#     :dΩ_act_s => Measure(Ω_act_s,degree),
#     :Ω_act_f  => Ω_act_f,
#     :dΩ_act_f => Measure(Ω_act_f,degree),
#     :Γi       => Γi,
#     :dΓi      => Measure(Γi,degree),
#     :n_Γi     => get_normal_vector(Γi),
#   )
# end

Ωs = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),M)
Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),M)
Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),M)

# Ωs = Triangulation(cutgeo,PHYSICAL)
# Ωf = Triangulation(cutgeo,PHYSICAL_OUT)
# Γ  = EmbeddedBoundary(cutgeo)
Γg = GhostSkeleton(cutgeo)
Ω_act_s = Triangulation(cutgeo,ACTIVE)
Ω_act_f = Triangulation(cutgeo,ACTIVE_OUT)
Γi = SkeletonTriangulation(cutgeo_facets,ACTIVE_OUT)
dΩs      = Measure(Ωs,degree)
dΩf      = Measure(Ωf,degree)
dΓg      = Measure(Γg,degree)
n_Γg     = get_normal_vector(Γg)
dΓ       = Measure(Γ,degree)
n_Γ      = get_normal_vector(Γ);
dΩ_act_s = Measure(Ω_act_s,degree)
dΩ_act_f = Measure(Ω_act_f,degree)
dΓi      = Measure(Γi,degree)
n_Γi     = get_normal_vector(Γi)

# Setup spaces
reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
reffe_d = ReferenceFE(lagrangian,VectorValue{2,Float64},order)

# Test spaces
V = TestFESpace(Ω_act_f,reffe_u,conformity=:H1)
Q = TestFESpace(Ω_act_f,reffe_p,conformity=:L2)
T = TestFESpace(Ω_act_s,reffe_d,conformity=:H1)

# Trial spaces
U = TrialFESpace(V)
P = TrialFESpace(Q)
R = TrialFESpace(T)

# Multifield spaces
UP = MultiFieldFESpace([U,P])
VQ = MultiFieldFESpace([V,Q])

### Weak form
## Fluid
# Properties
μ = 1.0

# Stabilization parameters
α_Nu = 100
α_u  = 0.1
α_p  = 0.25

# Stabilization functions
hₕ = CellField(1,Ω_act)

γ_Nu(h) = α_Nu*μ/h
γ_u(h) = α_u*μ*h
γ_p(h) = α_p*h/μ
k_p    = 1.0
γ_Nu_h = γ_Nu ∘ hₕ
γ_u_h = mean(γ_u ∘ hₕ)
γ_p_h = mean(γ_p ∘ hₕ)

# Terms
σf_n(u,p,n) = μ*∇(u) ⋅ n - p*n
a_Ω(∇u,∇v) = μ*(∇u ⊙ ∇v)
b_Ω(div_v,p) = -p*(div_v)
a_Γ(u,∇u,v,∇v,n) = - μ*n⋅(∇u ⋅ v + ∇v⋅ u) + γ_Nu_h*(u⋅v)
b_Γ(v,p,n) = (n⋅v)*p
ju(∇u,∇v) = γ_u_h*(jump(n_Γg ⋅ ∇u) ⋅ jump(n_Γg ⋅ ∇v))
jp(p,q) = γ_p_h*(jump(p) * jump(q))

function a_fluid((),(u,p),(v,q),φ)
  ∇u = ∇(u); ∇v = ∇(v);
  div_u = ∇⋅u; div_v = ∇⋅v
  n_Γ = -get_normal_vector(Γ)
  return ∫(a_Ω(∇u,∇v) + b_Ω(div_v,p) + b_Ω(div_u,q))dΩf +
    ∫(a_Γ(u,∇u,v,∇v,n_Γ) + b_Γ(v,p,n_Γ) + b_Γ(u,q,n_Γ))dΓ +
    ∫(ju(∇u,∇v))dΓg - ∫(jp(p,q))dΓi
end

l_fluid((),(v,q),φ) =  ∫(0q)dΩf

xdh = zero(UP);

λᵀ1_∂R1∂φ = ∇(φ -> a_fluid((),xdh,xdh,φ) - l_fluid((),xdh,φ),φh)
vecdata = collect_cell_vector(M,λᵀ1_∂R1∂φ)
assem_deriv = SparseMatrixAssembler(M,M)
Σ_λᵀs_∂Rs∂φ = allocate_vector(assem_deriv,vecdata)


######################################################################

# xdh = zero(UP);

# λᵀ1_∂R1∂φ = ∇(x -> a_fluid((),x,xdh,φh) - l_fluid((),xdh,φh),xdh);
# vecdata = collect_cell_vector(UP,λᵀ1_∂R1∂φ);
# assem_deriv = SparseMatrixAssembler(UP,UP);
# Σ_λᵀs_∂Rs∂φ = allocate_vector(assem_deriv,vecdata);

#####################

## Structure
_I = one(SymTensorValue{2,Float64})
# Material parameters
function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end
λs, μs = lame_parameters(0.1,0.05)
# Stabilization
α_Gd = 1e-3
k_d = 1.0
γ_Gd(h) = α_Gd*(λs + μs)*h^3
γ_Gd_h = mean(γ_Gd ∘ hₕ)
# Terms
σ(ε) = λs*tr(ε)*_I + 2*μs*ε
a_s_Ω(d,s) = ε(s) ⊙ (σ ∘ ε(d)) # Elasticity
j_s_k(d,s) = γ_Gd_h*(jump(n_Γg ⋅ ∇(s)) ⋅ jump(n_Γg ⋅ ∇(d)))
v_s_ψ(d,s) = (k_d*ψ_s)*(d⋅s) # Isolated volume term

function a_solid(((u,p),),d,s,φ)
  return ∫(a_s_Ω(d,s))dΩs +
    ∫(j_s_k(d,s))dΓg
end
function l_solid(((u,p),),s,φ)
  n = -get_normal_vector(Γ)
  return ∫(-σf_n(u,p,n) ⋅ s)dΓ
end

d0h = zero(R);

λᵀ2_∂R2∂φ = ∇(φ -> a_solid((xdh,),d0h,d0h,φ) - l_solid((xdh,),d0h,φ),φh);
vecdata2 = collect_cell_vector(M,λᵀ2_∂R2∂φ);
Σ_λᵀ2_∂R2∂φ = allocate_vector(assem_deriv,vecdata2);

# λᵀ2_∂R2∂φ = ∇(x -> a_solid((xdh,),d0h,x,φh) - l_solid((xdh,),x,φh),d0h);
# vecdata2 = collect_cell_vector(R,λᵀ2_∂R2∂φ);
# assem_deriv = SparseMatrixAssembler(R,R);
# Σ_λᵀ2_∂R2∂φ = allocate_vector(assem_deriv,vecdata2);