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
_f((x,y)) = -cos(4π*x)*cos(4π*y)-0.5
φh = interpolate(_f,V_φ)

V_df = TestFESpace(Ω,reffe_scalar,dirichlet_tags=["boundary"])
U_df = TrialFESpace(V_df,1)
dΩ = Measure(Ω,2order)

a = 1e-4

C = 2;
χ(φ) = φ <= 0 ? C : 0;
A(u,v) = ∫( a*∇(v)⋅∇(u) + u*v)dΩ
L(v) = ∫( (χ ∘ φh )*v )dΩ

op = AffineFEOperator(A,L,U_df,V_df)
uah = solve(op)
ua = get_free_dof_values(uah)

## Local SDF
function compute_local_sdf(φ,u)
  if φ > 0
    -sqrt(a)*log(u)
  else
    sqrt(a)*log(C-u)
  end
end

uah_V_φ = interpolate(uah,V_φ)
sdf = map(compute_local_sdf,get_free_dof_values(φh),get_free_dof_values(uah_V_φ))
sdfh = FEFunction(V_φ,sdf)

writevtk(Ω,"results/yamada_sdf",cellfields=["φh"=>φh,"χ"=>χ ∘ φh,"uah"=>uah,"sdfh"=>sdfh,"|∇(sdfh)|"=>(norm ∘ ∇(sdfh))])