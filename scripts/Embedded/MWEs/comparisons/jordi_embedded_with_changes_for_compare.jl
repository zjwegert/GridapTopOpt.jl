using GridapTopOpt
using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity
import Gridap.Geometry: get_node_coordinates, collect1d

include("../../differentiable_trians.jl")

order = 1
n = 10
N = 8

_model = CartesianDiscreteModel((0,1,0,1),(n,n))
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = ref_model.model
# model = simplexify(_model)

Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)

reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe_scalar)
# ls_evo = HamiltonJacobiEvolution(FirstOrderStencil(2,Float64),model,V_φ,1/(10n),0)

# φh = interpolate(x->max(abs(x[1]-0.5),abs(x[2]-0.5))-0.25,V_φ) # Square
#φh = interpolate(x->((x[1]-0.5)^N+(x[2]-0.5)^N)^(1/N)-0.25,V_φ) # Curved corner example
#φh = interpolate(x->abs(x[1]-0.5)+abs(x[2]-0.5)-0.25-0/n/10,V_φ) # Diamond
# φh = interpolate(x->(1+0.25)abs(x[1]-0.5)+0abs(x[2]-0.5)-0.25,V_φ) # Straight lines with scaling
# φh = interpolate(x->abs(x[1]-0.5)+0abs(x[2]-0.5)-0.25/(1+0.25),V_φ) # Straight lines without scaling
# φh = interpolate(x->tan(-pi/4)*(x[1]-0.5)+(x[2]-0.5),V_φ) # Angled interface
φh = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-0.303,V_φ) # Circle

x_φ = get_free_dof_values(φh)
# @assert ~any(isapprox(0.0;atol=10^-10),x_φ)
if any(isapprox(0.0;atol=10^-10),x_φ)
  println("Issue with lvl set alignment")
  idx = findall(isapprox(0.0;atol=10^-10),x_φ)
  x_φ[idx] .+= 10eps()
end

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

Ωin = Triangulation(cutgeo,PHYSICAL_IN).a
Ωout = Triangulation(cutgeo,PHYSICAL_OUT).a

diff_Ωin = DifferentiableTriangulation(Ωin)
diff_Ωout = DifferentiableTriangulation(Ωout)

dΩin = Measure(diff_Ωin,3*order)
j_in(φ) = ∫(1)dΩin
dj_in = gradient(j_in,φh)
dj_vec_in = assemble_vector(dj_in,V_φ)
norm(dj_vec_in)

dΩout = Measure(diff_Ωout,3*order)
j_out(φ) = ∫(1)dΩout
dj_out = gradient(j_out,φh)
dj_vec_out = -assemble_vector(dj_out,V_φ)
norm(dj_vec_out)

Γ = EmbeddedBoundary(cutgeo)
dΓ = Measure(Γ,3*order)
dj_expected(q) = ∫(-q)dΓ
dj_exp_vec = assemble_vector(dj_expected,V_φ)
norm(dj_exp_vec)

norm(dj_vec_in-dj_exp_vec)
norm(dj_vec_out-dj_exp_vec)

diff_Γ = DifferentiableTriangulation(Γ)
dΓ2 = Measure(diff_Γ,3*order)
jΓ(φ) = ∫(1)dΓ2
djΓ = gradient(jΓ,φh)

############################################################################################

include("../SubFacetSkeletons.jl")

Γ = EmbeddedBoundary(cutgeo)
Λ = Skeleton(Γ)

dΓ = Measure(Γ,3*order)
dΛ = Measure(Λ,2*order)

Γpts = get_cell_points(Γ)
Λpts = get_cell_points(Λ)

function orient(a::VectorValue{2,T},b::VectorValue{2,T}) where T
  if a ⋅ b <= 0
    -a
  else
    a
  end
end

n_∂Ω = get_subfacet_normal_vector(Λ)
n_k = get_ghost_normal_vector(Λ)
#t_S = get_tangent_vector(Λ)
n_S = get_normal_vector(Λ)
# n_S = Operation(orient)(n_S,n_∂Ω.plus)
# _2d_cross(n) = VectorValue(n[2],-n[1]);
# n_1 = CellField(evaluate(get_data(n_k.plus),Point(0)),Λ)
# n_S = Operation(v->(-_2d_cross(v)))(n_1)
m_k = get_conormal_vector(Λ)

fh = interpolate(x->1,V_φ)
∇ˢφ = Operation(abs)(n_S ⋅ ∇(φh).plus)
dJ2(w) = ∫(-n_S ⋅ (jump(fh*m_k) * mean(w) / ∇ˢφ))dΛ
dj2 = assemble_vector(dJ2,V_φ)

diff_Γ = DifferentiableTriangulation(Γ)
dΓ_AD = Measure(diff_Γ,2*order)
J2(φ) = ∫(fh)dΓ_AD
dJ2_AD = gradient(J2,φh)
dj2_AD = assemble_vector(dJ2_AD,V_φ)

############################################################################################

writevtk(
  Ω,"results/test",
  cellfields=["φh"=>φh,"∇φh"=>∇(φh),"dj2"=>FEFunction(V_φ,dj2),"dj2_AD"=>FEFunction(V_φ,dj2_AD)],
  # celldata=["inoutcut"=>bgcell_to_inoutcut]
)

writevtk(
  Λ,
  "results/GammaSkel",
  cellfields=[
    "n_∂Ω.plus" => n_∂Ω.plus,"n_∂Ω.minus" => n_∂Ω.minus,
    # "n_k.plus" => n_k.plus,"n_k.minus" => n_k.minus,
    "n_S" => n_S,
    "m_k.plus" => m_k.plus,"m_k.minus" => m_k.minus,
    "∇ˢφ"=>∇ˢφ,
    "∇φh_Γs_plus"=>∇(φh).plus,"∇φh_Γs_minus"=>∇(φh).minus,
    "jump(fh*m_k)"=>jump(fh*m_k)
  ];
  append=false
)

writevtk(
  Λ.ghost_skeleton,
  "results/GhostSkel",
  cellfields=[
    "n_k.plus" => get_normal_vector(Λ.ghost_skeleton).plus,"n_k.minus" => get_normal_vector(Λ.ghost_skeleton).minus,
  ];
  append=false
)

writevtk(
  Triangulation(cutgeo,PHYSICAL_IN),"results/trian_in"
)
writevtk(
  Triangulation(cutgeo,PHYSICAL_OUT),"results/trian_out"
)
writevtk(
  Γ,"results/gamma"
)