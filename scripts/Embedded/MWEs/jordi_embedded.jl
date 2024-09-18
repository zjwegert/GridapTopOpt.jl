using GridapTopOpt

using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData
import Gridap.Geometry: get_node_coordinates, collect1d

include("../differentiable_trians.jl")

order = 1
n = 20
N = 8

model = CartesianDiscreteModel((0,1,0,1),(n,n))
model = simplexify(model)
Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)

reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe_scalar)
# ls_evo = HamiltonJacobiEvolution(FirstOrderStencil(2,Float64),model,V_φ,1/(10n),0)

# φh = interpolate(x->max(abs(x[1]-0.5),abs(x[2]-0.5))-0.25,V_φ) # Square
# φh = interpolate(x->((x[1]-0.5)^N+(x[2]-0.5)^N)^(1/N)-0.25,V_φ) # Curved corner example
φh = interpolate(x->abs(x[1]-0.5)+abs(x[2]-0.5)-0.25-0/n/10,V_φ) # Diamond
# φh = interpolate(x->(1+0.25)abs(x[1]-0.5)+0abs(x[2]-0.5)-0.25,V_φ) # Straight lines with scaling
# φh = interpolate(x->abs(x[1]-0.5)+0abs(x[2]-0.5)-0.25/(1+0.25),V_φ) # Straight lines without scaling
# φh = interpolate(x->tan(-pi/4)*(x[1]-0.5)+(x[2]-0.5),V_φ) # Angled interface
# φh = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-0.303,V_φ) # Circle

x_φ = get_free_dof_values(φh)
# @assert ~any(isapprox(0.0;atol=10^-10),x_φ)
if any(isapprox(0.0;atol=10^-10),x_φ)
  println("Issue with lvl set alignment")
  idx = findall(isapprox(0.0;atol=10^-10),x_φ)
  x_φ[idx] .+= 10eps()
end
any(x -> x < 0,x_φ)
any(x -> x > 0,x_φ)
# reinit!(ls_evo,φh,0.5)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

Ωin = Triangulation(cutgeo,PHYSICAL_IN).a
Ωout = Triangulation(cutgeo,PHYSICAL_OUT).a

diff_Ωin = DifferentiableTriangulation(Ωin)
diff_Ωout = DifferentiableTriangulation(Ωout)

oh = interpolate(1.0,V_φ)

dΩin = Measure(diff_Ωin,3*order)
j_in(φ) = ∫(1)dΩin
dj_in = gradient(j_in,φh)
dj_vec_in = assemble_vector(dj_in,V_φ)
norm(dj_vec_in)

dΩout = Measure(diff_Ωout,3*order)
j_out(φ) = ∫(oh)dΩout
dj_out = gradient(j_out,φh)
dj_vec_out = -assemble_vector(dj_out,V_φ)
norm(dj_vec_out)

Γ = EmbeddedBoundary(cutgeo)
dΓ = Measure(Γ,3*order)
dj_expected(q) = ∫(-q)dΓ
dj_exp_vec = assemble_vector(dj_expected,V_φ)
norm(dj_exp_vec)

trian = Ωin
dtrian = DifferentiableTriangulation(trian)
update_trian!(dtrian,φh)
meas = Measure(dtrian,2*order)
quad = meas.quad
pts = quad.cell_point

f(dΩ) = sum(∫(1)dΩ)
f(meas)
f(Measure(trian,2*order))

for (i,(a,b)) in enumerate(zip(dj_exp_vec,dj_vec_in))
  if abs(a) < 1.e-10
    @assert abs(b) < 1.e-10
  else
    println(i," - ",a/b)
  end
end

# α = 10^-2
# vel_ext = VelocityExtension((u,v)->∫(α^2*∇(u)⋅∇(v) + u⊙v )dΩ,V_φ,V_φ)
# project!(vel_ext,dj_exp_vec)
# project!(vel_ext,dj_vec_in)

# l2(u) = norm(u)
# l2(u) = sqrt(sum( ∫( u⊙u )dΩ))
# vel_extnorm(u) = sqrt(sum( ∫(α^2*∇(u)⋅∇(u) + u⊙u )dΩ))

# l2_norm = l2(dj_exp_vec-dj_vec_in)/l2(dj_exp_vec)
# l2_norm = l2(FEFunction(V_φ,dj_exp_vec)-FEFunction(V_φ,dj_vec_in))/l2(FEFunction(V_φ,dj_exp_vec))

norm(dj_vec_in-dj_exp_vec)
norm(dj_vec_out-dj_exp_vec)
bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo)

writevtk(
  Ω,"results/test",
  cellfields=["φh"=>φh,"∇φh"=>∇(φh),"dj_in"=>FEFunction(V_φ,dj_vec_in),"dj_expected"=>FEFunction(V_φ,dj_exp_vec),"dj_out"=>FEFunction(V_φ,dj_vec_out)],
  celldata=["inoutcut"=>bgcell_to_inoutcut]
)

writevtk(
  Triangulation(cutgeo,PHYSICAL_IN),"results/trian_in"
)
writevtk(
  Triangulation(cutgeo,PHYSICAL_OUT),"results/trian_out"
)

meas = dΩin.state
quad = meas.quad
trian = quad.trian

cmaps = get_cell_map(trian)
ca = cmaps.a
cb = cmaps.b


get_cell_dof_ids(V_φ)
get_cell_dof_ids(V_φ,trian).a


trian = Ωin.state.a
meas = Measure(trian,2*order)
quad = meas.quad

cell_map = get_cell_map(quad.trian)
cell_Jt = lazy_map(∇,cell_map)
cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
cell_detJtx = lazy_map(Broadcasting(Gridap.TensorValues.meas),cell_Jtx)
println(cell_detJtx)


# To-Do:
#   1. Add updateability condition
#   2. Add caches so we don't have to recompute everything every time
#   3. Figure out a way to share the cache between different triangulations created from the
#      same cut geometry, i.e PHYS_IN/PHYS_OUT/Boundary
#   4. Anything else?
