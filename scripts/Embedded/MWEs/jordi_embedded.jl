
using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData
import Gridap.Geometry: get_node_coordinates, collect1d

include("../differentiable_trians.jl")

order = 1
model = CartesianDiscreteModel((0,1,0,1),(8,8))
Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)

reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe_scalar)
U = TrialFESpace(V)
V_φ = TestFESpace(model,reffe_scalar)

φh = interpolate(x->max(abs(x[1]-0.5),abs(x[2]-0.5))-0.25,V_φ)
# φh = interpolate(x->max(abs(x[1]-0.5),abs(x[2]-0.5))-0.11,V_φ)
#φh = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-0.302,V_φ)

Ωin = DifferentiableTriangulation(φh) do φh
  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)
  return Triangulation(cutgeo,PHYSICAL_IN)
end

Ωout = DifferentiableTriangulation(φh) do φh
  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)
  return Triangulation(cutgeo,PHYSICAL_OUT)
end

dΩin = Measure(Ωin,5*order)
j_in(φ) = ∫(1)dΩin
dj_in = gradient(j_in,φh)
dj_vec_in = assemble_vector(dj_in,V_φ)

dj_in_a = dj_in[Ωin].a

dΩout = Measure(Ωout,5*order)
j_out(φ) = ∫(1)dΩout
dj_out = gradient(j_out,φh)
dj_vec_out = -assemble_vector(dj_out,V_φ)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)
Γ = EmbeddedBoundary(cutgeo)
dΓ = Measure(Γ,2*order)
dj_expected(q) = ∫(-q)dΓ
dj_exp_vec = assemble_vector(dj_expected,V_φ)

norm(dj_vec_in-dj_exp_vec)
norm(dj_vec_out-dj_exp_vec)
bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo)

writevtk(
  Ω,"results/test",
  cellfields=["φh"=>φh,"dj_in"=>FEFunction(V_φ,dj_vec_in),"dj_expected"=>FEFunction(V_φ,dj_exp_vec),"dj_out"=>FEFunction(V_φ,dj_vec_out)],
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
