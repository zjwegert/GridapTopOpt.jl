using Gridap, Gridap.FESpaces, Gridap.CellData
using Test

model = CartesianDiscreteModel((0,1,0,1),(4,4))
dΩ = Measure(get_triangulation(model),2)
V = FESpace(model,ReferenceFE(lagrangian,Float64,1);dirichlet_tags="boundary")
U = TrialFESpace(V,x->x[1])

UB = MultiFieldFESpace([U,U])
VB = MultiFieldFESpace([V,V])

function c(((u,p),),d)
  return ∫(d*d)dΩ
end
function c_fix(((u,p),),d)
  return ∫(d*d + 0p)dΩ
end

uph = interpolate([x->x[1],x->x[2]],UB)
dh = interpolate(x->2x[1],U)

c((uph,),dh)
∇(up->c((up,),dh),uph)
∇(up->c_fix((up,),dh),uph)