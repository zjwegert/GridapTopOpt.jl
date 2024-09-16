using GridapTopOpt

using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData
import Gridap.Geometry: get_node_coordinates, collect1d

include("../differentiable_trians.jl")

order = 1
n = 101

_model = CartesianDiscreteModel((0,1,0,1),(n,n))
cd = Gridap.Geometry.get_cartesian_descriptor(_model)
h = maximum(cd.sizes)

model = simplexify(_model)
Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe_scalar)

# φh = interpolate(x->(x[1]-0.5)^2+(x[2]-0.5)^2-0.25^2,V_φ)
φh = interpolate(x->cos(2π*x[1])*cos(2π*x[2])-0.11,V_φ)
geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)
Γ = EmbeddedBoundary(cutgeo)
dΓ = Measure(Γ,2*order)

bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo);

begin
  γd = 20
  γg = 0.1
  ν  = 1
  cₐ = 0.5 # <- 3 in connor's paper
  ϵ = 1e-20
  sgn(ϕ₀) = sign ∘ ϕ₀
  d1(∇u) = 1 / ( ϵ + norm(∇u) )
  W(u) =  sgn(u) * ∇(u) * (d1 ∘ (∇(u)))
  νₐ(w) = cₐ*h * (sqrt∘( w ⋅ w ))
  a_ν(w,u,v) = ∫((γd/h)*v*u)dΓ + ∫(νₐ(W(w))*∇(u)⋅∇(v) +  v*W(w)⋅∇(u))dΩ
  b_ν(w,v) = ∫( sgn(w)*v )dΩ
  res(u,v)    = a_ν(u,u,v)  - b_ν(u,v)
  jac(u,du,v)  = a_ν(u,du,v)

  op = FEOperator(res,jac,V_φ,V_φ)
  ls = LUSolver()
  nls = NLSolver(ftol=1e-14, iterations= 50, show_trace=true)
  solver = FESolver(nls)
  φh_new = FEFunction(V_φ,copy(φh.free_values))
  Gridap.solve!(φh_new,nls,op)

  writevtk(
    Ω,"results/test_reinit",
    cellfields=["φh"=>φh,"|∇φh|"=>norm ∘ ∇(φh),"φh_new"=>φh_new,"|∇φh_new|"=>norm ∘∇(φh_new)],
    celldata=["inoutcut"=>bgcell_to_inoutcut]
  )
end