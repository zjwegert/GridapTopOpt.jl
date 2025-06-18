module tmp
using Gridap,GridapTopOpt, GridapSolvers
using Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt: StateParamMap
using GridapEmbedded.LevelSetCutters: DifferentiableTriangulation
using Zygote
using ChainRulesCore
n=10
model = simplexify(CartesianDiscreteModel((0,1,0,1),(n,n)))
el_Δ = 1/n
h = maximum(el_Δ)
f_Γ_N(x) = (x[1] ≈ 1 && 0.4 - eps() <= x[2] <= 0.6 + eps())
update_labels!(2,model,f_Γ_N,"Gamma_N")
## Triangulations and measures
Ω = Triangulation(model)
order = 1 
dΩ = Measure(Ω,2*order)
## Levet-set function space and derivative regularisation space
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
U_reg = TrialFESpace(V_reg,0)
V_φ = TestFESpace(model,reffe_scalar)
U_φ_ = TrialFESpace(V_reg,-1.0)
## Levet-set function
φh = interpolate(x->-cos(4π*x[1])*cos(4π*x[2])-0.4,V_φ)
Ωs = EmbeddedCollection(model,φh) do cutgeo,_,_
  Ωin = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),U_φ_)
  Ωact = Triangulation(cutgeo,ACTIVE)
  (;
    :Ωin  => Ωin,
    :dΩin => Measure(Ωin,2*order),
    :Ωact => Ωact,
  )
end
j(u,φ) = ∫(∇(u)⋅∇(u))Ωs.dΩin
U = V_φ
V = V_φ
assem_U = SparseMatrixAssembler(U,V)
assem_deriv = SparseMatrixAssembler(U_reg,U_reg)

#X = V_φ
X = U_φ_

J = StateParamMap(j,U,X,U_reg,assem_U,assem_deriv)
u = interpolate(x->x[1],U)
j_val, j_pullback = rrule(J,u.free_values,φh.free_values)   # Compute functional and pull back
_, _, dFdφ     = j_pullback(1)
sum(dFdφ)

dsds


end


