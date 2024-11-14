using Gridap,GridapTopOpt
using Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters

using GridapTopOpt: StateParamIntegrandWithMeasure

path="./results/UnfittedFEM_thermal_compliance_ALM/"
n = 51
order = 1
γ = 0.1
γ_reinit = 0.5
max_steps = floor(Int,order*minimum(n)/10)
tol = 1/(5*order^2)/minimum(n)
vf = 0.4
α_coeff = 4max_steps*γ
iter_mod = 1

_model = CartesianDiscreteModel((0,1,0,1),(n,n))
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = ref_model.model
el_Δ = get_el_Δ(_model)
h = maximum(el_Δ)
f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= 0.2 + eps() || x[2] >= 0.8 - eps()))
f_Γ_N(x) = (x[1] ≈ 1 && 0.4 - eps() <= x[2] <= 0.6 + eps())
update_labels!(1,model,f_Γ_D,"Gamma_D")
update_labels!(2,model,f_Γ_N,"Gamma_N")

## Triangulations and measures
Ω = Triangulation(model)
Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
dΩ = Measure(Ω,2*order)
dΓ_N = Measure(Γ_N,2*order)
vol_D = sum(∫(1)dΩ)

## Levet-set function space and derivative regularisation space
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
U_reg = TrialFESpace(V_reg,0)
V_φ = TestFESpace(model,reffe_scalar)

## Levet-set function
φh = interpolate(x->-cos(4π*x[1])*cos(4π*x[2])-0.2,V_φ)
geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)
Ωs = EmbeddedCollection(model,φh) do cutgeo
  Ωin = Triangulation(cutgeo,PHYSICAL)
  Γg = GhostSkeleton(cutgeo)
  n_Γg = get_normal_vector(Γg)
  (; 
    :Ωin  => Ωin,
    :dΩin => Measure(Ωin,2*order),
    :Γg   => Γg,
    :dΓg  => Measure(Γg,2*order),
    :n_Γg => get_normal_vector(Γg)  
  )
end  

## Weak form
const γg = 0.1
a(u,v,φ) = ∫(∇(v)⋅∇(u))Ωs.dΩin + ∫((γg*h)*jump(Ωs.n_Γg⋅∇(v))*jump(Ωs.n_Γg⋅∇(u)))Ωs.dΓg
l(v,φ) = ∫(v)dΓ_N

## Optimisation functionals
J(u,φ) = a(u,u,φ)
Vol(u,φ) = ∫(1/vol_D)Ωs.dΩin - ∫(vf/vol_D)dΩ

## Setup solver and FE operators
state_collection = EmbeddedCollection(model,φh) do cutgeo
  Ωact = Triangulation(cutgeo,ACTIVE)
  V = TestFESpace(Ωact,reffe_scalar;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,0.0)

  Ωin = Triangulation(cutgeo,PHYSICAL)
  Γg = GhostSkeleton(cutgeo)
  n_Γg = get_normal_vector(Γg)
  dΩin = Measure(Ωin,2*order)
  dΓg  = Measure(Γg,2*order)
  n_Γg = get_normal_vector(Γg)  

  dΩact = Measure(Ωact,2)
  Ωact_out = Triangulation(cutgeo,ACTIVE_OUT)
  dΩact_out = Measure(Ωact_out,2)

  a(u,v,φ) = ∫(∇(v)⋅∇(u))dΩin + ∫((γg*h)*jump(n_Γg⋅∇(v))*jump(n_Γg⋅∇(u)))dΓg
  l( v,φ) = ∫(v)dΓ_N
  J(u,φ) = a(u,u,φ)
  Vol(u,φ) = ∫(1/vol_D)dΩin - ∫(vf/vol_D)dΩact - ∫(vf/vol_D)dΩact_out
  state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh)
  (; 
    :state_map => state_map,
    :J => StateParamIntegrandWithMeasure(J,state_map),
    :C => map(Ci -> StateParamIntegrandWithMeasure(Ci,state_map),(Vol,))
  )
end  
pcfs = EmbeddedPDEConstrainedFunctionals(state_collection)

evaluate!(pcfs,φh)

using Gridap.Arrays
uh = u
ttrian = Ω
strian = get_triangulation(uh)

D = num_cell_dims(strian)
sglue = get_glue(strian,Val(D))
tglue = get_glue(ttrian,Val(D))

scells = Arrays.IdentityVector(Int32(num_cells(strian)))
mcells = extend(scells,sglue.mface_to_tface)
tcells = lazy_map(Reindex(mcells),tglue.tface_to_mface)
collect(tcells)


## Evolution Method
evo = CutFEMEvolve(V_φ,Ωs,dΩ,h)
reinit = StabilisedReinit(V_φ,Ωs,dΩ,h;stabilisation_method=InteriorPenalty(V_φ))
ls_evo = UnfittedFEEvolution(evo,reinit)
reinit!(ls_evo,φh)

## Hilbertian extension-regularisation problems
α = α_coeff*maximum(el_Δ)
a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

Ωs(φh)

evaluate!(pcfs,φh)
 
# ## Optimiser
# optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
#   γ,γ_reinit,verbose=true,constraint_names=[:Vol])
# for (it,uh,φh) in optimiser
#   if iszero(it % iter_mod)
#     writevtk(Ω,path*"Omega$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
#     writevtk(Ωs.Γ,path*"Gamma_out$it",cellfields=["normal"=>get_normal_vector(_Γ)])
#   end
#   write_history(path*"/history.txt",optimiser.history)
# end
# it = get_history(optimiser).niter; uh = get_state(pcfs)
# writevtk(Ω,path*"Omega$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
# writevtk(Ωs.Γ,path*"Gamma_out$it",cellfields=["normal"=>get_normal_vector(_Γ)])