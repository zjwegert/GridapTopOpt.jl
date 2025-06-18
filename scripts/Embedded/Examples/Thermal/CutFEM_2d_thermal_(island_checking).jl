module tmp
using Gridap,GridapTopOpt, GridapSolvers
using Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters

using GridapTopOpt: StateParamMap

using GridapEmbedded.LevelSetCutters: DifferentiableTriangulation

path="./results/CutFEM_thermal_compliance_ALM_island_detect/"
rm(path,force=true,recursive=true)
mkpath(path)
n = 30
order = 1
γ = 0.2
max_steps = floor(Int,order*n/10)
vf = 0.4
α_coeff = max_steps*γ
iter_mod = 1

_model = (CartesianDiscreteModel((0,1,0,1),(n,n)))
# base_model = UnstructuredDiscreteModel(_model)
# ref_model = refine(base_model, refinement_method = "barycentric")
# model = ref_model.model
model = _model
el_Δ = 1/n #get_el_Δ(_model)
h = maximum(el_Δ)
h_refine = h# maximum(el_Δ)/2
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
U_φ_ = TrialFESpace(V_reg,-1.0)

## Levet-set function
φh = interpolate(x->-cos(4π*x[1])*cos(4π*x[2])-0.4,V_φ)
Ωs = EmbeddedCollection(model,φh) do cutgeo,_,_
  Ωin = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
  Γ = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
  Γg = GhostSkeleton(cutgeo)
  Ωact = Triangulation(cutgeo,ACTIVE)
  (;
    :Ωin  => Ωin,
    :dΩin => Measure(Ωin,2*order),
    :Γg   => Γg,
    :dΓg  => Measure(Γg,2*order),
    :n_Γg => get_normal_vector(Γg),
    :Γ    => Γ,
    :dΓ   => Measure(Γ,2*order),
    :n_Γ  => get_normal_vector(Γ), # Note, need to recompute inside obj/constraints to compute derivs
    :Ωact => Ωact,
    :χ => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_D"])
  )
end

## Weak form
const γg = 0.1
a(u,v,φ) = ∫(∇(v)⋅∇(u))Ωs.dΩin +
  ∫((γg*h)*jump(Ωs.n_Γg⋅∇(v))*jump(Ωs.n_Γg⋅∇(u)))Ωs.dΓg +
  ∫(Ωs.χ*v*u)Ωs.dΩin
l(v,φ) = ∫(v)dΓ_N

## Optimisation functionals
J(u,φ) = ∫(∇(u)⋅∇(u))Ωs.dΩin
Vol(u,φ) = ∫(1/vol_D)Ωs.dΩin - ∫(vf/vol_D)dΩ
dVol(q,u,φ) = ∫(-1/vol_D*q/(abs(Ωs.n_Γ ⋅ ∇(φ))))Ωs.dΓ

## Setup solver and FE operators
state_collection = GridapTopOpt.EmbeddedCollection_in_φh(model,φh) do _φh
  V = TestFESpace(Ωs.Ωact,reffe_scalar;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,0.0)
  state_map = AffineFEStateMap(a,l,U,V,V_φ)
  (;
    :state_map => state_map,
    :J => StateParamMap(J,state_map),
    :C => map(Ci -> StateParamMap(Ci,state_map),[Vol,])
  )
end

function φ_to_jc(φ)
  u = state_collection.state_map(φ)
  j = state_collection.J(u,φ)
  c = state_collection.C[1](u,φ)
  [j,c]
end
#φ_to_jc(state_collection) = φ -> φ_to_jc(φ,state_collection)
pcfs = CustomEmbeddedPDEConstrainedFunctionals(φ_to_jc,state_collection,Ωs,φh)


# ## Evolution Method
# evo = CutFEMEvolve(V_φ,Ωs,dΩ,h;max_steps,γg=0.1)
# reinit1 = StabilisedReinit(V_φ,Ωs,dΩ,h;stabilisation_method=ArtificialViscosity(3.0))
# reinit2 = StabilisedReinit(V_φ,Ωs,dΩ,h;stabilisation_method=GridapTopOpt.InteriorPenalty(V_φ,γg=2.0))
# reinit = GridapTopOpt.MultiStageStabilisedReinit([reinit1,reinit2])
# ls_evo = UnfittedFEEvolution(evo,reinit)


Ω = dΩ.quad.trian
model = get_background_model(Ω)
el_size = el_Δ #model.grid.node_coords.data.partition 
max_steps = floor(Int,order*minimum(el_size)/10)
tol = 1/(5order^2)/minimum(el_size)
α_coeff = 4max_steps*γ * α_coeff
ls_evo = HamiltonJacobiEvolution(FirstOrderStencil(2,Float64),model,V_φ,tol,max_steps)
# 



## Hilbertian extension-regularisation problems
α = α_coeff*(h_refine/order)^2
a_hilb(p,q) =∫(α*∇(p)⋅∇(q) + p*q)dΩ;
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)


optimiser = HilbertianProjection(pcfs,ls_evo,vel_ext,φh;debug=true,
  γ,verbose=true,constraint_names=[:Vol])
for (it,uh,φh,state) in optimiser
  x_φ = get_free_dof_values(φh)
  idx = findall(isapprox(0.0;atol=10^-10),x_φ)
  !isempty(idx) && @warn "Boundary intersects nodes!"
  if iszero(it % iter_mod)
    writevtk(Ω,path*"Omega$it",cellfields=["φ"=>φh,"χ"=>Ωs.χ])
    writevtk(Ωs.Ωin,path*"Omega_in$it",cellfields=["uh"=>uh])
  end
  write_history(path*"/history.txt",optimiser.history)
end
it = get_history(optimiser).niter; uh = get_state(pcfs)
writevtk(Ω,path*"Omega$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"χ"=>Ωs.χ])
writevtk(Ωs.Ωin,path*"Omega_in$it",cellfields=["uh"=>uh])

end