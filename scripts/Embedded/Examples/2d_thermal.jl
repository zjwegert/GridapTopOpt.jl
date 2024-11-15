using Gridap,GridapTopOpt
using Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters

using GridapTopOpt: StateParamIntegrandWithMeasure

path="./results/UnfittedFEM_thermal_compliance_ALM/"
mkpath(path)
n = 101
order = 1
γ = 0.1
γ_reinit = 0.5
max_steps = floor(Int,order*minimum(n)/10)
vf = 0.4
α_coeff = 4max_steps*γ
iter_mod = 1

_model = CartesianDiscreteModel((0,1,0,1),(n,n))
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = ref_model.model
el_Δ = get_el_Δ(_model)
h = maximum(el_Δ)
h_refine = maximum(el_Δ)/2
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
Ωs = EmbeddedCollection(model,φh) do cutgeo
  Ωin = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL))
  Γ = DifferentiableTriangulation(EmbeddedBoundary(cutgeo))
  Γg = GhostSkeleton(cutgeo)
  n_Γg = get_normal_vector(Γg)
  Ωact = Triangulation(cutgeo,ACTIVE)
  (; 
    :Ωin  => Ωin,
    :dΩin => Measure(Ωin,2*order),
    :Γg   => Γg,
    :dΓg  => Measure(Γg,2*order),
    :n_Γg => get_normal_vector(Γg),
    :Γ    => Γ,
    :dΓ   => Measure(Γ,2*order),
    :Ωact => Ωact
  )
end  

## Weak form
const γg = 0.1
a(u,v,φ) = ∫(∇(v)⋅∇(u))Ωs.dΩin + ∫((γg*h)*jump(Ωs.n_Γg⋅∇(v))*jump(Ωs.n_Γg⋅∇(u)))Ωs.dΓg
l(v,φ) = ∫(v)dΓ_N

## Optimisation functionals
J(u,φ) = a(u,u,φ)
Vol(u,φ) = ∫(1/vol_D)Ωs.dΩin - ∫(vf/vol_D)dΩ
dVol(q,u,φ) = ∫(-1/vol_D*q/(norm ∘ (∇(φ))))Ωs.dΓ

## Setup solver and FE operators
state_collection = EmbeddedCollection(model,φh) do _
  # update_collection!(Ωs,φh)
  V = TestFESpace(Ωs.Ωact,reffe_scalar;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,0.0)
  state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh)
  (; 
    :state_map => state_map,
    :J => StateParamIntegrandWithMeasure(J,state_map),
    :C => map(Ci -> StateParamIntegrandWithMeasure(Ci,state_map),[Vol,])
  )
end  
pcfs = EmbeddedPDEConstrainedFunctionals(state_collection;analytic_dC=(dVol,))

## Evolution Method
evo = CutFEMEvolve(V_φ,Ωs,dΩ,h)
reinit = StabilisedReinit(V_φ,Ωs,dΩ,h;stabilisation_method=ArtificialViscosity(3.0))#InteriorPenalty(V_φ))
ls_evo = UnfittedFEEvolution(evo,reinit)
reinit!(ls_evo,φh)

## Hilbertian extension-regularisation problems
## α = α_coeff*(h_refine/order)^2
## a_hilb(p,q) =∫(α*∇(p)⋅∇(q) + p*q)dΩ;
α = α_coeff*h_refine
a_hilb(p,q) = ∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

## Optimiser
optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;debug=true,
  γ,γ_reinit,verbose=true,constraint_names=[:Vol])
for (it,uh,φh,state) in optimiser
  if iszero(it % iter_mod)
    writevtk(Ω,path*"Omega$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"velh"=>FEFunction(V_φ,state.vel)])
    writevtk(Ωs.Γ,path*"Gamma_out$it")
  end
  write_history(path*"/history.txt",optimiser.history)
end
it = get_history(optimiser).niter; uh = get_state(pcfs)
writevtk(Ω,path*"Omega$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
writevtk(Ωs.Γ,path*"Gamma_out$it")