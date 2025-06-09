using Gridap,GridapTopOpt, GridapSolvers
using Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters

using GridapTopOpt: StateParamMap

path="./results/FCM_elastic_compliance_LShape_ALM_n100/"
rm(path,force=true,recursive=true)
mkpath(path)
n = 100
order = 1
γ = 0.2
max_steps = floor(Int,order*n/10)
vf = 0.4
α_coeff = max_steps*γ
iter_mod = 1

_model = CartesianDiscreteModel((0,1,0,1),(n,n))
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = ref_model.model
el_Δ = get_el_Δ(_model)
h = maximum(el_Δ)
h_refine = maximum(el_Δ)/2
f_Γ_D(x) = x[2] ≈ 1.0
f_Γ_N(x) = (x[1] ≈ 1 && 0.2 - eps() <= x[2] <= 0.3 + eps())
update_labels!(1,model,f_Γ_D,"Gamma_D")
update_labels!(2,model,f_Γ_N,"Gamma_N")

f(x) = ~(0.5 + eps() < x[1] < 1 - eps() && 0.5 + eps() < x[2] < 1 - eps());
mask = GridapTopOpt.mark_nodes(f,model)
mask_in = findall(isone,mask)
topo = get_grid_topology(model)
cell_to_nodes = Gridap.Geometry.get_faces(topo,2,0);
cell_mask = findall(x -> all(in.(x, Ref(mask_in))), cell_to_nodes)
model = UnstructuredDiscreteModel(DiscreteModelPortion(model,cell_mask))

## Triangulations and measures
Ω = Triangulation(model)
Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
dΩ = Measure(Ω,2*order)
dΓ_N = Measure(Γ_N,2*order)
vol_D = sum(∫(1)dΩ)

## Levet-set function space and derivative regularisation space
reffe_scalar = ReferenceFE(lagrangian,Float64,1)
V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
U_reg = TrialFESpace(V_reg,0)
V_φ = TestFESpace(model,reffe_scalar)

## Levet-set function
φh = interpolate(x->-cos(6π*x[1])*cos(8π*x[2])-0.2,V_φ)
Ωs = EmbeddedCollection(model,φh) do cutgeo,_,_
  Ωin = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_IN),V_φ)
  Ωout = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
  Γ = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
  Γg = GhostSkeleton(cutgeo)
  Ωact = Triangulation(cutgeo,ACTIVE)
  (;
    :Ωin  => Ωin,
    :dΩin => Measure(Ωin,2*order),
    :Ωout  => Ωout,
    :dΩout => Measure(Ωout,2*order),
    :Γg   => Γg,
    :dΓg  => Measure(Γg,2*order),
    :n_Γg => get_normal_vector(Γg),
    :Γ    => Γ,
    :dΓ   => Measure(Γ,2*order),
    :n_Γ  => get_normal_vector(Γ), # Note, need to recompute inside obj/constraints to compute derivs
    :Ωact => Ωact
  )
end

## Weak form
function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end
E = 1.0
ν = 0.3
λ, μ = lame_parameters(E,ν)
σ(ε) = λ*tr(ε)*one(ε) + 2*μ*ε
g = VectorValue(0,-0.1)

const ϵ = (λ + μ)*1e-3
a(u,v,φ) = ∫(ε(v) ⊙ (σ ∘ ε(u)))Ωs.dΩin +
  ∫(ϵ*(ε(v) ⊙ (σ ∘ ε(u))))Ωs.dΩout
l(v,φ) = ∫(v⋅g)dΓ_N

## Optimisation functionals
J(u,φ) = ∫(ε(u) ⊙ (σ ∘ ε(u)))Ωs.dΩin
Vol(u,φ) = ∫(1/vol_D)Ωs.dΩin - ∫(vf/vol_D)dΩ
dVol(q,u,φ) = ∫(-1/vol_D*q/(abs(Ωs.n_Γ ⋅ ∇(φ))))Ωs.dΓ

## Setup solver and FE operators
reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
V = TestFESpace(Ω,reffe;dirichlet_tags=["Gamma_D"])
U = TrialFESpace(V,VectorValue(0.0,0.0))
state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh)
pcfs = PDEConstrainedFunctionals(J,[Vol],state_map;analytic_dC=(dVol,))

## Evolution Method
evo = CutFEMEvolve(V_φ,Ωs,dΩ,h;max_steps,γg=0.1)
reinit1 = StabilisedReinit(V_φ,Ωs,dΩ,h;stabilisation_method=ArtificialViscosity(3.0))
reinit2 = StabilisedReinit(V_φ,Ωs,dΩ,h;stabilisation_method=GridapTopOpt.InteriorPenalty(V_φ,γg=2.0))
reinit = GridapTopOpt.MultiStageStabilisedReinit([reinit1,reinit2])
ls_evo = UnfittedFEEvolution(evo,reinit)

## Hilbertian extension-regularisation problems
α = (α_coeff*h_refine/order)^2
a_hilb(p,q) =∫(α*∇(p)⋅∇(q) + p*q)dΩ;
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

## Optimiser
converged(m) = GridapTopOpt.default_al_converged(
  m;
  L_tol = 0.01*h_refine,
  C_tol = 0.01
)
optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;debug=true,
  γ,verbose=true,constraint_names=[:Vol],converged)
for (it,uh,φh,state) in optimiser
  x_φ = get_free_dof_values(φh)
  idx = findall(isapprox(0.0;atol=10^-10),x_φ)
  !isempty(idx) && @warn "Boundary intersects nodes!"
  if iszero(it % iter_mod)
    writevtk(Ω,path*"Omega$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"velh"=>FEFunction(V_φ,state.vel)])
    writevtk(Ωs.Ωin,path*"Omega_in$it",cellfields=["uh"=>uh])
  end
  write_history(path*"/history.txt",optimiser.history)
end
it = get_history(optimiser).niter; uh = get_state(pcfs)
writevtk(Ω,path*"Omega$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
writevtk(Ωs.Ωin,path*"Omega_in$it",cellfields=["uh"=>uh])