using Gridap, Gridap.Geometry, Gridap.Adaptivity
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapGmsh
using GridapTopOpt

#############

path = "./results/GMSH-TO-6-Brinkmann_stokes_P2-P1_Ersatz_elast_fsi_multistage_reinit/results/"
mkpath(path)

γ_evo = 0.1
max_steps = 20
vf = 0.025
α_coeff = 2
iter_mod = 1
D = 2

# Load gmsh mesh (Currently need to update mesh.geo and these values concurrently)
L = 2.0;
H = 0.5;
x0 = 0.5;
l = 0.4;
w = 0.025;
a = 0.3;
b = 0.01;

model = GmshDiscreteModel((@__DIR__)*"/mesh.msh")
writevtk(model,path*"model")

Ω_act = Triangulation(model)
hₕ = CellField(get_element_diameters(model),Ω_act)
hmin = minimum(get_element_diameters(model))

# Cut the background model
reffe_scalar = ReferenceFE(lagrangian,Float64,1)
V_φ = TestFESpace(model,reffe_scalar)
V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Omega_NonDesign","Gamma_s_D"])
U_reg = TrialFESpace(V_reg)

_e = 1e-3
f0((x,y),W,H) = max(2/W*abs(x-x0),1/(H/2+1)*abs(y-H/2+1))-1
f1((x,y),q,r) = - cos(q*π*x)*cos(q*π*y)/q - r/q
fin(x) = f0(x,l*(1+5_e),a*(1+5_e))
fsolid(x) = min(f0(x,l*(1+_e),b*(1+_e)),f0(x,w*(1+_e),a*(1+_e)))
fholes((x,y),q,r) = max(f1((x,y),q,r),f1((x-1/q,y),q,r))
φf(x) = min(max(fin(x),fholes(x,22,0.6)),fsolid(x))
φh = interpolate(φf,V_φ)
writevtk(get_triangulation(φh),path*"initial_lsf",cellfields=["φ"=>φh,"h"=>hₕ])

# Setup integration meshes and measures
order = 2
degree = 2*order

# Ω_act = Triangulation(model)
dΩ_act = Measure(Ω_act,degree)
vol_D = L*H

Γf_D = BoundaryTriangulation(model,tags="Gamma_f_D")
Γf_N = BoundaryTriangulation(model,tags="Gamma_f_N")
dΓf_D = Measure(Γf_D,degree)
dΓf_N = Measure(Γf_N,degree)
Ω = EmbeddedCollection(model,φh) do cutgeo,_
  Ωs = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
  Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
  Γ = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
  Γg = GhostSkeleton(cutgeo)
  Ωact = Triangulation(cutgeo,ACTIVE)
  (;
    :Ωs  => Ωs,
    :dΩs => Measure(Ωs,degree),
    :Ωf  => Ωf,
    :dΩf => Measure(Ωf,degree),
    :Γg   => Γg,
    :dΓg  => Measure(Γg,degree),
    :n_Γg => get_normal_vector(Γg),
    :Γ    => Γ,
    :dΓ   => Measure(Γ,degree),
    :n_Γ  => get_normal_vector(Γ.trian),
    :Ωact => Ωact
  )
end

# Setup FESpace
uin(x) = VectorValue(16x[2]*(H-x[2]),0.0)

reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
reffe_p = ReferenceFE(lagrangian,Float64,order-1,space=:P)
reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order-1)

V = TestFESpace(Ω_act,reffe_u,conformity=:H1,
  dirichlet_tags=["Gamma_f_D","Gamma_NoSlipTop","Gamma_NoSlipBottom","Gamma_s_D"])
Q = TestFESpace(Ω_act,reffe_p,conformity=:C0)
T = TestFESpace(Ω_act ,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_s_D"])

U = TrialFESpace(V,[uin,VectorValue(0.0,0.0),VectorValue(0.0,0.0),VectorValue(0.0,0.0)])
P = TrialFESpace(Q)
R = TrialFESpace(T)

X = MultiFieldFESpace([U,P,R])
Y = MultiFieldFESpace([V,Q,T])

# Weak form
## Fluid
# Properties
Re = 60 # Reynolds number
ρ = 1.0 # Density
cl = H # Characteristic length
u0_max = maximum(abs,get_dirichlet_dof_values(U))
μ = ρ*cl*u0_max/Re # Viscosity
ν = μ/ρ # Kinematic viscosity
# Stabilization parameters
γ(h) = 1000/h #1e5*μ/h

# Terms
σf_n(u,p,n) = μ*∇(u) ⋅ n - p*n
a_Ω(u,v) = μ*(∇(u) ⊙ ∇(v))
b_Ω(v,p) = - (∇⋅v)*p

a_fluid((u,p),(v,q)) =
  ∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p)) * Ω.dΩf +
  ∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p) + (γ ∘ hₕ)*u⋅v ) * Ω.dΩs

## Structure
# Stabilization and material parameters
function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end
λs, μs = lame_parameters(0.1,0.05)
ϵ = (λs + 2μs)*1e-3
# Terms
σ(ε) = λs*tr(ε)*one(ε) + 2*μs*ε
a_solid(d,s) = ∫(ε(s) ⊙ (σ ∘ ε(d)))Ω.dΩs +
  ∫(ϵ*(ε(s) ⊙ (σ ∘ ε(d))))Ω.dΩf # Ersatz

## Full problem
vec0 = VectorValue(0.0,0.0)
# minus sign because of the normal direction
function a_coupled((u,p,d),(v,q,s),φ)
  n_AD = get_normal_vector(Ω.Γ)
  return a_fluid((u,p),(v,q)) + a_solid(d,s) +
    ∫(-σf_n(u,p,n_AD) ⋅ s)Ω.dΓ + ∫((-σf_n(u,p,vec0) ⋅ s))dΩ_act
end
l_coupled((v,q,s),φ) = ∫(0.0q)dΩ_act

## Optimisation functionals
J_pres((u,p,d),φ) = ∫(p)dΓf_D - ∫(p)dΓf_N
J_comp((u,p,d),φ) = ∫(ε(d) ⊙ (σ ∘ ε(d)))Ω.dΩs
Vol((u,p,d),φ) = ∫(vol_D)Ω.dΩs - ∫(vf/vol_D)dΩ_act

## Setup solver and FE operators
state_map = AffineFEStateMap(a_coupled,l_coupled,X,Y,V_φ,U_reg,φh)
pcfs = PDEConstrainedFunctionals(J_comp,[Vol],state_map)

## Evolution Method
# evo = CutFEMEvolve(V_φ,Ω,dΩ_act,hmin;max_steps,γg=0.075)
# reinit = StabilisedReinit(V_φ,Ω,dΩ_act,hmin;stabilisation_method=ArtificialViscosity(3.0))
# ls_evo = UnfittedFEEvolution(evo,reinit)
evo = CutFEMEvolve(V_φ,Ω,dΩ_act,hₕ;max_steps,γg=0.1)
reinit1 = StabilisedReinit(V_φ,Ω,dΩ_act,hₕ;stabilisation_method=ArtificialViscosity(2.0))
reinit2 = StabilisedReinit(V_φ,Ω,dΩ_act,hₕ;stabilisation_method=GridapTopOpt.InteriorPenalty(V_φ,γg=1.0))
reinit = GridapTopOpt.MultiStageStabilisedReinit([reinit1,reinit2])
ls_evo = UnfittedFEEvolution(evo,reinit)

## Hilbertian extension-regularisation problems
_α(hₕ) = (α_coeff*hₕ)^2
a_hilb(p,q) =∫((_α ∘ hₕ)*∇(p)⋅∇(q) + p*q)dΩ_act;
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

## Optimiser
converged(m) = GridapTopOpt.default_al_converged(
  m;
  L_tol = 0.5hmin,
  C_tol = 0.01vf
)
function has_oscillations(m,os_it)
  history = GridapTopOpt.get_history(m)
  it = GridapTopOpt.get_last_iteration(history)
  all(@.(abs(history[:C,it]) < 0.05vf)) && GridapTopOpt.default_has_oscillations(m,os_it)
end
optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
  γ=γ_evo,verbose=true,constraint_names=[:Vol],converged,has_oscillations,Λ_update_tol=0)
for (it,(uh,ph,dh),φh) in optimiser
  GC.gc()
  if iszero(it % iter_mod)
    writevtk(Ω_act,path*"Omega_act_$it",
      cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ph"=>ph,"dh"=>dh])
    writevtk(Ω.Ωf,path*"Omega_f_$it",
      cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
    writevtk(Ω.Ωs,path*"Omega_s_$it",
      cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
    writevtk(Ω.Γ,path*"Gamma_$it",cellfields=["σ⋅n"=>(σ ∘ ε(dh))⋅Ω.n_Γ,"σf_n"=>σf_n(uh,ph,φh)])
  end
  write_history(path*"/history.txt",optimiser.history)
end
it = get_history(optimiser).niter; uh,ph,dh = get_state(pcfs)
writevtk(Ω_act,path*"Omega_act_$it",
  cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ph"=>ph,"dh"=>dh])
writevtk(Ω.Ωf,path*"Omega_f_$it",
  cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
writevtk(Ω.Ωs,path*"Omega_s_$it",
  cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
writevtk(Ω.Γ,path*"Gamma_$it",cellfields=["σ⋅n"=>(σ ∘ ε(dh))⋅Ω.n_Γ,"σf_n"=>σf_n(uh,ph,φh)])