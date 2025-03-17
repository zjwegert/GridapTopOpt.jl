using Gridap, Gridap.Geometry, Gridap.Adaptivity
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt

path = "./results/TO-6-Brinkmann_stokes_P1-P1_Ersatz_elast_fsi/"
mkpath(path)

n = 100
γ_evo = 0.05
max_steps = floor(Int,n/5)
vf = 0.03
α_coeff = 2
iter_mod = 1

# Cut the background model
L = 2.0
H = 0.5
x0 = 0.5
l = 0.4
w = 0.05
a = 0.3
b = 0.03

partition = (4n,n)
D = length(partition)
_model = CartesianDiscreteModel((0,L,0,H),partition)
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = ref_model.model

el_Δ = get_el_Δ(_model)
h = maximum(el_Δ)
h_refine = maximum(el_Δ)/2

f_Γ_D(x) = x[1] ≈ 0
f_Γ_N(x) = x[1] ≈ L
f_Γ_NoSlipTop(x) = x[2] ≈ H
f_Γ_NoSlipBottom(x) = x[2] ≈ 0
f_NonDesign(x) = ((x0 - w/2 - eps() <= x[1] <= x0 + w/2 + eps() && 0.0 <= x[2] <= a + eps()) ||
  (x0 - l/2 - eps() <= x[1] <= x0 + l/2 + eps() && 0.0 <= x[2] <= b + eps()))
update_labels!(1,model,f_Γ_D,"Gamma_f_D")
update_labels!(2,model,f_Γ_N,"Gamma_f_N")
update_labels!(3,model,f_Γ_NoSlipTop,"Gamma_NoSlipTop")
update_labels!(4,model,f_Γ_NoSlipBottom,"Gamma_NoSlipBottom")
update_labels!(5,model,f_NonDesign,"NonDesign")
update_labels!(6,model,x->f_NonDesign(x) && f_Γ_NoSlipBottom(x),"Gamma_s_D")
# writevtk(model,path*"model")

# Cut the background model
reffe_scalar = ReferenceFE(lagrangian,Float64,1)
V_φ = TestFESpace(model,reffe_scalar)
V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["NonDesign","Gamma_s_D"])
U_reg = TrialFESpace(V_reg)

f0((x,y),W,H) = max(2/W*abs(x-x0),1/(H/2+1)*abs(y-H/2+1))-1
f1((x,y),q,r) = - cos(q*π*x)*cos(q*π*y)/q - r/q
fin(x) = f0(x,l,a)
fsolid(x) = min(f0(x,l,b),f0(x,w,a))
fholes((x,y),q,r) = max(f1((x,y),q,r),f1((x-1/q,y),q,r))
φf(x) = min(max(fin(x),fholes(x,15,0.5)),fsolid(x))
φh = interpolate(φf,V_φ)
# writevtk(get_triangulation(φh),path*"initial_lsf",cellfields=["φ"=>φh])

# Setup integration meshes and measures
order = 1
degree = 2*order

Ω_act = Triangulation(model)
dΩ_act = Measure(Ω_act,degree)
vol_D = L*H

Γf_D = BoundaryTriangulation(model,tags="Gamma_f_D")
Γf_N = BoundaryTriangulation(model,tags="Gamma_f_N")
dΓf_D = Measure(Γf_D,degree)
dΓf_N = Measure(Γf_N,degree)
Ω = EmbeddedCollection(model,φh) do cutgeo,_,_
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
      :n_Γ        => get_normal_vector(Γ), # Note, need to recompute inside obj/constraints to compute derivs
    :n_Γ  => get_normal_vector(Γ.trian),
    :Ωact => Ωact
  )
end

# Setup FESpace
uin(x) = VectorValue(16x[2]*(H-x[2]),0.0)

reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
reffe_p = ReferenceFE(lagrangian,Float64,order,space=:P)
reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

V = TestFESpace(Ω_act,reffe_u,conformity=:H1,
  dirichlet_tags=["Gamma_f_D","Gamma_NoSlipTop","Gamma_NoSlipBottom","Gamma_s_D"])
Q = TestFESpace(Ω_act,reffe_p,conformity=:H1)
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
μ = 1.0#ρ*cl*u0_max/Re # Viscosity
# Stabilization parameters
# TODO: Need to look at this much closer
γ = 10^16#10.0/μ
β1 = 0.2#1/μ

# Terms
σf_n(u,p,n) = μ*∇(u) ⋅ n - p*n
a_Ω(u,v) = μ*(∇(u) ⊙ ∇(v))
b_Ω(v,p) = - (∇⋅v)*p
c_Ω(p,q) = (β1*h^2)*∇(p)⋅∇(q)

a_fluid((u,p),(v,q)) =
  ∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p)-c_Ω(p,q) ) * Ω.dΩf +
  ∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p)-c_Ω(p,q)+(γ/h)*u⋅v ) * Ω.dΩs

## Structure
# Stabilization and material parameters
function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end
λs, μs = lame_parameters(1.0,0.3) #0.1,0.05)
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
Vol((u,p,d),φ) = ∫(100*1/vol_D)Ω.dΩs - ∫(100*vf/vol_D)dΩ_act

## Setup solver and FE operators
state_map = AffineFEStateMap(a_coupled,l_coupled,X,Y,V_φ,U_reg,φh)
pcfs = PDEConstrainedFunctionals(J_pres,[Vol],state_map)

## Evolution Method
evo = CutFEMEvolve(V_φ,Ω,dΩ_act,h;max_steps)
reinit = StabilisedReinit(V_φ,Ω,dΩ_act,h;stabilisation_method=ArtificialViscosity(3.0))
ls_evo = UnfittedFEEvolution(evo,reinit)

## Hilbertian extension-regularisation problems
α = α_coeff*h_refine
a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ_act;
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
  γ=γ_evo,verbose=true,constraint_names=[:Vol])
for (it,(uh,ph,dh),φh) in optimiser
  if iszero(it % iter_mod)
    writevtk(Ω_act,path*"Omega_act_$it",
      cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ph"=>ph,"dh"=>dh])
    writevtk(Ω.Ωf,path*"Omega_f_$it",
      cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
    writevtk(Ω.Ωs,path*"Omega_s_$it",
      cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
    writevtk(Ω.Γ,path*"Gamma_$it",cellfields=["σ⋅n"=>(σ ∘ ε(dh))⋅Ω.n_Γ,"σf_n"=>σf_n(uh,ph,φh)])
    error()
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