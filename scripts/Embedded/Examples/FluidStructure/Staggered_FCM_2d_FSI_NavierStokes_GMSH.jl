using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt

using LinearAlgebra
LinearAlgebra.norm(x::VectorValue,p::Real) = norm(x.data,p)

if isassigned(ARGS,1)
  global γg_evo =  parse(Float64,ARGS[1])
else
  global γg_evo =  0.1
end

path = "./results/Staggered_FCM_2d_FSI_NavierStokes_GMSH_gammag$γg_evo/"
mkpath(path)

γ_evo = 0.2
max_steps = 24 # Based on number of elements in vertical direction divided by 10
vf = 0.025
α_coeff = γ_evo*max_steps
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
vol_D = 2.0*0.5

model = GmshDiscreteModel((@__DIR__)*"/Meshes/mesh_finer.msh")
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
φf(x) = min(max(fin(x),fholes(x,25,0.2)),fsolid(x))
# φf(x) = min(max(fin(x),fholes(x,22,0.6)),fsolid(x))
φh = interpolate(φf,V_φ)
writevtk(get_triangulation(φh),path*"initial_lsf",cellfields=["φ"=>φh,"h"=>hₕ])

φh_nondesign = interpolate(fsolid,V_φ)

# Setup integration meshes and measures
order = 1
degree = 2*(order+1)

dΩ_act = Measure(Ω_act,degree)
Γf_D = BoundaryTriangulation(model,tags="Gamma_f_D")
Γf_N = BoundaryTriangulation(model,tags="Gamma_f_N")
dΓf_D = Measure(Γf_D,degree)
dΓf_N = Measure(Γf_N,degree)
Ω = EmbeddedCollection(model,φh) do cutgeo,_
  Ωs = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
  Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
  Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
  Γg = GhostSkeleton(cutgeo)
  (;
    :Ωs      => Ωs,
    :dΩs     => Measure(Ωs,degree),
    :Ωf      => Ωf,
    :dΩf     => Measure(Ωf,degree),
    :Γg      => Γg,
    :dΓg     => Measure(Γg,degree),
    :n_Γg    => get_normal_vector(Γg),
    :Γ       => Γ,
    :dΓ      => Measure(Γ,degree),
    :χ_s     => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_s_D"];IN_is=IN),
    :χ_f     => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_f_D"];IN_is=OUT)
  )
end
writevtk(get_triangulation(φh),path*"initial_islands",cellfields=["χ_s"=>Ω.χ_s,"χ_f"=>Ω.χ_f])

# Setup spaces
uin(x) = VectorValue(16x[2]*(H-x[2]),0.0)
uin_dot_e1(x) = uin(x)⋅VectorValue(1.0,0.0)

reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
reffe_p = ReferenceFE(lagrangian,Float64,order,space=:P)
reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

# Test spaces
V = TestFESpace(Ω_act,reffe_u,conformity=:H1,
  dirichlet_tags=["Gamma_f_D","Gamma_NoSlipTop","Gamma_NoSlipBottom","Gamma_s_D"])
Q = TestFESpace(Ω_act,reffe_p,conformity=:H1)
T = TestFESpace(Ω_act,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_s_D"])

# Trial spaces
U = TrialFESpace(V,[uin,VectorValue(0.0,0.0),VectorValue(0.0,0.0),VectorValue(0.0,0.0)])
P = TrialFESpace(Q)
R = TrialFESpace(T)

# Multifield spaces
mfs = BlockMultiFieldStyle(2,(2,1))
X = MultiFieldFESpace([U,P,R];style=mfs)
Y = MultiFieldFESpace([V,Q,T];style=mfs)

### Weak form

## Fluid
NS = 1 # Turn convection on and off
SUPG = 1 # Turn SUPG on and off

# Properties
Re = 50 # Reynolds number
ρ = 1.0 # Density
cl = a # Characteristic length
u0_max = sum(∫(uin_dot_e1)dΓf_D)/sum(∫(1)dΓf_D)
μ = ρ*cl*u0_max/Re # Viscosity
ν = μ/ρ # Kinematic viscosity

# Stabilization parameters
α_Nu   = 2.5*10
α_SUPG = 1/3

γ_Nu         = α_Nu*(μ/0.001^2)
τ_SUPG(h,u)  = α_SUPG*(SUPG*(2norm(u)/h)^2 + 9*(4ν/h^2)^2)^-0.5 # (Eqn. 31, Peterson et al., 2018)
τ_PSPG(h,u)  = τ_SUPG(h,u) # (Sec. 3.2.2, Peterson et al., 2018)

# Terms
δ = one(SymTensorValue{D,Float64})
σ_f(ε,p) = -p*δ + 2μ*ε
σ_f_β(ε,p) = -βp*p*δ + βμ*2μ*ε

conv(u,∇u) = (∇u') ⋅ u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

r_conv(u,v) = NS*ρ*v ⋅ (conv∘(u,∇(u)))
r_Ωf((u,p),(v,q)) = ε(v) ⊙ (σ_f ∘ (ε(u),p)) + q*(∇⋅u)
r_SUPG((u,p),(v,q),w) = (NS*SUPG*(τ_SUPG ∘ (hₕ,w))*(conv ∘ (u,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅
  (NS*SUPG*ρ*(conv∘(u,∇(u))) + ∇(p) - μ*Δ(u))
r_SUPG_picard((u,p),(v,q),w) = (NS*SUPG*(τ_SUPG ∘ (hₕ,w))*(conv ∘ (w,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅
  (NS*SUPG*ρ*(conv∘(w,∇(u))) + ∇(p) - μ*Δ(u))

dr_conv(u,du,v) = NS*ρ*v ⋅ (dconv∘(du,∇(du),u,∇(u)))
dr_SUPG((u,p),(du,dp),(v,q),w) =
  (NS*SUPG*(τ_SUPG ∘ (hₕ,w))*(conv ∘ (du,∇(v))))⋅(NS*SUPG*ρ*(conv∘(u,∇(u))) + ∇(p) - μ*Δ(u)) +
  (NS*SUPG*(τ_SUPG ∘ (hₕ,w))*(conv ∘ (u,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅(NS*SUPG*ρ*(dconv∘(du,∇(du),u,∇(u))) + ∇(dp) - μ*Δ(du))

function res_fluid((),(u,p),(v,q),φ)
  return ∫(r_conv(u,v) + r_Ωf((u,p),(v,q)))Ω.dΩf +
    ∫(r_conv(u,v) + r_Ωf((u,p),(v,q)) + γ_Nu*(u⋅v))Ω.dΩs +
    ∫(r_SUPG((u,p),(v,q),u))dΩ_act
end

function jac_fluid_picard((),(u,p),(du,dp),(v,q),φ)
  return ∫(NS*ρ*v ⋅ (conv∘(u,∇(du))) + r_Ωf((du,dp),(v,q)))Ω.dΩs +
    ∫(NS*ρ*v ⋅ (conv∘(u,∇(du))) + r_Ωf((du,dp),(v,q)) + γ_Nu*(du⋅v))Ω.dΩf +
    ∫(r_SUPG_picard((du,dp),(v,q),u))dΩ_act
end

function jac_fluid_newton((),(u,p),(du,dp),(v,q),φ)
  return ∫(dr_conv(u,du,v) + r_Ωf((du,dp),(v,q)))Ω.dΩf +
    ∫(dr_conv(u,du,v) + r_Ωf((du,dp),(v,q)) + γ_Nu*(du⋅v))Ω.dΩs +
    ∫(dr_SUPG((u,p),(du,dp),(v,q),u))dΩ_act
end

## Structure
# Material parameters
function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end
λs, μs = lame_parameters(0.1,0.05)
# Ersatz parameter
ϵ = (λs + 2μs)*1e-3
# Terms
σ(ε) = λs*tr(ε)*one(ε) + 2*μs*ε
a_s_Ω(s,d) = ε(s) ⊙ (σ ∘ ε(d))

function a_solid(((u,p),),d,s,φ)
  return ∫(a_s_Ω(s,d))Ω.dΩs + ∫(ϵ*a_s_Ω(s,d))Ω.dΩf
end
function l_solid(((u,p),),s,φ)
  n = get_normal_vector(Ω.Γ)
  return ∫(s ⋅ (σ_f(ε(u),p) ⋅ n))Ω.dΓ
end

res_solid(((u,p),),d,s,φ) = a_solid(((u,p),),d,s,φ) - l_solid(((u,p),),s,φ)
jac_solid(((u,p),),d,dd,s,φ) = a_solid(((u,p),),dd,s,φ)

∂R2∂xh1((du,dp),((u,p),),d,s,φ) = -1*l_solid(((du,dp),),s,φ)
∂Rk∂xhi = ((∂R2∂xh1,),)

## Optimisation functionals
J_comp(((u,p),d),φ) = ∫(ε(d) ⊙ (σ ∘ ε(d)))Ω.dΩs
∂Jcomp∂up((du,dp),((u,p),d),φ) = ∫(0dp)Ω.dΩs
∂Jcomp∂d(dd,((u,p),d),φ) = ∫(ε(dd) ⊙ (σ ∘ ε(d)))Ω.dΩs + ∫(ε(d) ⊙ (σ ∘ ε(dd)))Ω.dΩs
∂Jcomp∂xhi = (∂Jcomp∂up,∂Jcomp∂d)

Vol(((u,p),d),φ) = ∫(vol_D)Ω.dΩs - ∫(vf/vol_D)dΩ_act
dVol(q,(u,p,d),φ) = ∫(-1/vol_D*q/(norm ∘ (∇(φ))))Ω.dΓ
∂Vol∂up((du,dp),((u,p),d),φ) = ∫(0dp)dΩ_act
∂Vol∂d(dd,((u,p),d),φ) = ∫(0dd ⋅ d)dΩ_act
∂Vol∂xhi = (∂Vol∂up,∂Vol∂d)

## Staggered operators
op = StaggeredNonlinearFEOperator([res_fluid,res_solid],[jac_fluid_newton,jac_solid],X,Y)
state_map = StaggeredNonlinearFEStateMap(op,∂Rk∂xhi,V_φ,U_reg,φh;adjoint_jacobians=[jac_fluid_newton,jac_solid])
J_sm = GridapTopOpt.StaggeredStateParamMap(J_comp,∂Jcomp∂xhi,state_map)
C_sm = map(((Ci,∂Ci),) -> GridapTopOpt.StaggeredStateParamMap(Ci,∂Ci,state_map),[(Vol,∂Vol∂xhi),])
pcfs = PDEConstrainedFunctionals(J_sm,C_sm,state_map;analytic_dC=[dVol])

## Evolution Method
evo = CutFEMEvolve(V_φ,Ω,dΩ_act,hₕ;max_steps,γg=γg_evo)
reinit1 = StabilisedReinit(V_φ,Ω,dΩ_act,hₕ;stabilisation_method=ArtificialViscosity(1.0))
reinit2 = StabilisedReinit(V_φ,Ω,dΩ_act,hₕ;stabilisation_method=GridapTopOpt.InteriorPenalty(V_φ,γg=1.0))
reinit = GridapTopOpt.MultiStageStabilisedReinit([reinit1,reinit2])
ls_evo = UnfittedFEEvolution(evo,reinit)

reinit!(ls_evo,φh_nondesign)

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
  γ=γ_evo,verbose=true,constraint_names=[:Vol],converged,has_oscillations)
for (it,(uh,ph,dh),φh) in optimiser
  GC.gc()
  if iszero(it % iter_mod)
    writevtk(Ω_act,path*"Omega_act_$it",
      cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ph"=>ph,"dh"=>dh])
    writevtk(Ω.Ωf,path*"Omega_f_$it",
      cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
    writevtk(Ω.Ωs,path*"Omega_s_$it",
      cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
  end
  write_history(path*"/history.txt",optimiser.history)

  φ = get_free_dof_values(φh)
  φ .= min.(φ,get_free_dof_values(φh_nondesign))
  reinit!(ls_evo,φh)
end
it = get_history(optimiser).niter; uh,ph,dh = get_state(pcfs)
writevtk(Ω_act,path*"Omega_act_$it",
  cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ph"=>ph,"dh"=>dh])
writevtk(Ω.Ωf,path*"Omega_f_$it",
  cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
writevtk(Ω.Ωs,path*"Omega_s_$it",
  cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])