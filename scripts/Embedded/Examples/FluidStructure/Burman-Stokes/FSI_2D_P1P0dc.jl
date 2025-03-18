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
  global γg_evo =  0.05
end

path = "./results/FSI_2D_Burman_P1P0dc/"
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

model = GmshDiscreteModel((@__DIR__)*"/../Meshes/mesh_finer.msh")
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
φh_nondesign = interpolate(fsolid,V_φ)

# Ensure values at DoFs are non-zero to satify assumptions for derivatives
φ = get_free_dof_values(φh)
idx = findall(isapprox(0.0;atol=1e-10),φ)
if length(idx)>0
  println("    Correcting level values at $(length(idx)) nodes")
  φ[idx] .+= 1e-10
end

# Setup integration meshes and measures
order = 1
degree = 2*(order+1)

dΩ_act = Measure(Ω_act,degree)
Γf_D = BoundaryTriangulation(model,tags="Gamma_f_D")
Γf_N = BoundaryTriangulation(model,tags="Gamma_f_N")
dΓf_D = Measure(Γf_D,degree)
dΓf_N = Measure(Γf_N,degree)
Ω = EmbeddedCollection(model,φh) do cutgeo,cutgeo_facets,_φh
  Ωs = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
  Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
  Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
  Γg = GhostSkeleton(cutgeo)
  Ω_act_s = Triangulation(cutgeo,ACTIVE)
  Ω_act_f = Triangulation(cutgeo,ACTIVE_OUT)
  Γi = SkeletonTriangulation(cutgeo_facets,ACTIVE_OUT)
  # Isolated volumes
  φ_cell_values = get_cell_dof_values(_φh)
  ψ_s,_ = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Gamma_s_D"])
  _,ψ_f = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Gamma_f_D"])

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
      :n_Γ        => get_normal_vector(Γ), # Note, need to recompute inside obj/constraints to compute derivs
    :Ω_act_s => Ω_act_s,
    :dΩ_act_s => Measure(Ω_act_s,degree),
    :Ω_act_f => Ω_act_f,
    :dΩ_act_f => Measure(Ω_act_f,degree),
    :Γi => Γi,
    :dΓi => Measure(Γi,degree),
    :n_Γi    => get_normal_vector(Γi),
    :ψ_s     => ψ_s,
    :ψ_f     => ψ_f,
  )
end
writevtk(get_triangulation(φh),path*"initial_islands",cellfields=["φh"=>φh,"ψ_s"=>Ω.ψ_s,"ψ_f"=>Ω.ψ_f])

# Setup spaces
uin(x) = VectorValue(16x[2]*(H-x[2]),0.0)
uin_dot_e1(x) = uin(x)⋅VectorValue(1.0,0.0)

reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
reffe_p = ReferenceFE(lagrangian,Float64,order-1,space=:P)
reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

function build_spaces(Ω_act_s,Ω_act_f)
  # Test spaces
  V = TestFESpace(Ω_act_f,reffe_u,conformity=:H1,
    dirichlet_tags=["Gamma_f_D","Gamma_NoSlipTop","Gamma_NoSlipBottom","Gamma_s_D"])
  Q = TestFESpace(Ω_act_f,reffe_p,conformity=:L2)
  T = TestFESpace(Ω_act_s,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_s_D"])

  # Trial spaces
  U = TrialFESpace(V,[uin,VectorValue(0.0,0.0),VectorValue(0.0,0.0),VectorValue(0.0,0.0)])
  P = TrialFESpace(Q)
  R = TrialFESpace(T)

  # Multifield spaces
  mfs = BlockMultiFieldStyle(2,(2,1))
  X = MultiFieldFESpace([U,P,R];style=mfs)
  Y = MultiFieldFESpace([V,Q,T];style=mfs)
  return X,Y
end
init_X,_ = build_spaces(Ω.Ω_act_s,Ω.Ω_act_f)

### Weak form

## Fluid
# Properties
Re = 60 # Reynolds number
ρ = 1.0 # Density
cl = a # Characteristic length
u0_max = sum(∫(uin_dot_e1)dΓf_D)/sum(∫(1)dΓf_D)
μ = ρ*cl*u0_max/Re # Viscosity
ν = μ/ρ # Kinematic viscosity

# Stabilization parameters
α_Nu = 100
α_u  = 0.1
α_p  = 0.25

γ_Nu(h) = α_Nu*μ/h
γ_u(h) = α_u*μ*h
γ_p(h) = α_p*h/μ
k_p    = 1.0 # (Villanueva and Maute, 2017)

# Terms
σf_n(u,p,n) = μ*∇(u) ⋅ n - p*n
a_Ω(u,v) = μ*(∇(u) ⊙ ∇(v))
b_Ω(v,p) = -p*(∇⋅v)
a_Γ(u,v,n) = - μ*(n⋅∇(u)) ⋅ v - μ*(n⋅∇(v)) ⋅ u + (γ_Nu ∘ hₕ)*(u⋅v)
b_Γ(v,p,n) = (n⋅v)*p
ju(u,v) = mean(γ_u ∘ hₕ)*jump(Ω.n_Γg ⋅ ∇(u)) ⋅ jump(Ω.n_Γg ⋅ ∇(v))
jp(p,q) = mean(γ_p ∘ hₕ)*jump(p) * jump(q)
v_ψ(p,q) = k_p * Ω.ψ_f*p*q # (Isolated volume term, Eqn. 15, Villanueva and Maute, 2017)

function a_fluid((),(u,p),(v,q),φ)
  n_Γ = -get_normal_vector(Ω.Γ)
  return ∫(a_Ω(u,v) + b_Ω(v,p) + b_Ω(u,q) + v_ψ(p,q))Ω.dΩf +
    ∫(a_Γ(u,v,n_Γ) + b_Γ(v,p,n_Γ) + b_Γ(u,q,n_Γ))Ω.dΓ +
    ∫(ju(u,v))Ω.dΓg - ∫(jp(p,q))Ω.dΓi
end

l_fluid((),(v,q),φ) =  ∫(0q)Ω.dΩf

## Structure
# Material parameters
function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end
λs, μs = lame_parameters(0.1,0.05)
# Stabilization
α_Gd = 1e-3
k_d = 1.0
γ_Gd(h) = α_Gd*(λs + μs)*h^3
# Terms
σ(ε) = λs*tr(ε)*one(ε) + 2*μs*ε
a_s_Ω(d,s) = ε(s) ⊙ (σ ∘ ε(d)) # Elasticity
j_s_k(d,s) = mean(γ_Gd ∘ hₕ)*jump(Ω.n_Γg ⋅ ∇(s)) ⋅ jump(Ω.n_Γg ⋅ ∇(d))
v_s_ψ(d,s) = k_d*Ω.ψ_s*d⋅s # Isolated volume term

function a_solid(((u,p),),d,s,φ)
  return ∫(a_s_Ω(d,s) + v_s_ψ(d,s))Ω.dΩs + ∫(j_s_k(d,s))Ω.dΓg
end
function l_solid(((u,p),),s,φ)
  n = -get_normal_vector(Ω.Γ)
  return ∫(-σf_n(u,p,n) ⋅ s)Ω.dΓ
end

∂R2∂xh1((du,dp),((u,p),),d,s,φ) = -1*l_solid(((du,dp),),s,φ)
∂Rk∂xhi = ((∂R2∂xh1,),)

## Optimisation functionals
iso_vol_frac(φ) = ∫(Ω.ψ_s/vol_D)Ω.dΩs

J_comp(((u,p),d),φ) = ∫(ε(d) ⊙ (σ ∘ ε(d)))Ω.dΩs + iso_vol_frac(φ)
∂Jcomp∂up((du,dp),((u,p),d),φ) = ∫(0dp)Ω.dΩs
∂Jcomp∂d(dd,((u,p),d),φ) = ∫(2*ε(d) ⊙ (σ ∘ ε(dd)))Ω.dΩs
∂Jpres∂xhi = (∂Jcomp∂up,∂Jcomp∂d)

Vol(((u,p),d),φ) = ∫(vol_D)Ω.dΩs - ∫(vf/vol_D)dΩ_act
dVol(q,(u,p,d),φ) = ∫(-1/vol_D*q/(abs(Ω.n_Γ ⋅ ∇(φ))))Ω.dΓ
∂Vol∂up((du,dp),((u,p),d),φ) = ∫(0dp)dΩ_act
∂Vol∂d(dd,((u,p),d),φ) = ∫(0dd ⋅ d)dΩ_act
∂Vol∂xhi = (∂Vol∂up,∂Vol∂d)

## Staggered operators
state_collection = GridapTopOpt.EmbeddedCollection_in_φh(model,φh) do _φh
  update_collection!(Ω,_φh)
  X,Y = build_spaces(Ω.Ω_act_s,Ω.Ω_act_f)
  op = StaggeredAffineFEOperator([a_fluid,a_solid],[l_fluid,l_solid],X,Y)
  state_map = StaggeredAffineFEStateMap(op,∂Rk∂xhi,V_φ,U_reg,_φh)
  (;
    :state_map => state_map,
    :J => GridapTopOpt.StaggeredStateParamMap(J_comp,∂Jpres∂xhi,state_map),
    :C => map(((Ci,∂Ci),) -> GridapTopOpt.StaggeredStateParamMap(Ci,∂Ci,state_map),[(Vol,∂Vol∂xhi),])
  )
end

pcf = EmbeddedPDEConstrainedFunctionals(state_collection;analytic_dC=[dVol])

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
optimiser = AugmentedLagrangian(pcf,ls_evo,vel_ext,φh;
  γ=γ_evo,verbose=true,constraint_names=[:Vol],converged,has_oscillations)
try
  for (it,(uh,ph,dh),φh) in optimiser
    GC.gc()
    if iszero(it % iter_mod)
      writevtk(Ω_act,path*"Omega_act_$it",
        cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ph"=>ph,"dh"=>dh,
          "ψ_s"=>Ω.ψ_s,"ψ_f"=>Ω.ψ_f])
      writevtk(Ω.Ωf,path*"Omega_f_$it",
        cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
      writevtk(Ω.Ωs,path*"Omega_s_$it",
        cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
    end
    write_history(path*"/history.txt",optimiser.history)

    isolated_vol = sum(iso_vol_frac(φh))
    println(" --- Isolated volume: ",isolated_vol)

    _φ = get_free_dof_values(φh)
    _φ .= min.(_φ,get_free_dof_values(φh_nondesign))
    reinit!(ls_evo,φh)
  end
catch e
  println("Error: $e\nPrinting history and exiting...")
  writevtk(Ω_act,path*"Omega_act_errored",
    cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),
      "ψ_s"=>Ω.ψ_s,"ψ_f"=>Ω.ψ_f])
    writevtk(Ω.Ωf,path*"Omega_f_errored")
    writevtk(Ω.Ωs,path*"Omega_s_errored")
end
it = get_history(optimiser).niter; uh,ph,dh = get_state(pcf)
writevtk(Ω_act,path*"Omega_act_$it",
  cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ph"=>ph,"dh"=>dh])
writevtk(Ω.Ωf,path*"Omega_f_$it",
  cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
writevtk(Ω.Ωs,path*"Omega_s_$it",
  cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])