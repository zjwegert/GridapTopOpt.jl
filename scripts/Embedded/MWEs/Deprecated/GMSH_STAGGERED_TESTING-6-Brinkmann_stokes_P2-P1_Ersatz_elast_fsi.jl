using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapGmsh
using GridapTopOpt

using GridapSolvers

#############

path = "./results/GMSH_STAGGERED_TESTING-6-Brinkmann_stokes_P2-P1_Ersatz_elast_fsi/results/"
mkpath(path)

γ_evo = 0.1
max_steps = 20
vf = 0.025
α_coeff = 1
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

# mfs_VQ = BlockMultiFieldStyle(2,(1,1)) #<- allows to specify block structure for first problem
VQ = MultiFieldFESpace([V,Q])#;style=mfs_VQ)
UP = MultiFieldFESpace([U,P])#;style=mfs_VQ)

mfs = BlockMultiFieldStyle(2,(2,1))
X = MultiFieldFESpace([U,P,R];style=mfs)

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

a_fluid((),(u,p),(v,q)) =
  ∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p)) * Ω.dΩf +
  ∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p) + (γ ∘ hₕ)*u⋅v) * Ω.dΩs
l_fluid((),(v,q)) = 0.0

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
a_solid(((u,p),),d,s) = ∫(ε(s) ⊙ (σ ∘ ε(d)))Ω.dΩs +
  ∫(ϵ*(ε(s) ⊙ (σ ∘ ε(d))))Ω.dΩf # Ersatz

function l_solid(((u,p),),s)
  n_AD = get_normal_vector(Ω.Γ)
  return ∫(σf_n(u,p,n_AD) ⋅ s)Ω.dΓ
end

## FEOperator way
fluid_op = AffineFEOperator((x,y)->a_fluid((),x,y),y->l_fluid((),y),UP,VQ)
uh_1,ph_1 = solve(fluid_op)
solid_op = AffineFEOperator((x,y)->a_solid(((uh_1,ph_1),),x,y),y->l_solid(((uh_1,ph_1),),y),R,T)
dh_1 = solve(solid_op)

## Staggered
op = GridapSolvers.StaggeredAffineFEOperator([a_fluid,a_solid],[l_fluid,l_solid],[UP,R],[VQ,T])


xh = zero(X)
# lsolver = CGSolver(JacobiLinearSolver();rtol=1.e-12,verbose=true)
solver = GridapSolvers.StaggeredFESolver(fill(LUSolver(),2))
xh,cache = solve!(xh,solver,op)
uh_2,ph_2,dh_2 = xh

writevtk(Ω.Ωf,path*"Omega_f",
  cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
writevtk(Ω.Ωs,path*"Omega_s",
  cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])

norm(get_free_dof_values(uh_1)-get_free_dof_values(uh_2),Inf)
norm(get_free_dof_values(ph_1)-get_free_dof_values(ph_2),Inf)
norm(get_free_dof_values(dh_1)-get_free_dof_values(dh_2),Inf)

## Old way
function old_main()
  _X = MultiFieldFESpace([U,P,R])
  _Y = MultiFieldFESpace([V,Q,T])

  # Terms
  _a_fluid((u,p),(v,q)) =
    ∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p)) * Ω.dΩf +
    ∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p) + (γ ∘ hₕ)*u⋅v ) * Ω.dΩs

  ## Structure
  _a_solid(d,s) = ∫(ε(s) ⊙ (σ ∘ ε(d)))Ω.dΩs +
    ∫(ϵ*(ε(s) ⊙ (σ ∘ ε(d))))Ω.dΩf # Ersatz

  ## Full problem
  function a_coupled((u,p,d),(v,q,s))
    n_AD = get_normal_vector(Ω.Γ)
    return _a_fluid((u,p),(v,q)) + _a_solid(d,s) +
      ∫(-σf_n(u,p,n_AD) ⋅ s)Ω.dΓ
  end
  l_coupled((v,q,s)) = 0.0

  return AffineFEOperator(a_coupled,l_coupled,_X,_Y)
end

op_old = old_main()

_xh = solve(op_old)
_uh,_ph,_dh = _xh

norm(get_free_dof_values(uh)-get_free_dof_values(_uh),Inf)
norm(get_free_dof_values(ph)-get_free_dof_values(_ph),Inf)
norm(get_free_dof_values(dh)-get_free_dof_values(_dh),Inf)



########
_f1((),x,ϕ) = ϕ
_f2((x,),y,ϕ) = ϕ
_f3((x,y,),z,ϕ) = ϕ

_fvec0 = [_f1,_f2,_f3]

_fvec = map(x->((xs,u)->x(xs,u,0)),_fvec0)

_fvec[1]((),1)
_fvec[2]((1,),2)
_fvec[3]((1,1),2)