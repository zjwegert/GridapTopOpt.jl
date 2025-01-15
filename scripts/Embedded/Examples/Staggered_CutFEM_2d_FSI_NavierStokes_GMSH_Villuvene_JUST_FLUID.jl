using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt

using LinearAlgebra
LinearAlgebra.norm(x::VectorValue,p::Real) = norm(x.data,p)
Base.abs(x::VectorValue) = VectorValue(abs.(x.data))
Base.sign(x::VectorValue) = VectorValue(sign.(x.data))

path = "./results/Staggered_CutFEM_2d_FSI_NavierStokes_GMSH_Villuvene/"
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

model = GmshDiscreteModel((@__DIR__)*"/fsi/gmsh/mesh_finer.msh")
writevtk(model,path*"model")

Ω_act = Triangulation(model)
hₕ = CellField(get_element_diameters(model),Ω_act)
hmin = minimum(get_element_diameters(model))

# Cut the background model
reffe_scalar = ReferenceFE(lagrangian,Float64,1)
V_φ = TestFESpace(model,reffe_scalar)

# φh = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.15)^2)-0.1,V_φ)

_e = 1e-3
f0((x,y),W,H) = max(2/W*abs(x-x0),1/(H/2+1)*abs(y-H/2+1))-1
f1((x,y),q,r) = - cos(q*π*x)*cos(q*π*y)/q - r/q
fin(x) = f0(x,l*(1+5_e),a*(1+5_e))
fsolid(x) = min(f0(x,l*(1+_e),b*(1+_e)),f0(x,w*(1+_e),a*(1+_e)))
fholes((x,y),q,r) = max(f1((x,y),q,r),f1((x-1/q,y),q,r))
φf(x) = min(max(fin(x),fholes(x,25,0.2)),fsolid(x))
# φf(x) = min(max(fin(x),fholes(x,22,0.6)),fsolid(x))
# φh = interpolate(φf,V_φ)

# Bite test
φf2(x) = max(φf(x),-(max(2/0.2*abs(x[1]-0.3),2/0.2*abs(x[2]-0.3))-1))
φh = interpolate(φf2,V_φ)

writevtk(get_triangulation(φh),path*"initial_lsf",cellfields=["φ"=>φh,"h"=>hₕ])

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
  Ω_act_s = Triangulation(cutgeo,ACTIVE)
  Ω_act_f = Triangulation(cutgeo,ACTIVE_OUT)
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
    :dΩ_act_s => Ω_act_s,
    :Ω_act_s => Measure(Ω_act_s,degree),
    :Ω_act_f => Ω_act_f,
    :dΩ_act_f => Measure(Ω_act_f,degree),
    :ψ_s     => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_s_D"];IN_is=IN),
    :ψ_f     => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_f_D"];IN_is=OUT)
  )
end
writevtk(get_triangulation(φh),path*"initial_islands",cellfields=["ψ_s"=>Ω.ψ_s,"ψ_f"=>Ω.ψ_f])

# Setup spaces
uin(x) = VectorValue(16x[2]*(H-x[2]),0.0)

reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
reffe_p = ReferenceFE(lagrangian,Float64,order,space=:P)

function build_spaces(Ω_act_f)
  # Test spaces
  V = TestFESpace(Ω_act_f,reffe_u,conformity=:H1,
    dirichlet_tags=["Gamma_f_D","Gamma_NoSlipTop","Gamma_NoSlipBottom","Gamma_s_D"])
  Q = TestFESpace(Ω_act_f,reffe_p,conformity=:H1)

  # Trial spaces
  U = TrialFESpace(V,[uin,VectorValue(0.0,0.0),VectorValue(0.0,0.0),VectorValue(0.0,0.0)])
  P = TrialFESpace(Q)

  # Multifield spaces
  X = MultiFieldFESpace([U,P])
  Y = MultiFieldFESpace([V,Q])
  return X,Y
end
X,Y = build_spaces(Ω.Ω_act_f)

### Weak form

## Fluid
# Properties
Re = 60 # Reynolds number
ρ = 1.0 # Density
cl = a # Characteristic length
u0_max = maximum(abs,get_dirichlet_dof_values(X[1]))
μ = ρ*cl*u0_max/Re # Viscosity
ν = μ/ρ # Kinematic viscosity

# Stabilization parameters
α_Nu   = 1000
α_SUPG = 1/3
α_GPμ  = 0.5
α_GPp  = 0.05
α_GPu  = 0.5

γ_Nu(h,u)    = α_Nu*(μ/h + ρ*norm(u,Inf)/6) # (Eqn. 13, Villanueva and Maute, 2017)
τ_SUPG(h,u)  = α_SUPG*((2norm(u)/h)^2 + 9*(4ν/h^2)^2)^-0.5 # (Eqn. 31, Peterson et al., 2018)
τ_PSPG(h,u)  = τ_SUPG(h,u) # (Sec. 3.2.2, Peterson et al., 2018)
γ_GPμ(h)     = α_GPμ*μ*h # (Eqn. 32, Villanueva and Maute, 2017)
γ_GPp(h,u)   = α_GPp*(μ/h+ρ*norm(u,Inf)/6)^-1*h^2 # (Eqn. 35, Villanueva and Maute, 2017)
γ_GPu(h,un) = α_GPu*ρ*abs(un)*h^2 # (Eqn. 37, Villanueva and Maute, 2017)
γ_GPu(h,u,n) = (γ_GPu ∘ (h.plus,(u⋅n).plus) + γ_GPu ∘ (h.minus,(u⋅n).minus))/2
k_p = 1.0 # (Villanueva and Maute, 2017)

# Terms
βp = 1; βμ = 1;

δ = one(SymTensorValue{D,Float64})
σ_f(ε,p) = -p*δ + 2μ*ε
σ_f_β(ε,p) = -βp*p*δ + βμ*2μ*ε

conv(u,∇u) = (∇u') ⋅ u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

r_conv(u,v) = ρ*v ⋅ (conv∘(u,∇(u)))
r_Ωf((u,p),(v,q)) = ε(v) ⊙ (σ_f ∘ (ε(u),p)) + q*(∇⋅u) # (Eqn. 6 without conv, Villanueva and Maute, 2017)
r_Γ((u,p),(v,q),n,w) = -v⋅((σ_f ∘ (ε(u),p))⋅n) - u⋅((σ_f_β ∘ (ε(v),q))⋅n) + (γ_Nu ∘ (hₕ,w))*u⋅v # (Eqn. 12, Villanueva and Maute, 2017)
r_ψ(p,q) = k_p * Ω.ψ_f*p*q # (Eqn. 15, Villanueva and Maute, 2017)
r_SUPG((u,p),(v,q),w) = ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (u,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅
  (ρ*(conv∘(u,∇(u))) + ∇(p) - μ*Δ(u))
r_SUPG_picard((u,p),(v,q),w) = ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (w,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅
  (ρ*(conv∘(w,∇(u))) + ∇(p) - μ*Δ(u))
r_GP_μ(u,v) = mean(γ_GPμ ∘ hₕ)*jump(Ω.n_Γg ⋅ ∇(u)) ⋅ jump(Ω.n_Γg ⋅ ∇(v))
r_GP_p(p,q,w) = mean(γ_GPp ∘ (hₕ,w))*jump(Ω.n_Γg ⋅ ∇(p)) * jump(Ω.n_Γg ⋅ ∇(q))
r_GP_u(u,v,w,n) = γ_GPu(hₕ,w,n)*jump(Ω.n_Γg ⋅ ∇(u)) ⋅ jump(Ω.n_Γg ⋅ ∇(v))

dr_conv(u,du,v) = ρ*v ⋅ (dconv∘(du,∇(du),u,∇(u)))
dr_SUPG((u,p),(du,dp),(v,q),w) =
  ((τ_SUPG ∘ (hₕ,w))*(dconv∘(du,∇(du),u,∇(u))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅(ρ*(conv∘(u,∇(u))) + ∇(p) - μ*Δ(u)) +
  ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (u,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅(ρ*(dconv∘(du,∇(du),u,∇(u))) + ∇(dp) - μ*Δ(du))

function res((u,p),(v,q))
  n_Γ = get_normal_vector(Ω.Γ)
  return ∫(r_conv(u,v) + r_Ωf((u,p),(v,q)))Ω.dΩf +
    ∫(r_SUPG((u,p),(v,q),u))Ω.dΩ_act_f +
    ∫(r_ψ(p,q))Ω.dΩf +
    ∫(r_Γ((u,p),(v,q),n_Γ,u))Ω.dΓ +
    ∫(r_GP_μ(u,v))Ω.dΓg + ∫(r_GP_p(p,q,u))Ω.dΓg + ∫(r_GP_u(u,v,u,Ω.n_Γg))Ω.dΓg
end

function jac_picard((u,p),(du,dp),(v,q))
  n_Γ = get_normal_vector(Ω.Γ)
  return ∫(ρ*v ⋅ (conv∘(u,∇(du))) + r_Ωf((du,dp),(v,q)))Ω.dΩf +
    ∫(r_SUPG_picard((du,dp),(v,q),u))Ω.dΩ_act_f +
    ∫(r_ψ(dp,q))Ω.dΩf +
    ∫(r_Γ((du,dp),(v,q),n_Γ,u))Ω.dΓ +
    ∫(r_GP_μ(du,v))Ω.dΓg + ∫(r_GP_p(dp,q,u))Ω.dΓg + ∫(r_GP_u(du,v,u,Ω.n_Γg))Ω.dΓg
end

function jac_newton((u,p),(du,dp),(v,q))
  n_Γ = get_normal_vector(Ω.Γ)
  return ∫(dr_conv(u,du,v) + r_Ωf((du,dp),(v,q)))Ω.dΩf +
    ∫(dr_SUPG((u,p),(du,dp),(v,q),u))Ω.dΩ_act_f +
    ∫(r_ψ(dp,q))Ω.dΩf +
    ∫(r_Γ((du,dp),(v,q),n_Γ,u))Ω.dΓ +
    ∫(r_GP_μ(du,v))Ω.dΓg + ∫(r_GP_p(dp,q,u))Ω.dΓg + ∫(r_GP_u(du,v,u,Ω.n_Γg))Ω.dΓg
end

op = FEOperator(res,jac_picard,X,Y)
nls = NewtonSolver(LUSolver();maxiter=50,rtol=1.e-8,verbose=true)
uh,ph = solve(nls,op)

writevtk(Ω.Ωf,path*"Omega_f",cellfields=["uh"=>uh,"ph"=>ph])