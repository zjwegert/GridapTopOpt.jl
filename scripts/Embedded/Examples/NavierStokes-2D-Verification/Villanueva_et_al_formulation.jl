using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt

using LinearAlgebra
LinearAlgebra.norm(x::VectorValue,p::Real) = norm(x.data,p)

path = "./results/NavierStokes-2D-Verification/"
mkpath(path)

R = 0.1 # Disk radius
x0,y0 = 0.5,0.2 # Disk centroid

model = GmshDiscreteModel((@__DIR__)*"/fsi/gmsh/mesh_finer.msh")
writevtk(model,path*"model")

Ω_act = Triangulation(model)
hₕ = CellField(get_element_diameters(model),Ω_act)

# Cut the background model
reffe_scalar = ReferenceFE(lagrangian,Float64,1)
V_φ = TestFESpace(model,reffe_scalar)

φh = interpolate(((x,y),)->-sqrt((x-x0)^2+(y-y0)^2)+R,V_φ)
writevtk(get_triangulation(φh),path*"initial_lsf",cellfields=["φ"=>φh,"h"=>hₕ])

# Setup integration meshes and measures
order = 1
degree = 2*(order+1)

dΩ_act = Measure(Ω_act,degree)
Γf_D = BoundaryTriangulation(model,tags="Gamma_f_D")
dΓf_D = Measure(Γf_D,degree)
dΓf_N = Measure(Γf_N,degree)
Ω = EmbeddedCollection(model,φh) do cutgeo,_
  Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
  Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
  Γg = GhostSkeleton(cutgeo)
  Ω_act_f = Triangulation(cutgeo,ACTIVE_OUT)
  (;
    :Ωf      => Ωf,
    :dΩf     => Measure(Ωf,degree),
    :Γg      => Γg,
    :dΓg     => Measure(Γg,degree),
    :n_Γg    => get_normal_vector(Γg),
    :Γ       => Γ,
    :dΓ      => Measure(Γ,degree),
    :Ω_act_f => Ω_act_f,
    :dΩ_act_f => Measure(Ω_act_f,degree),
    :ψ_f     => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_f_D"];IN_is=OUT),
    # :ψ_f     => GridapTopOpt.get_isolated_volumes_mask_without_cuts(cutgeo,["Gamma_f_D"];IN_is=OUT)
  )
end
writevtk(get_triangulation(φh),path*"initial_islands",cellfields=["ψ_f"=>Ω.ψ_f])

# Setup spaces
uin(x) = VectorValue(16x[2]*(H-x[2]),0.0)

reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
reffe_p = ReferenceFE(lagrangian,Float64,order,space=:P)
reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

V = TestFESpace(Ω_act_f,reffe_u,conformity=:H1,
  dirichlet_tags=["Gamma_f_D","Gamma_NoSlipTop","Gamma_NoSlipBottom"])
Q = TestFESpace(Ω_act_f,reffe_p,conformity=:H1)

# Trial spaces
U = TrialFESpace(V,[uin,VectorValue(0.0,0.0),VectorValue(0.0,0.0)])
P = TrialFESpace(Q)

# Multifield spaces
X = MultiFieldFESpace([U,P])
Y = MultiFieldFESpace([V,Q])

### Weak form

## Fluid
# Properties
Re = 60 # Reynolds number
ρ = 1.0 # Density
cl = 2R # Characteristic length
u0_max = sum(∫(uin⋅VectorValue(1.0,0.0))dΓf_D)/sum(∫(1)dΓf_D)
μ = ρ*cl*u0_max/Re # Viscosity
ν = μ/ρ # Kinematic viscosity

# Stabilization parameters
α_Nu   = 1000
α_SUPG = 1/3
α_GPμ  = 0.05
α_GPp  = 0.05
α_GPu  = 0.05

γ_Nu(h,u)    = α_Nu*(μ/h + ρ*norm(u,Inf)/6) # (Eqn. 13, Villanueva and Maute, 2017)
τ_SUPG(h,u)  = α_SUPG*((2norm(u)/h)^2 + 9*(4ν/h^2)^2)^-0.5 # (Eqn. 31, Peterson et al., 2018)
τ_PSPG(h,u)  = τ_SUPG(h,u) # (Sec. 3.2.2, Peterson et al., 2018)
γ_GPμ(h)     = α_GPμ*μ*h # (Eqn. 32, Villanueva and Maute, 2017)
γ_GPp(h,u)   = α_GPp*(μ/h+ρ*norm(u,Inf)/6)^-1*h^2 # (Eqn. 35, Villanueva and Maute, 2017)
γ_GPu(h,un)  = α_GPu*ρ*abs(un)*h^2 # (Eqn. 37, Villanueva and Maute, 2017)
γ_GPu(h,u,n) = (γ_GPu ∘ (h.plus,(u⋅n).plus) + γ_GPu ∘ (h.minus,(u⋅n).minus))/2
k_p          = 1.0 # (Villanueva and Maute, 2017)

# Terms
βp = 1; βμ = 1;

δ = one(SymTensorValue{D,Float64})
σ_f(ε,p) = -p*δ + 2μ*ε
σ_f_β(ε,p) = -βp*p*δ + βμ*2μ*ε

conv(u,∇u) = (∇u') ⋅ u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

r_conv(u,v) = ρ*v ⋅ (conv∘(u,∇(u)))
r_Ωf((u,p),(v,q)) = ε(v) ⊙ (σ_f ∘ (ε(u),p)) + q*(∇⋅u)
r_Γ((u,p),(v,q),n,w) = -v⋅((σ_f ∘ (ε(u),p))⋅n) - u⋅((σ_f_β ∘ (ε(v),q))⋅n) + (γ_Nu ∘ (hₕ,w))*u⋅v
r_ψ(p,q) = k_p * Ω.ψ_f*p*q
r_SUPG((u,p),(v,q),w) = ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (u,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅
  (ρ*(conv∘(u,∇(u))) + ∇(p) - μ*Δ(u))
r_SUPG_picard((u,p),(v,q),w) = ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (w,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅
  (ρ*(conv∘(w,∇(u))) + ∇(p) - μ*Δ(u))
r_GP_μ(u,v) = mean(γ_GPμ ∘ hₕ)*jump(Ω.n_Γg ⋅ ∇(u)) ⋅ jump(Ω.n_Γg ⋅ ∇(v))
r_GP_p(p,q,w) = mean(γ_GPp ∘ (hₕ,w))*jump(Ω.n_Γg ⋅ ∇(p)) * jump(Ω.n_Γg ⋅ ∇(q))
r_GP_u(u,v,w,n) = γ_GPu(hₕ,w,n)*jump(Ω.n_Γg ⋅ ∇(u)) ⋅ jump(Ω.n_Γg ⋅ ∇(v))

dr_conv(u,du,v) = ρ*v ⋅ (dconv∘(du,∇(du),u,∇(u)))
dr_SUPG((u,p),(du,dp),(v,q),w) =
  ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (du,∇(v))))⋅(ρ*(conv∘(u,∇(u))) + ∇(p) - μ*Δ(u)) +
  ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (u,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅(ρ*(dconv∘(du,∇(du),u,∇(u))) + ∇(dp) - μ*Δ(du))

function res_fluid((u,p),(v,q))
  n_Γ = get_normal_vector(Ω.Γ)
  return ∫(r_conv(u,v) + r_Ωf((u,p),(v,q)))Ω.dΩf +
    ∫(r_SUPG((u,p),(v,q),u))Ω.dΩ_act_f +
    ∫(r_ψ(p,q))Ω.dΩf +
    ∫(r_Γ((u,p),(v,q),n_Γ,u))Ω.dΓ +
    ∫(r_GP_μ(u,v) + r_GP_p(p,q,u) + r_GP_u(u,v,u,Ω.n_Γg))Ω.dΓg
end

function jac_fluid_picard((u,p),(du,dp),(v,q))
  n_Γ = get_normal_vector(Ω.Γ)
  return ∫(ρ*v ⋅ (conv∘(u,∇(du))) + r_Ωf((du,dp),(v,q)))Ω.dΩf +
    ∫(r_SUPG_picard((du,dp),(v,q),u))Ω.dΩ_act_f +
    ∫(r_ψ(dp,q))Ω.dΩf +
    ∫(r_Γ((du,dp),(v,q),n_Γ,u))Ω.dΓ +
    ∫(r_GP_μ(du,v) + r_GP_p(dp,q,u) + r_GP_u(du,v,u,Ω.n_Γg))Ω.dΓg
end

function jac_fluid_newton((u,p),(du,dp),(v,q))
  n_Γ = get_normal_vector(Ω.Γ)
  return ∫(dr_conv(u,du,v) + r_Ωf((du,dp),(v,q)))Ω.dΩf +
    ∫(dr_SUPG((u,p),(du,dp),(v,q),u))Ω.dΩ_act_f +
    ∫(r_ψ(dp,q))Ω.dΩf +
    ∫(r_Γ((du,dp),(v,q),n_Γ,u))Ω.dΓ +
    ∫(r_GP_μ(du,v) + r_GP_p(dp,q,u) + r_GP_u(du,v,u,Ω.n_Γg))Ω.dΓg
end

op = FEOperator(res_fluid,jac_fluid_newton,X,Y)
solver = NewtonSolver(LUSolver();maxiter=100,rtol=1.e-14,verbose=true)
uh,ph,dh = solve(solver,op);
# writevtk(Ω_act,path*"Omega_act",
#   cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ph"=>ph,"dh"=>dh,"ψ_s"=>Ω.ψ_s,"ψ_f"=>Ω.ψ_f])
# writevtk(Ω.Ωf,path*"Omega_f",
#   cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
# writevtk(Ω.Ωs,path*"Omega_s",
#   cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])

mass_flow_rate = sum(∫(uh⋅get_normal_vector(Ω.Γ))Ω.dΓ)
surface_force = sum(∫(σ_f(ε(uh),ph) ⋅ get_normal_vector(Ω.Γ))Ω.dΓ)
pressure_difference = sum(∫(ph+ρ/2*(uh⋅uh))dΓf_D-∫(ph+ρ/2*(uh⋅uh))dΓf_N)