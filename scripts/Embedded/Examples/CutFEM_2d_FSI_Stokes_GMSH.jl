using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers
using GridapGmsh
using GridapTopOpt

path = "./results/CutFEM_2d_FSI_Stokes_GMSH/"
mkpath(path)

γ_evo = 0.2
max_steps = 20
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

model = GmshDiscreteModel((@__DIR__)*"/fsi/gmsh/mesh.msh")
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
order = 1
degree = 2*order

Ω_bg = Triangulation(model)
dΩ_bg = Measure(Ω_bg,2*order)
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
    :Ω_act_s => Triangulation(cutgeo,ACTIVE),
    :Ω_act_f => Triangulation(cutgeo,ACTIVE_OUT),
    :χ_s     => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_s_D"];IN_is=IN),
    :χ_f     => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_f_D"];IN_is=OUT)
  )
end
writevtk(get_triangulation(φh),path*"initial_islands",cellfields=["χ_s"=>Ω.χ_s,"χ_f"=>Ω.χ_f])

# Setup spaces
uin(x) = VectorValue(16x[2]*(H-x[2]),0.0)

reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
reffe_p = ReferenceFE(lagrangian,Float64,order,space=:P)
reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

function build_spaces(Ω_act_s,Ω_act_f)
  # Test spaces
  V = TestFESpace(Ω_act_f,reffe_u,conformity=:H1,
    dirichlet_tags=["Gamma_f_D","Gamma_NoSlipTop","Gamma_NoSlipBottom","Gamma_s_D"])
  Q = TestFESpace(Ω_act_f,reffe_p,conformity=:H1)
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

### Weak form

## Fluid
# Properties
Re = 60 # Reynolds number
ρ = 1.0 # Density
cl = H # Characteristic length
u0_max = maximum(abs,get_dirichlet_dof_values(init_X[1]))
μ = ρ*cl*u0_max/Re # Viscosity
ν = μ/ρ # Kinematic viscosity

# Stabilization parameters
α_Nu    = 100
α_PSUPG = 1/3
α_GPμ   = 0.5
α_GPp   = 0.05

γ_Nu(h)    = α_Nu*μ/h         # (Eqn. 13, Villanueva and Maute, 2017)
τ_PSUPG(h) = α_PSUPG*(h^2/4ν) # (Eqn. 32, Peterson et al., 2018)
γ_GPμ(h)   = α_GPμ*μ*h        # (Eqn. 32, Villanueva and Maute, 2017)
γ_GPp(h)   = α_GPp*h^3/μ      # (Eqn. 35, Villanueva and Maute, 2017)
k_p = 1.0                     # (Villanueva and Maute, 2017)

# Terms
σf_n(u,p,n) = μ*∇(u) ⋅ n - p*n
a_Ω(u,v) = μ*(∇(u) ⊙ ∇(v)) # (Eqn. 3.3, Massing et al., 2014)
b_Ω(v,p) = - (∇ ⋅ v)*p # (Eqn. 3.4, Massing et al., 2014)
c_Ω(p,q) = (τ_PSUPG ∘ hₕ)*1/ρ*∇(p) ⋅ ∇(q) # (Eqn. 3.7, Massing et al., 2014)
a_Γ(u,v,n) = - (n ⋅ ∇(u)) ⋅ v - u ⋅ (n ⋅ ∇(v)) + (γ_Nu ∘ hₕ)*u⋅v # (Eqn. 3.9, Massing et al., 2014))
b_Γ(v,p,n) = (n ⋅ v)*p # (Eqn. 3.10, Massing et al., 2014)
i_Γg(u,v) = mean(γ_GPμ ∘ hₕ)*jump(Ω.n_Γg ⋅ ∇(u)) ⋅ jump(Ω.n_Γg ⋅ ∇(v)) # (Eqn. 3.11, Massing et al., 2014)
j_Γg(p,q) = mean(γ_GPp ∘ hₕ)*jump(Ω.n_Γg ⋅ ∇(p)) * jump(Ω.n_Γg ⋅ ∇(q)) # (Eqn. 3.12, Massing et al., 2014)
v_χ(p,q) = k_p * Ω.χ_f*p*q # (Isolated volume term, Eqn. 15, Villanueva and Maute, 2017)

function a_fluid((),(u,p),(v,q),φ)
  n_Γ = get_normal_vector(Ω.Γ)
  return ∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p)-c_Ω(p,q) )Ω.dΩf + # Volume terms
    ∫( a_Γ(u,v,n_Γ)+b_Γ(u,q,n_Γ)+b_Γ(v,p,n_Γ) )Ω.dΓ +    # Interface terms
    ∫( i_Γg(u,v) - j_Γg(p,q) )Ω.dΓg +                    # Ghost penalty terms
    ∫( v_χ(p,q) )Ω.dΩf                                        # Isolated volume term
end

l_fluid((),(v,q),φ) = ∫(0q)Ω.dΩf

## Structure
# Material parameters
function lame_parameters(E,ν)
  λ = (E*ν)/((1+ν)*(1-2*ν))
  μ = E/(2*(1+ν))
  (λ, μ)
end
λs, μs = lame_parameters(0.1,0.05)
# Stabilization
α_Gd = 0.1
k_d = 1.0
γ_Gd(h) = α_Gd*(λs + 2μs)*h^3
# Terms
σ(ε) = λs*tr(ε)*one(ε) + 2*μs*ε
a_s_Ω(s,d) = ε(s) ⊙ (σ ∘ ε(d)) # Elasticity
j_s_k(s,d) = mean(γ_Gd ∘ hₕ)*jump(Ω.n_Γg ⋅ ∇(s)) ⋅ jump(Ω.n_Γg ⋅ ∇(d)) # (Eqn. 3.11, Burman et al., 2018)
v_s_χ(s,d) = k_d*Ω.χ_s*d⋅s # Isolated volume term

function a_solid(((u,p),),d,s,φ)
  return ∫(a_s_Ω(s,d))Ω.dΩs + ∫(j_s_k(s,d))Ω.dΓg + ∫(v_s_χ(s,d))Ω.dΩs
end
function l_solid(((u,p),),s,φ)
  n = get_normal_vector(Ω.Γ)
  return ∫(σf_n(u,p,n) ⋅ s)Ω.dΓ
end

## Optimisation functionals
J_pres(((u,p),d),φ) = ∫(p)dΓf_D - ∫(p)dΓf_N
J_comp(((u,p),d),φ) = ∫(ε(d) ⊙ (σ ∘ ε(d)))Ω.dΩs
Vol(((u,p),d),φ) = ∫(vol_D)Ω.dΩs - ∫(vf/vol_D)dΩ_bg
dVol(q,((u,p),d),φ) = ∫(-1/vol_D*q/(norm ∘ (∇(φ))))Ω.dΓ

## Staggered operators
state_collection = GridapTopOpt.EmbeddedCollection_in_φh(model,φh) do _φh
  update_collection!(Ω,_φh)
  X,Y = build_spaces(Ω.Ω_act_s,Ω.Ω_act_f)
  op = StaggeredAffineFEOperator([a_fluid,a_solid],[l_fluid,l_solid],X,Y)
  state_map = StaggeredAffineFEStateMap(op,V_φ,U_reg,_φh)
  (;
    :state_map => state_map,
    :J => GridapTopOpt.StaggeredStateParamMap(J_comp,state_map),
    :C => map(Ci -> GridapTopOpt.StaggeredStateParamMap(Ci,state_map),[Vol,])
  )
end

pcf = EmbeddedPDEConstrainedFunctionals(state_collection)
evaluate!(pcf,φh)

# op = GridapTopOpt.get_staggered_operator_at_φ(_op,φh)
# xh = solve(StaggeredFESolver(fill(LUSolver(),2)),op);
# writevtk(Ω_bg,path*"Omega_act_0",
#   cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>xh[1],"ph"=>xh[2],"dh"=>xh[3]])
# writevtk(Ω.Ωf,path*"Omega_f_0",
#   cellfields=["uh"=>xh[1],"ph"=>xh[2],"dh"=>xh[3]])
# writevtk(Ω.Ωs,path*"Omega_s_0",
#   cellfields=["uh"=>xh[1],"ph"=>xh[2],"dh"=>xh[3]])
# writevtk(Ω.Γ,path*"Gamma_0",cellfields=["σ⋅n"=>(σ ∘ ε(xh[3]))⋅get_normal_vector(Ω.Γ),"σf_n"=>σf_n(xh[1],xh[2],get_normal_vector(Ω.Γ))])

