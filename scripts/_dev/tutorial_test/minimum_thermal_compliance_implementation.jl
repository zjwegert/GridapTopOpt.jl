using Gridap, LevelSetTopOpt

# FE parameters
order = 1                                               # Finite element order
xmax = ymax = 1.0                                       # Domain size
dom = (0,xmax,0,ymax)                                   # Bounding domain
el_size = (200,200)                                     # Mesh partition size
prop_Γ_N = 0.2                                          # Γ_N size parameter
prop_Γ_D = 0.2                                          # Γ_D size parameter
f_Γ_N(x) = (x[1] ≈ xmax &&                              # Γ_N indicator function
  ymax/2-ymax*prop_Γ_N/2 - eps() <= x[2] <= ymax/2+ymax*prop_Γ_N/2 + eps())
f_Γ_D(x) = (x[1] ≈ 0.0 &&                               # Γ_D indicator function
  (x[2] <= ymax*prop_Γ_D + eps() || x[2] >= ymax-ymax*prop_Γ_D - eps()))
# FD parameters
γ = 0.1                                                 # HJ equation time step coefficient
γ_reinit = 0.5                                          # Reinit. equation time step coefficient
max_steps = floor(Int,minimum(el_size)/10)              # Max steps for advection
tol = 1/(5order^2)/minimum(el_size)                     # Advection tolerance
# Problem parameters
κ = 1                                                   # Diffusivity
g = 1                                                   # Heat flow in
vf = 0.4                                                # Volume fraction constraint
lsf_func = initial_lsf(4,0.2)                           # Initial level set function
iter_mod = 10                                           # Output VTK files every 10th iteration
path = "./results/tut1/"                                # Output path
mkpath(path)                                            # Create path
# Model
model = CartesianDiscreteModel(dom,el_size);
update_labels!(1,model,f_Γ_D,"Gamma_D")
update_labels!(2,model,f_Γ_N,"Gamma_N")
# Triangulation and measures
Ω = Triangulation(model)
Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
dΩ = Measure(Ω,2*order)
dΓ_N = Measure(Γ_N,2*order)
# Spaces
reffe = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"])
U = TrialFESpace(V,0.0)
V_φ = TestFESpace(model,reffe)
V_reg = TestFESpace(model,reffe;dirichlet_tags=["Gamma_N"])
U_reg = TrialFESpace(V_reg,0)
# Level set and interpolator
φh = interpolate(lsf_func,V_φ)
interp = SmoothErsatzMaterialInterpolation(η = 2*maximum(get_el_Δ(model)))
I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ
# Weak formulation
a(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*κ*∇(u)⋅∇(v))dΩ
l(v,φ,dΩ,dΓ_N) = ∫(g*v)dΓ_N
state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh,dΩ,dΓ_N)
# Objective and constraints
J(u,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*κ*∇(u)⋅∇(u))dΩ
dJ(q,u,φ,dΩ,dΓ_N) = ∫(κ*∇(u)⋅∇(u)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
vol_D = sum(∫(1)dΩ)
C(u,φ,dΩ,dΓ_N) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
dC(q,u,φ,dΩ,dΓ_N) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
pcfs = PDEConstrainedFunctionals(J,[C],state_map,analytic_dJ=dJ,analytic_dC=[dC])
# Velocity extension
α = 4*maximum(get_el_Δ(model))
a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)
# Finite difference scheme
scheme = FirstOrderStencil(length(el_size),Float64)
ls_evo = HamiltonJacobiEvolution(scheme,model,V_φ,tol,max_steps)
# Optimiser
optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;γ,γ_reinit,verbose=true,constraint_names=[:Vol])
# Solve
for (it,uh,φh) in optimiser
  data = ["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi)|"=>(norm ∘ ∇(φh)),"uh"=>uh]
  iszero(it % iter_mod) && (writevtk(Ω,path*"struc_$it",cellfields=data);GC.gc())
  write_history(path*"/history.txt",get_history(optimiser))
end
# Final structure
it = get_history(optimiser).niter; uh = get_state(pcfs)
writevtk(Ω,path*"struc_$it",cellfields=["phi"=>φh,
  "H(phi)"=>(H ∘ φh),"|nabla(phi)|"=>(norm ∘ ∇(φh)),"uh"=>uh])