using Gridap, Gridap.Adaptivity, Gridap.Geometry, Gridap.MultiField
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt, GridapSolvers
using GridapSolvers, GridapSolvers.BlockSolvers

# Example of shape and topology optimisation of a two-phase diffusion problem described by two level-set functions.
# See https://github.com/zjwegert/Wegert_et_al_2026_MP for further examples.

# The formulation is based on the following paper:
# > Zachary J. Wegert, Martin Berggren, and Vivien J. Challis (2026). "Shape calculus and automatic differentiation
#   for multi-phase level-set topology optimisation with unfitted finite elements". arXiv:...

n = 25
vf = 0.05
α_coeff = 3
iter_mod = 1
path = "results/aniso_vf$vf/"
mkpath(path)

# Model and some refinement
base_model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1),(n,n)))
ref_model = refine(base_model, refinement_method = "barycentric")
ref_model = refine(ref_model)
ref_model = refine(ref_model)
model = get_model(ref_model)
h = minimum(get_element_diameters(model))
max_steps = round(Int,1/h/10)
f_Γ_D(x) = (x[1]-0.5)^2 + (x[2]-0.5)^2 <= 0.025^2
f_Γ_N(x) = ((x[1] ≈ 0 || x[1] ≈ 1) && (0.45 <= x[2] <= 0.55 + eps())) ||
  ((x[2] ≈ 0 || x[2] ≈ 1) && (0.45 - eps() <= x[1] <= 0.55))
update_labels!(1,model,f_Γ_D,"Omega_D")
update_labels!(2,model,f_Γ_N,"Gamma_N")
writevtk(model,path*"model")

# Level-set function space and derivative regularisation space
reffe_scalar = ReferenceFE(lagrangian,Float64,1)
V_regs = [TestFESpace(model,reffe_scalar) for _ in 1:2]
V_reg = MultiFieldFESpace(V_regs;style=BlockMultiFieldStyle())
U_reg = MultiFieldFESpace(TrialFESpace.(V_reg.spaces);style=BlockMultiFieldStyle())
V_φ = MultiFieldFESpace([TestFESpace(model,reffe_scalar) for _ in 1:2])

# Level-set function
f1 = (x,y,a) -> -cos(6π*(x-1/12))*cos(6π*(y-1/12))-a
f2 = (x,y,a) -> -cos(6π*(x-3/12))*cos(6π*(y-1/12))-a
f3 = (x,y,r) -> (x-0.5)^2 + (y-0.5)^2 - r^2
f((x,y),a1,a2,r) = max(f1(x,y,a1),f2(x,y,a2))
φh = interpolate([x->f(x,0.9,0.9,0.06),x->f(x,0.4,0.4,0.06)],V_φ)

# Level-set function for fixed Dirichlet region
V_φ_diri = TestFESpace(model,reffe_scalar)
f_diri((x,y),) = (x-0.5)^2 + (y-0.5)^2 - (0.025+h/2)^2
φh_diri = interpolate(f_diri,V_φ_diri)
GridapTopOpt.correct_ls!(φh_diri)
geo_diri = DiscreteGeometryFromFEFunction(φh_diri,model,name="φ_diri")

# Check LS
GridapTopOpt.correct_ls!(φh)

# Triangulations and measures
Ω_bg = Triangulation(model)
Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
dΩ_bg = Measure(Ω_bg,2)
dΓ_N = Measure(Γ_N,2)
vol_D = sum(∫(1)dΩ_bg)

reinits = [StabilisedReinitialiser(V_φ[i],dΩ_bg,h;stabilisation_method=ArtificialViscosity(0.75)) for i in 1:2]
evos = [CutFEMEvolver(V_φ[i],dΩ_bg,h;max_steps,γg=0.1) for i in 1:2]
ls_evo = MultiLevelSetEvolution(evos,reinits,V_φ;reuse_cache=false);
reinit!(ls_evo,φh)

function compute_geo(φh1,φh2)
  geo1 = DiscreteGeometryFromFEFunction(φh1,model,name="φ1")
  geo2 = DiscreteGeometryFromFEFunction(φh2,model,name="φ2")
  Ω1 = setdiff(setdiff(geo1,geo2),geo_diri,name="Ω1")
  Ω3 = union(intersect(geo1,geo2),geo_diri,name="Ω3")
  return cut(PolytopalLevelSetCutter(),model,Ω1 ∪ Ω3)
end

# Construct collection of triangulations and measures for the two-phase problem
# Here, we set compute_cut=false to avoid recomputing the cut geometry for each level-set function, since we will compute it manually in the closure.
Ωs = EmbeddedCollection(model,φh;compute_cut=false) do φh
  φh1, φh2 = φh
  cutgeo = compute_geo(φh1,φh2)
  # Physical triangulations
  Ω1 = DifferentiableTriangulation(cutgeo,"Ω1")
  Ω3 = DifferentiableTriangulation(cutgeo,"Ω3")
  Γ13 = DifferentiableEmbeddedBoundary(cutgeo,"Ω1","Ω3")
  # Ghost triangulations
  Γg1 = GhostSkeleton(cutgeo,"Ω1")
  Γg3 = GhostSkeleton(cutgeo,"Ω3")
  # Active triangulations
  Ω1act = Triangulation(cutgeo,ACTIVE,"Ω1")
  Ω3act = Triangulation(cutgeo,ACTIVE,"Ω3")
  # Isolated volumes
  φ1 = get_free_dof_values(φh1)
  φ2 = get_free_dof_values(φh2)
  φ_diri = get_free_dof_values(φh_diri)
  φ_Ω1orΩ3 = min.(max.(φ1,-φ2),max.(φ1,φ2),φ_diri);
  φ_Ω1orΩ3_cv = get_cell_dof_values(FEFunction(V_φ[1],φ_Ω1orΩ3))
  χ,_ = get_isolated_volumes_mask_polytopal(model,φ_Ω1orΩ3_cv,["Omega_D"])
  (;
    :Ω1  => Ω1, :Ω1act => Ω1act, :dΩ1 => Measure(Ω1,2),
    :Ω3  => Ω3, :Ω3act => Ω3act, :dΩ3 => Measure(Ω3,2),
    :Γ13 => Γ13, :dΓ13 => Measure(Γ13,2), :n_Γ13 => get_normal_vector(Γ13),
    :Γg1 => Γg1, :dΓg1 => Measure(Γg1,2), :n_Γg1 => get_normal_vector(Γg1),
    :Γg3 => Γg3, :dΓg3 => Measure(Γg3,2), :n_Γg3 => get_normal_vector(Γg3),
    :χ => χ,cutgeo
  )
end

# Weak form
using Gridap.TensorValues
A1 = SymTensorValue(5,0.0,1)
A2 = SymTensorValue(1,0.0,5)
γg = 0.1*maximum(A1)
λ = 10^2*maximum(A1)/h
κ1(n) = n⋅A2⋅n/(n⋅A1⋅n + n⋅A2⋅n)
κ2(n) = n⋅A1⋅n/(n⋅A1⋅n + n⋅A2⋅n)

jump_u(u1,u2) = u1 - u2
mean_q(u1,u2,n) = (κ1 ∘ n)*(A1⋅∇(u1)) + (κ2 ∘ n)*(A2⋅∇(u2))

function a((u1,u2),(v1,v2),(φh1,φh2))
  # Compute normal
  n_Γ13 = get_normal_vector(Ωs.Γ13)
  n_Γg1 = Ωs.n_Γg1; n_Γg3 = Ωs.n_Γg3
  return ∫( ∇(v1)⋅(A1⋅∇(u1)) )Ωs.dΩ1 + ∫( ∇(v2)⋅(A2⋅∇(u2)) )Ωs.dΩ3 +
    ∫(λ*jump_u(v1,v2)*jump_u(u1,u2)
      - n_Γ13⋅mean_q(u1,u2,n_Γ13)*jump_u(v1,v2)
      - n_Γ13⋅mean_q(v1,v2,n_Γ13)*jump_u(u1,u2) )Ωs.dΓ13 +
    ∫( (γg*h)*jump(n_Γg1⋅∇(v1))*jump(n_Γg1⋅∇(u1)) )Ωs.dΓg1 +
    ∫( (γg*h)*jump(n_Γg3⋅∇(v2))*jump(n_Γg3⋅∇(u2)) )Ωs.dΓg3 +
    ∫(Ωs.χ*v1*u1)Ωs.dΩ1 + ∫(Ωs.χ*v2*u2)Ωs.dΩ3
end

l((v1,v2),(φh1,φh2)) = ∫(v1+v2)dΓ_N

# Optimisation functionals
J((u1,u2),φ) = ∫(∇(u1)⋅(A1⋅∇(u1)))Ωs.dΩ1 + ∫(∇(u2)⋅(A2⋅∇(u2)))Ωs.dΩ3
Vol_Ω1(u,φ) = ∫(1)Ωs.dΩ1
Vol_Ω3(u,φ) = ∫(1)Ωs.dΩ3

# FE operators
state_collection = EmbeddedCollection(model,φh;compute_cut=false) do _φh
  update_collection!(Ωs,_φh)
  V1 = TestFESpace(Ωs.Ω1act,reffe_scalar;dirichlet_tags=["Omega_D"])
  U1 = TrialFESpace(V1,0.0)
  V2 = TestFESpace(Ωs.Ω3act,reffe_scalar;dirichlet_tags=["Omega_D"])
  U2 = TrialFESpace(V2,0.0)
  V = MultiFieldFESpace([V1,V2])
  U = MultiFieldFESpace([U1,U2])
  state_map = AffineFEStateMap(a,l,U,V,V_φ)
  (;
    :state_map => state_map,
    :J => StateParamMap(J,state_map),
    :C => map(Ci -> StateParamMap(Ci,state_map),[Vol_Ω1,Vol_Ω3])
  )
end

function φ_to_jc(φ)
  GridapTopOpt.ignore_derivatives() do
    update_collection!(state_collection,FEFunction(V_φ,φ))
  end
  u = state_collection.state_map(φ)
  j = state_collection.J(u,φ)
  c1 = state_collection.C[1](u,φ)/vol_D - vf
  c2 = state_collection.C[2](u,φ)/vol_D - vf
  return [j,c1,c2]
end

function dCi!(Ci,dC,φ)
  φh = FEFunction(V_φ,φ)
  _dC(q) = gradient(φ -> Ci(nothing,φ),φh)
  Gridap.FESpaces.assemble_vector!(_dC,dC,V_φ)
end

pcfs = CustomPDEConstrainedFunctionals(φ_to_jc,2,
    analytic_dC=[(dC,φ)->dCi!(Vol_Ω1,dC,φ), (dC,φ)->dCi!(Vol_Ω3,dC,φ)])

# Hilbertian extension-regularisation problems
α = α_coeff*h
a_hilb((p1,p2),(q1,q2)) = ∫(α^2*∇(p1)⋅∇(q1) + p1*q1 + α^2*∇(p2)⋅∇(q2) + p2*q2)dΩ_bg;
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg;ls=BlockDiagonalSolver([LUSolver(),LUSolver()]))

# Optimiser
optimiser = HilbertianProjection(pcfs,ls_evo,vel_ext,φh;verbose=true,constraint_names=[:Vol_Ω1,:Vol_Ω3],γ=0.1,ls_γ_max=0.1)
for (it,_,φh) in optimiser
  if iszero(it % iter_mod)
    uh = get_state(state_collection.state_map)
    writevtk(Ω_bg,path*"Omega_$it",cellfields=[
      "φ1"=>φh[1],"|∇(φ1)|"=>(norm ∘ ∇(φh[1])),"uh1"=>uh[1],
      "φ2"=>φh[2],"|∇(φ2)|"=>(norm ∘ ∇(φh[2])),"uh2"=>uh[2],"χ"=>Ωs.χ
    ])
    writevtk(Ωs.Ω1,path*"Omega1_$it",cellfields=["uh"=>uh[1]])
    writevtk(Ωs.Ω3,path*"Omega3_$it",cellfields=["uh"=>uh[2]])
    writevtk(Ωs.Γg1,path*"Γg1_$it")
    writevtk(Ωs.Γg3,path*"Γg3_$it")
    writevtk(Ωs.Γ13,path*"Gamma13_$it",cellfields=["jump"=>jump_u(uh[1],uh[2])])
  end
  write_history(path*"/history.txt",optimiser.history)
end
it = get_history(optimiser).niter; uh = get_state(state_collection.state_map)
writevtk(Ω_bg,path*"Omega_$it",cellfields=[
  "φ1"=>φh[1],"|∇(φ1)|"=>(norm ∘ ∇(φh[1])),"uh1"=>uh[1],
  "φ2"=>φh[2],"|∇(φ2)|"=>(norm ∘ ∇(φh[2])),"uh2"=>uh[2],"χ"=>Ωs.χ
])
writevtk(Ωs.Ω1,path*"Omega1_$it",cellfields=["uh"=>uh[1]])
writevtk(Ωs.Ω3,path*"Omega3_$it",cellfields=["uh"=>uh[2]])
writevtk(Ωs.Γ13,path*"Gamma13_$it",cellfields=["jump"=>jump_u(uh[1],uh[2])])

