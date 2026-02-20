module FSICutFEMTest
using Test
using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapDistributed
using PartitionedArrays
using GridapPETSc
using GridapTopOpt

using GridapTopOpt: StaggeredStateParamMap

function main(distribute,mesh_partition)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  model, geo_params = build_model(ranks,mesh_partition,4*5,5;b=1/5,w=1/5)

  # Triangulation
  Ω_act = Triangulation(model)
  hₕ = get_element_diameter_field(model)
  hmin = minimum(get_element_diameters(model))

  # Params
  vf = 0.03
  γ_evo =  0.1
  max_steps = 1/hmin/10
  D = 2

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Omega_NonDesign","Gamma_Bottom"])
  U_reg = TrialFESpace(V_reg)

  _e = 1/3*hmin
  L,H,x0,l,w,a,b = geo_params
  f0((x,y),W,H) = max(2/W*abs(x-x0),1/(H/2+1)*abs(y-H/2+1))-1
  f1((x,y),q,r) = - cos(q*π*x)*cos(q*π*y)/q - r/q
  fin(x) = f0(x,l*(1+5_e),a*(1+5_e))
  fsolid(x) = min(f0(x,l*(1+_e),b*(1+_e)),f0(x,w*(1+_e),a*(1+_e)))
  fholes((x,y),q,r) = max(f1((x,y),q,r),f1((x-1/q,y),q,r))
  lsf(x) = min(max(fin(x),fholes(x,4,0.2)),fsolid(x))
  φh = interpolate(lsf,V_φ)

  # Check LS
  GridapTopOpt.correct_ls!(φh)

  # Setup integration meshes and measures
  order = 1
  degree = 2*(order+1)

  dΩ_act = Measure(Ω_act,degree)
  Γf_D = BoundaryTriangulation(model,tags="Gamma_f_D")
  dΓf_D = Measure(Γf_D,degree)
  Ω = EmbeddedCollection(model,φh) do cutgeo,cutgeo_facets,_φh
    Ωs = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
    Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
    Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
    Γg = GhostSkeleton(cutgeo)
    Ω_act_s = Triangulation(cutgeo,ACTIVE)
    Ω_act_f = Triangulation(cutgeo,ACTIVE_OUT)
    Γi = SkeletonTriangulation(cutgeo_facets,ACTIVE_OUT)
    # Isolated volumes
    φ_cell_values = map(get_cell_dof_values,local_views(_φh))
    ψ_s,_ = get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Gamma_s_D"])
    _,ψ_f = get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Gamma_f_D"])
    (;
      :Ωs       => Ωs,
      :dΩs      => Measure(Ωs,degree),
      :Ωf       => Ωf,
      :dΩf      => Measure(Ωf,degree),
      :Γg       => Γg,
      :dΓg      => Measure(Γg,degree),
      :n_Γg     => get_normal_vector(Γg),
      :Γ        => Γ,
      :dΓ       => Measure(Γ,degree),
      :n_Γ      => get_normal_vector(Γ),
      :Ω_act_s  => Ω_act_s,
      :dΩ_act_s => Measure(Ω_act_s,degree),
      :Ω_act_f  => Ω_act_f,
      :dΩ_act_f => Measure(Ω_act_f,degree),
      :Γi       => Γi,
      :dΓi      => Measure(Γi,degree),
      :n_Γi     => get_normal_vector(Γi),
      :ψ_s      => ψ_s,
      :ψ_f      => ψ_f,
    )
  end

  # Setup spaces
  uin(x) = VectorValue(16x[2]*(H-x[2]),0.0)
  uin_dot_e1(x) = uin(x)⋅VectorValue(1.0,0.0)

  reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1,space=:P)
  reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

  function build_spaces(Ω_act_s,Ω_act_f)
    # Test spaces
    V = TestFESpace(Ω_act_f,reffe_u,conformity=:H1,dirichlet_tags=["Gamma_f_D","Gamma_Top","Gamma_Bottom","Gamma_s_D"])
    Q = TestFESpace(Ω_act_f,reffe_p,conformity=:L2)
    T = TestFESpace(Ω_act_s,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_s_D","Gamma_Bottom"])

    # Trial spaces
    U = TrialFESpace(V,[uin,VectorValue(0.0,0.0),VectorValue(0.0,0.0),VectorValue(0.0,0.0)])
    P = TrialFESpace(Q)
    R = TrialFESpace(T)

    # Multifield spaces
    UP = MultiFieldFESpace([U,P])
    VQ = MultiFieldFESpace([V,Q])
    return (UP,VQ),(R,T)
  end

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
  γ_Nu_h = γ_Nu ∘ hₕ
  γ_u_h = mean(γ_u ∘ hₕ)
  γ_p_h = mean(γ_p ∘ hₕ)

  # Terms
  _I = one(SymTensorValue{D,Float64})
  σf(u,p) = 2μ*ε(u) - p*_I
  a_Ω(∇u,∇v) = μ*(∇u ⊙ ∇v)
  b_Ω(div_v,p) = -p*(div_v)
  ab_Γ(u,∇u,v,∇v,p,q,n) = n ⋅ ( - μ*(∇u ⋅ v + ∇v ⋅ u) + v*p + u*q) + γ_Nu_h*(u⋅v)
  ju(∇u,∇v) = γ_u_h*(jump(Ω.n_Γg ⋅ ∇u) ⋅ jump(Ω.n_Γg ⋅ ∇v))
  jp(p,q) = γ_p_h*(jump(p) * jump(q))
  v_ψ(p,q) = k_p * Ω.ψ_f*p*q

  function a_fluid((),(u,p),(v,q),φ)
    ∇u = ∇(u); ∇v = ∇(v);
    div_u = ∇⋅u; div_v = ∇⋅v
    n_Γ = -get_normal_vector(Ω.Γ)
    return ∫(a_Ω(∇u,∇v) + b_Ω(div_v,p) + b_Ω(div_u,q) + v_ψ(p,q))Ω.dΩf +
      ∫(ab_Γ(u,∇u,v,∇v,p,q,n_Γ))Ω.dΓ +
      ∫(ju(∇u,∇v))Ω.dΓg - ∫(jp(p,q))Ω.dΓi
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
  α_Gd = 1e-7
  k_d = 1.0
  γ_Gd(h) = α_Gd*(λs + μs)*h^3
  γ_Gd_h = mean(γ_Gd ∘ hₕ)
  # Terms
  σ(ε) = λs*tr(ε)*_I + 2*μs*ε
  a_s_Ω(d,s) = ε(s) ⊙ (σ ∘ ε(d)) # Elasticity
  j_s_k(d,s) = γ_Gd_h*(jump(Ω.n_Γg ⋅ ∇(s)) ⋅ jump(Ω.n_Γg ⋅ ∇(d)))
  v_s_ψ(d,s) = (k_d*Ω.ψ_s)*(d⋅s) # Isolated volume term

  function a_solid(((u,p),),d,s,φ)
    return ∫(a_s_Ω(d,s))Ω.dΩs +
      ∫(j_s_k(d,s))Ω.dΓg +
      ∫(v_s_ψ(d,s))Ω.dΩs
  end
  function l_solid(((u,p),),s,φ)
    n = -get_normal_vector(Ω.Γ)
    return ∫(-(1-Ω.ψ_s)*(n ⋅ σf(u,p)) ⋅ s)Ω.dΓ
  end

  ## Optimisation functionals
  vol_D = sum(∫(1)dΩ_act)
  iso_vol_frac(φ) = ∫(1000Ω.ψ_s)Ω.dΩs
  J_comp(((u,p),d),φ) = ∫(ε(d) ⊙ (σ ∘ ε(d)))Ω.dΩs + iso_vol_frac(φ)
  Vol(((u,p),d),φ) = ∫(1/vol_D)Ω.dΩs - ∫(vf/vol_D)dΩ_act
  dVol(q,(u,p,d),φ) = ∫(-1/vol_D*q/(abs(Ω.n_Γ ⋅ ∇(φ))))Ω.dΓ

  ## Staggered operators
  state_collection = EmbeddedCollection_in_φh(model,φh) do _φh
    update_collection!(Ω,_φh)
    (UP,VQ),(R,T) = build_spaces(Ω.Ω_act_s,Ω.Ω_act_f)
    solver = StaggeredFESolver([LUSolver(),LUSolver()]);
    op = StaggeredAffineFEOperator([a_fluid,a_solid],[l_fluid,l_solid],[UP,R],[VQ,T])
    state_map = StaggeredAffineFEStateMap(op,V_φ,_φh;solver,adjoint_solver=solver)
    (;
      :state_map => state_map,
      :J => StaggeredStateParamMap(J_comp,state_map),
      :C => map(Ci -> StaggeredStateParamMap(Ci,state_map),[Vol,])
    )
  end

  pcf = EmbeddedPDEConstrainedFunctionals(state_collection;analytic_dC=[dVol])

  ## Evolution Method
  evolve_nls = NewtonSolver(LUSolver();maxiter=1,verbose=i_am_main(ranks))
  reinit_nls = NewtonSolver(LUSolver();maxiter=20,rtol=1.e-14,verbose=i_am_main(ranks))

  evo = CutFEMEvolver(V_φ,Ω,dΩ_act,hₕ;max_steps,γg=0.01,ode_ls=LUSolver(),ode_nl=evolve_nls)
  reinit = StabilisedReinitialiser(V_φ,Ω,dΩ_act,hₕ;stabilisation_method=ArtificialViscosity(0.5),nls=reinit_nls)
  ls_evo = LevelSetEvolution(evo,reinit)

  ## Hilbertian extension-regularisation problems
  α_coeff = γ_evo*max_steps
  _α(hₕ) = (α_coeff*hₕ)^2
  a_hilb(p,q) =∫((_α ∘ hₕ)*∇(p)⋅∇(q) + p*q)dΩ_act;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg;ls=LUSolver())

  ## Optimiser
  optimiser = AugmentedLagrangian(pcf,ls_evo,vel_ext,φh;
    γ=γ_evo,verbose=i_am_main(ranks),constraint_names=[:Vol])

  # Do a few iterations
  vars, state = iterate(optimiser)
  vars, state = iterate(optimiser,state)
  true
end

function build_model(ranks,mesh_partition,nx,ny;L=2.0,H=0.5,x0=0.5,l=0.4,w=0.05,a=0.3,b=0.05)
  geo_params = (;L,H,x0,l,w,a,b)
  base_model = UnstructuredDiscreteModel(CartesianDiscreteModel(ranks,mesh_partition,(0,L,0,H),(nx,ny)))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = get_model(ref_model)
  f_Γ_Top(x) = x[2] == H
  f_Γ_Bottom(x) = x[2] == 0.0
  f_Γ_D(x) = x[1] == 0.0
  f_Γ_N(x) = x[1] == L
  f_box(x) = 0.0 <= x[2] <= a + eps() && (x0 - l/2 - eps() <= x[1] <= x0 + l/2 + eps())
  f_NonDesign(x) = ((x0 - w/2 - eps() <= x[1] <= x0 + w/2 + eps() && 0.0 <= x[2] <= a + eps()) ||
    (x0 - l/2 - eps() <= x[1] <= x0 + l/2 + eps() && 0.0 <= x[2] <= b + eps()))
  update_labels!(1,model,f_Γ_Top,"Gamma_Top")
  update_labels!(2,model,f_Γ_Bottom,"Gamma_Bottom")
  update_labels!(3,model,f_Γ_D,"Gamma_f_D")
  update_labels!(4,model,f_Γ_N,"Gamma_f_N")
  update_labels!(5,model,f_box,"RefineBox")
  update_labels!(6,model,f_NonDesign,"Omega_NonDesign")
  update_labels!(7,model,x->f_NonDesign(x) && f_Γ_Bottom(x),"Gamma_s_D")
  return model, geo_params
end


with_mpi() do distribute
  @test main(distribute,(2,2))
end

end