using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapTopOpt

using GridapDistributed,PartitionedArrays,GridapPETSc

if isassigned(ARGS,1)
  global γg_evo =  parse(Float64,ARGS[1])
else
  global γg_evo =  0.05
end

function test_mesh(model)
  grid_topology = Geometry.get_grid_topology(model)
  D = num_cell_dims(grid_topology)
  d = 0
  vertex_to_cells = Geometry.get_faces(grid_topology,d,D)
  bad_vertices = findall(i->i==0,map(length,vertex_to_cells))
  @assert isempty(bad_vertices) "Bad vertices detected: re-generate your mesh with a different resolution"
end

MUMPSSolver() = PETScLinearSolver(petsc_mumps_setup)

function petsc_mumps_setup(ksp)
  pc       = Ref{GridapPETSc.PETSC.PC}()
  mumpsmat = Ref{GridapPETSc.PETSC.Mat}()

  @check_error_code GridapPETSc.PETSC.KSPSetFromOptions(ksp[])

  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPPREONLY)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCLU)
  @check_error_code GridapPETSc.PETSC.PCFactorSetMatSolverType(pc[],GridapPETSc.PETSC.MATSOLVERMUMPS)
  @check_error_code GridapPETSc.PETSC.PCFactorSetUpMatSolverType(pc[])
  @check_error_code GridapPETSc.PETSC.PCFactorGetMatrix(pc[],mumpsmat)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  4, 1)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 28, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 1)
  # @check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[],  1, 0.00001) # relative thresh
  # @check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[], 3, 1.0e-6) # absolute thresh
  @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
end

CGAMGSolver() = PETScLinearSolver(gamg_ksp_setup)

function gamg_ksp_setup(;rtol=10^-8,maxits=100)

  function ksp_setup(ksp)
    pc = Ref{GridapPETSc.PETSC.PC}()

    rtol = PetscScalar(rtol)
    atol = GridapPETSc.PETSC.PETSC_DEFAULT
    dtol = GridapPETSc.PETSC.PETSC_DEFAULT
    maxits = PetscInt(maxits)

    @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPCG)
    @check_error_code GridapPETSc.PETSC.KSPSetTolerances(ksp[], rtol, atol, dtol, maxits)
    @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
    @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCGAMG)
    @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
  end

  return ksp_setup
end

function main(ranks,mesh_partition)
  path = "./results/FSI_3D_Burman_P1P0dc_Cart/"
  files_path = path*"data/"
  i_am_main(ranks) && mkpath(files_path)

  γ_evo = 0.2
  max_steps = 10 # Based on number of elements in vertical direction divided by 10
  vf = 0.025
  α_coeff = γ_evo*max_steps
  iter_mod = 10
  D = 3

  # Load gmsh mesh (Currently need to update mesh.geo and these values concurrently)
  L = 3.0;
  H = 1.0;
  x0 = 2;
  l = 1.0;
  w = 0.05;
  a = 0.7;
  b = 0.05;
  cw = 0.1;
  vol_D = L*H

  _model = CartesianDiscreteModel(ranks,mesh_partition,(0,4,0,1,0,1),(150,50,50))
  base_model = UnstructuredDiscreteModel(_model)
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)

  f_Γ_Left(x) = x[3] == 0.0
  f_Γ_Right(x) = x[3] == 1.0
  f_Γ_Top(x) = x[2] == 1.0
  f_Γ_Bottom(x) = x[2] == 0.0
  f_Γ_TopCorners(x) = x[2] == 1.0 && (x[3] == 0.0 || x[3] == 1.0)
  f_Γ_D(x) = x[1] == 0.0
  f_Γ_N(x) = x[1] == 4.0
  f_NonDesign(x) = ((x0 - w/2 - eps() <= x[1] <= x0 + w/2 + eps() && 0.0 <= x[2] <= a + eps() && cw - eps() <= x[3] <= H - cw + eps()) ||
    (x0 - l/2 - eps() <= x[1] <= x0 + l/2 + eps() && 0.0 <= x[2] <= b + eps() && cw - eps() <= x[3] <= H - cw + eps()))

  update_labels!(1,model,f_Γ_Left,"Gamma_Left")
  update_labels!(2,model,f_Γ_Right,"Gamma_Right")
  update_labels!(3,model,f_Γ_Top,"Gamma_Top")
  update_labels!(4,model,f_Γ_Bottom,"Gamma_Bottom")
  update_labels!(5,model,f_Γ_TopCorners,"Gamma_TopCorners")
  update_labels!(6,model,f_Γ_D,"Gamma_f_D")
  update_labels!(7,model,f_Γ_N,"Gamma_f_N")
  update_labels!(8,model,f_NonDesign,"Omega_NonDesign")
  update_labels!(9,model,x->f_NonDesign(x) && f_Γ_Bottom(x),"Gamma_s_D")
  writevtk(model,path*"model")

  Ω_act = Triangulation(model)
  hₕ = get_element_diameter_field(model)
  hmin = minimum(get_element_diameters(model))

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Omega_NonDesign","Gamma_s_D"])
  U_reg = TrialFESpace(V_reg)

  _e = 1/3*hmin
  f0((x,y,z),a,b) = max(2/a*abs(x-x0),1/(b/2+1)*abs(y-b/2+1),2/(H-2cw)*abs(z-H/2))-1
  f1((x,y,z),q,r) = - cos(q*π*x)*cos(q*π*y)*cos(q*π*z)/q - r/q
  fin(x) = f0(x,l*(1+_e),a*(1+_e))
  fsolid(x) = min(f0(x,l*(1+_e),b*(1+_e)),f0(x,w*(1+_e),a*(1+_e)))
  fholes((x,y,z),q,r) = max(f1((x,y,z),q,r),f1((x-1/q,y,z),q,r))
  lsf(x) = min(max(fin(x),fholes(x,5,0.5)),fsolid(x))
  φh = interpolate(lsf,V_φ)
  φh_nondesign = interpolate(fsolid,V_φ)

  _φ = get_free_dof_values(φh)
  map(local_views(_φ)) do φ
    idx = findall(isapprox(0.0;atol=eps()),φ)
    if !isempty(idx)
      i_am_main(ranks) && println("    Correcting level values at $(length(idx)) nodes")
    end
    φ[idx] .+= 100*eps(eltype(φ))
  end
  consistent!(_φ) |> wait

  writevtk(get_triangulation(φh),path*"initial_lsf",cellfields=["φh"=>φh])

  # Setup integration meshes and measures
  order = 1
  degree = 2*(order+1)

  dΩ_act = Measure(Ω_act,degree)
  Γf_D = BoundaryTriangulation(model,tags="Gamma_f_D")
  dΓf_D = Measure(Γf_D,degree)
  Ω = EmbeddedCollection(model,φh) do cutgeo,cutgeo_facets
    Ωs = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
    Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
    Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
    Γg = GhostSkeleton(cutgeo)
    Ω_act_s = Triangulation(cutgeo,ACTIVE)
    Ω_act_f = Triangulation(cutgeo,ACTIVE_OUT)
    Γi = SkeletonTriangulation(cutgeo_facets,ACTIVE_OUT)
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
      :Ω_act_s => Ω_act_s,
      :dΩ_act_s => Measure(Ω_act_s,degree),
      :Ω_act_f => Ω_act_f,
      :dΩ_act_f => Measure(Ω_act_f,degree),
      :Γi => Γi,
      :dΓi => Measure(Γi,degree),
      :n_Γi    => get_normal_vector(Γi),
      :ψ_s     => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_s_D"];groups=((GridapTopOpt.CUT,IN),OUT)),
      :ψ_f     => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_f_D"];groups=((GridapTopOpt.CUT,OUT),IN)),
    )
  end
  writevtk(get_triangulation(φh),path*"initial_islands",cellfields=["ψ_s"=>Ω.ψ_s,"ψ_f"=>Ω.ψ_f])

  # Setup spaces
  uin(x) = VectorValue(x[2],0.0,0.0)
  uin_dot_e1(x) = uin(x)⋅VectorValue(1.0,0.0,0.0)

  reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1,space=:P)
  reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

  function build_spaces(Ω_act_s,Ω_act_f)
    # Test spaces
    V = TestFESpace(Ω_act_f,reffe_u,conformity=:H1,
      dirichlet_tags=["Gamma_f_D","Gamma_s_D","Gamma_Bottom","Gamma_Top",
      "Gamma_Left","Gamma_Right","Gamma_TopCorners"],
      dirichlet_masks=[(true,true,true),(true,true,true),(true,true,true),
        (false,true,false),(false,false,true),(false,false,true),(false,true,true)])
    Q = TestFESpace(Ω_act_f,reffe_p,conformity=:L2)
    T = TestFESpace(Ω_act_s,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_s_D"])

    # Trial spaces
    U = TrialFESpace(V,[uin,[VectorValue(0.0,0.0,0.0) for _ = 1:6]...])
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

  # Terms
  _n(∇φ) = ∇φ/(10^-20+norm(∇φ)) # TODO: Currently required as diff through get_normal_vector is broken in distributed

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
    # n_Γ = -(_n ∘ ∇(φ))
    return ∫(a_Ω(u,v) + b_Ω(v,p) + b_Ω(u,q) + v_ψ(p,q))Ω.dΩf +
      ∫(a_Γ(u,v,n_Γ) + b_Γ(v,p,n_Γ) + b_Γ(u,q,n_Γ))Ω.dΓ +
      ∫(ju(u,v))Ω.dΓg - ∫(jp(p,q))Ω.dΓi
      # ∫(ju(u,v) + 0mean(φ))Ω.dΓg - ∫(jp(p,q) + 0mean(φ))Ω.dΓi
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
    return ∫(a_s_Ω(d,s))Ω.dΩs +
      # ∫(j_s_k(d,s) + 0mean(φ) + 0*jump(p*p))Ω.dΓg +
      ∫(j_s_k(d,s))Ω.dΓg +
      ∫(v_s_ψ(d,s))Ω.dΩs
  end
  function l_solid(((u,p),),s,φ)
    n = -get_normal_vector(Ω.Γ)
    # n = -(_n ∘ ∇(φ))
    return ∫(-σf_n(u,p,n) ⋅ s)Ω.dΓ
  end

  ∂R2∂xh1((du,dp),((u,p),),d,s,φ) = -1*l_solid(((du,dp),),s,φ)
  ∂Rk∂xhi = ((∂R2∂xh1,),)

  ## Optimisation functionals
  J_comp(((u,p),d),φ) = ∫(ε(d) ⊙ (σ ∘ ε(d)))Ω.dΩs
  ∂Jcomp∂up((du,dp),((u,p),d),φ) = ∫(0dp)Ω.dΩs
  ∂Jcomp∂d(dd,((u,p),d),φ) = ∫(2*ε(d) ⊙ (σ ∘ ε(dd)))Ω.dΩs
  ∂Jpres∂xhi = (∂Jcomp∂up,∂Jcomp∂d)

  Vol(((u,p),d),φ) = ∫(vol_D)Ω.dΩs - ∫(vf/vol_D)dΩ_act
  dVol(q,(u,p,d),φ) = ∫(-1/vol_D*q/(norm ∘ (∇(φ))))Ω.dΓ
  ∂Vol∂up((du,dp),((u,p),d),φ) = ∫(0dp)dΩ_act
  ∂Vol∂d(dd,((u,p),d),φ) = ∫(0dd ⋅ d)dΩ_act
  ∂Vol∂xhi = (∂Vol∂up,∂Vol∂d)

  ## Staggered operators
  fluid_ls = MUMPSSolver()
  elast_ls = MUMPSSolver()

  state_collection = GridapTopOpt.EmbeddedCollection_in_φh(model,φh) do _φh
    update_collection!(Ω,_φh)
    (UP,VQ),(R,T) = build_spaces(Ω.Ω_act_s,Ω.Ω_act_f)
    # elast_ls = ElasticitySolver(R;rtol=1.e-8,maxits=200)
    solver = StaggeredFESolver([fluid_ls,elast_ls]);
    op = StaggeredAffineFEOperator([a_fluid,a_solid],[l_fluid,l_solid],[UP,R],[VQ,T])
    state_map = StaggeredAffineFEStateMap(op,∂Rk∂xhi,V_φ,U_reg,_φh;solver,adjoint_solver=solver)
    (;
      :state_map => state_map,
      :J => GridapTopOpt.StaggeredStateParamMap(J_comp,∂Jpres∂xhi,state_map),
      :C => map(((Ci,∂Ci),) -> GridapTopOpt.StaggeredStateParamMap(Ci,∂Ci,state_map),[(Vol,∂Vol∂xhi),])
    )
  end

  pcf = EmbeddedPDEConstrainedFunctionals(state_collection;analytic_dC=[dVol])

  ## Evolution Method
  evolve_ls = MUMPSSolver()
  evolve_nls = NewtonSolver(evolve_ls;maxiter=1,verbose=i_am_main(ranks))
  reinit_nls = NewtonSolver(MUMPSSolver();maxiter=20,rtol=1.e-14,verbose=i_am_main(ranks))

  evo = CutFEMEvolve(V_φ,Ω,dΩ_act,hₕ;max_steps,γg=γg_evo,ode_ls=evolve_ls,ode_nl=evolve_nls)
  reinit1 = StabilisedReinit(V_φ,Ω,dΩ_act,hₕ;stabilisation_method=ArtificialViscosity(0.5),nls=reinit_nls)
  reinit2 = StabilisedReinit(V_φ,Ω,dΩ_act,hₕ;stabilisation_method=GridapTopOpt.InteriorPenalty(V_φ,γg=0.1),nls=reinit_nls)
  reinit = GridapTopOpt.MultiStageStabilisedReinit([reinit1,reinit2])
  ls_evo = UnfittedFEEvolution(evo,reinit)

  reinit!(ls_evo,φh_nondesign)

  ## Hilbertian extension-regularisation problems
  _α(hₕ) = (α_coeff*hₕ)^2
  a_hilb(p,q) =∫((_α ∘ hₕ)*∇(p)⋅∇(q) + p*q)dΩ_act;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg;ls=CGAMGSolver())

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
    γ=γ_evo,verbose=i_am_main(ranks),constraint_names=[:Vol],converged,has_oscillations)
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
    write_history(path*"/history.txt",optimiser.history;ranks)

    # Geometric operation to re-add the non-designable region # TODO: Move to a function
    _φ = get_free_dof_values(φh)
    _φ_nondesign = get_free_dof_values(φh_nondesign)
    map(local_views(_φ),local_views(_φ_nondesign)) do φ,φ_nondesign
      φ .= min.(φ,φ_nondesign)
    end
    consistent!(_φ) |> wait
    reinit!(ls_evo,φh)
    return "First Iteration done!"
  end
  it = get_history(optimiser).niter; uh,ph,dh = get_state(pcf)
  writevtk(Ω_act,path*"Omega_act_$it",
    cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ph"=>ph,"dh"=>dh])
  writevtk(Ω.Ωf,path*"Omega_f_$it",
    cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
  writevtk(Ω.Ωs,path*"Omega_s_$it",
    cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
end

with_mpi() do distribute
  mesh_partition = (2,2,2)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  petsc_options = "-ksp_converged_reason -ksp_error_if_not_converged true"
  GridapPETSc.with(;args=split(petsc_options)) do
    main(ranks,mesh_partition)
  end
end