using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapDistributed
using PartitionedArrays
using GridapPETSc
using GridapTopOpt

using GridapTopOpt: StaggeredStateParamMap

"""
  This example appears in our manuscript:
    "Level-set topology optimisation with unfitted finite elements and automatic shape differentiation"
    by Z.J. Wegert, J. Manyer, C. Mallon, S. Badia, V.J. Challis. (10.48550/arXiv.2504.09748)

  (MPI) Three-dimensional minimum elastic compliance of an elastic part in an FSI problem
    using a CutFEM formulation based on Burman et al. (2018) [10.1016/j.cma.2017.09.005]
    and Burman et al. (2015) [10.1002/nme.4823]. For the fluid part, we use a P1/P0dc
    formulation.

  Optimisation problem:
      Min J(Ωₛ) = ∫ ε(d) ⊙ σ(ε(d)) dΩ
       Ωₛ
    s.t., Vol(Ωₛ) = 0.06,
          R₁((u,p),(v,q)) = 0, ⟶ ⎡(u,p)∈U×Q [=H¹(Ωf;u(Γ_D)=uᵢₙ)×L²(Ωf)],
                                 ⎣a_f(u,v) + b_f(v,p) + b_f(u,q) + j_fu(u,v) + j_fp(p,q) + i_f(p,q) = 0, ∀(v,q)∈V×Q,
          R₂((u,p),d.s) = 0.   ⟶ ⎡d∈V=H¹(Ωₛ;u(Γ_D)=0),
                                 ⎣∫ ε(s) ⊙ σ ∘ ε(d) dΩₛ + j(d,s) + i(d,s) - ∫ n ⋅ σf(u,p)) ⋅ s dΓ = 0, ∀s∈V.

  - For R₁ above, a_f(u,v) is the velocity bilinear form, b_f(v,p) and b_f(u,q) are the
    velocity-pressure coupling terms, j_fu(u,v) is the velocity ghost penalty term over
    the ghost skeleton Γg with outward normal n_Γg, and j_fp(p,q) is the symmetric pressure
    penalty over the skeleton Γi. In addition, i_f(p,q) enforces zero pressure within the isolated volumes
    marked by ψ_f. These are given by
        a_f(u,v) = ∫ μf(∇u ⊙ ∇v) dΩf - ∫ μ(n ⋅ ∇u) ⋅ v + μ(n ⋅ ∇v) ⋅ u - (γ_N/h)u ⋅ v dΓ,
        b_f(v,p) = -∫ p∇ ⋅ v dΩf - ∫ pn⋅v dΓ,
        j_fu(∇u,∇v) = ∫ (γ_u*μ_f*h)[[∇(d)⋅n_Γg]]⋅[[(∇(s)⋅n_Γg]] dΓg,
        j_fp(p,q) = ∫ (h*γ_p/μ_f)h[[∇(d)⋅n_Γg]]⋅[[(∇(s)⋅n_Γg]] dΓi,
        i_f(p,q) = ∫ ψ_f(pq) dΩf.

  - For R₂ above, j(d,s) is the displacement ghost penalty term over the ghost skeleton Γg
    with outward normal n_Γg, and i(d,s) enforces zero displacement within the
    isolated volumes marked by ψ_s. These are given by
        j(d,s) = ∫ γh³[[∇(d)⋅n_Γg]]⋅[[(∇(s)⋅n_Γg]] dΓg, &
        i(d,s) = ∫ ψ_s(d ⋅ s) dΩₛ.
"""
function main(ranks)
  # Params
  vf = 0.06
  γ_evo =  0.1
  max_steps = 15
  α_coeff = γ_evo*max_steps
  iter_mod = 50
  D = 3
  mesh_name = "FSI_3d_symmetric.msh"
  mesh_file = (@__DIR__)*"/Meshes/$mesh_name"

  # Output path
  path = "./results/Unfitted_FSI_3d/"
  files_path = path*"data/"
  model_path = path*"model/"
  if i_am_main(ranks)
    mkpath(files_path); mkpath(model_path);
  end

  # Load gmsh mesh (Currently need to update mesh.geo and these values concurrently)
  L = 4.0;
  H = 1.0;
  x0 = 2;
  l = 1.0;
  w = 0.1;
  a = 0.7;
  b = 0.1;
  cw = 0.1;

  model = GmshDiscreteModel(ranks,mesh_file)
  model = UnstructuredDiscreteModel(model)
  writevtk(model,model_path*"model")

  Ω_act = Triangulation(model)
  hₕ = get_element_diameter_field(model)
  hmin = minimum(get_element_diameters(model))

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Omega_NonDesign","Gamma_Symm_NonDesign","Gamma_Bottom"])
  U_reg = TrialFESpace(V_reg)

  _e = 1/3*hmin
  f0((x,y,z),a,b) = (max(2/a*abs(x-x0),1/(b/2+1)*abs(y-b/2+1),2/(H-2cw)*abs(z-H/2))-1)*min(a,b,cw,H)
  f1((x,y,z),q,r) = - cos(q*π*x)*cos(q*π*y)*cos(q*π*z)/q - r/q
  fin(x) = f0(x,l*(1.1+_e),a*(1.1+_e))
  fsolid(x) = min(f0(x,l*(1.1+_e),b*(1+_e)),f0(x,w*(1+_e),a*(1.1+_e)))
  fholes((x,y,z),q,r) = max(f1((x,y,z),q,r),f1((x-1/q,y,z),q,r))
  lsf(x) = min(max(fin(x),fholes(x,5,0.5)),fsolid(x))
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
    ψ_s,_ = get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Gamma_Bottom"])
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
  uin(x) = VectorValue(x[2],0.0,0.0)
  uin_dot_e1(x) = uin(x)⋅VectorValue(1.0,0.0,0.0)

  reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1,space=:P)
  reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

  function build_spaces(Ω_act_s,Ω_act_f)
    # Test spaces
    V = TestFESpace(Ω_act_f,reffe_u,conformity=:H1,
      dirichlet_tags=["Gamma_f_D","Gamma_Bottom","Gamma_Top",
        "Gamma_Symm","Gamma_Symm_NonDesign","Gamma_Right","Gamma_TopCorners"],
      dirichlet_masks=[(true,true,true),(true,true,true),(false,true,false),
        (false,false,true),(false,false,true),(false,false,true),(false,true,true)])
    Q = TestFESpace(Ω_act_f,reffe_p,conformity=:L2)
    T = TestFESpace(Ω_act_s,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_Bottom","Gamma_Symm","Gamma_Symm_NonDesign"],
      dirichlet_masks=[(true,true,true),(false,false,true),(false,false,true)])

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
  γ_Nu_h = γ_Nu ∘ hₕ
  γ_u_h = mean(γ_u ∘ hₕ)
  γ_p_h = mean(γ_p ∘ hₕ)

  # Terms
  _I = one(SymTensorValue{3,Float64})
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
  fluid_ls = PETScLinearSolver()
  elast_ls = PETScLinearSolver()

  state_collection = EmbeddedCollection_in_φh(model,φh) do _φh
    update_collection!(Ω,_φh)
    (UP,VQ),(R,T) = build_spaces(Ω.Ω_act_s,Ω.Ω_act_f)
    solver = StaggeredFESolver([fluid_ls,elast_ls]);
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
  evolve_ls = PETScLinearSolver()
  evolve_nls = NewtonSolver(evolve_ls;maxiter=1,verbose=i_am_main(ranks))
  reinit_nls = NewtonSolver(PETScLinearSolver();maxiter=20,rtol=1.e-14,verbose=i_am_main(ranks))

  evo = CutFEMEvolve(V_φ,Ω,dΩ_act,hₕ;max_steps,γg=0.01,ode_ls=evolve_ls,ode_nl=evolve_nls)
  reinit = StabilisedReinit(V_φ,Ω,dΩ_act,hₕ;stabilisation_method=ArtificialViscosity(0.5),nls=reinit_nls)
  ls_evo = UnfittedFEEvolution(evo,reinit)

  ## Hilbertian extension-regularisation problems
  _α(hₕ) = (α_coeff*hₕ)^2
  a_hilb(p,q) =∫((_α ∘ hₕ)*∇(p)⋅∇(q) + p*q)dΩ_act;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg;ls=CGAMGSolver())

  ## Optimiser
  converged(m) = GridapTopOpt.default_al_converged(
    m;
    L_tol = 0.075hmin,
    C_tol = 0.05vf
  )
  optimiser = AugmentedLagrangian(pcf,ls_evo,vel_ext,φh;
    γ=γ_evo,verbose=i_am_main(ranks),constraint_names=[:Vol],converged)
  for (it,(uh,ph,dh),φh) in optimiser
    if iszero(it % iter_mod)
      writevtk(Ω_act,files_path*"Omega_act_$it",
        cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ph"=>ph,"dh"=>dh,
          "ψ_s"=>Ω.ψ_s,"ψ_f"=>Ω.ψ_f])
      writevtk(Ω.Ωf,files_path*"Omega_f_$it",
        cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
      writevtk(Ω.Ωs,files_path*"Omega_s_$it",
        cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
    end
    psave(files_path*"LSF_$it",get_free_dof_values(φh))
    write_history(path*"/history.txt",optimiser.history;ranks)

    isolated_vol = sum(iso_vol_frac(φh))
    i_am_main(ranks) && println(" --- Isolated volume: ",isolated_vol)
  end
  it = get_history(optimiser).niter; uh,ph,dh = get_state(pcf)
  writevtk(Ω_act,path*"Omega_act_$it",
    cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ph"=>ph,"dh"=>dh])
  writevtk(Ω.Ωf,path*"Omega_f_$it",
    cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
  writevtk(Ω.Ωs,path*"Omega_s_$it",
    cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
  psave(path*"LSF_$it",get_free_dof_values(φh))
  nothing
end

## CG-AMG solver
CGAMGSolver(;kwargs...) = PETScLinearSolver(gamg_ksp_setup(;kwargs...))

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

## Run
with_mpi() do distribute
  ncpus = 48
  ranks = distribute(LinearIndices((ncpus,)))
  petsc_options = "-ksp_converged_reason -ksp_error_if_not_converged true -pc_type lu -pc_factor_mat_solver_type superlu_dist"
  GridapPETSc.with(;args=split(petsc_options)) do
    main(ranks)
  end
end
