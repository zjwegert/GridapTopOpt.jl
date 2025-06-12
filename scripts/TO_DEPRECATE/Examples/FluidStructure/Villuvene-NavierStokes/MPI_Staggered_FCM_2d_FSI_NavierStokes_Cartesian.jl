using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt

using GridapDistributed,PartitionedArrays,GridapPETSc

using LinearAlgebra
LinearAlgebra.norm(x::VectorValue,p::Real) = norm(x.data,p)

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

GMRESAMGSolver(;rtol=10^-8,maxits=100) = PETScLinearSolver(gmres_amg_ksp_setup(rtol,maxits))

function gmres_amg_ksp_setup(rtol,maxits)

  function ksp_setup(ksp)
    pc = Ref{GridapPETSc.PETSC.PC}()

    @check_error_code GridapPETSc.PETSC.KSPSetFromOptions(ksp[])

    rtol = PetscScalar(rtol)
    atol = GridapPETSc.PETSC.PETSC_DEFAULT
    dtol = GridapPETSc.PETSC.PETSC_DEFAULT
    maxits = PetscInt(maxits)

    @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPGMRES)
    @check_error_code GridapPETSc.PETSC.KSPSetTolerances(ksp[], rtol, atol, dtol, maxits)
    @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
    @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCGAMG)
    @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
  end

  return ksp_setup
end

if isassigned(ARGS,1)
  global γg_evo =  parse(Float64,ARGS[1])
else
  global γg_evo =  0.1
end

function main(n,ranks,mesh_partition)
  path = "./results/MPI_Staggered_FCM_2d_FSI_NavierStokes_Cartesian/"
  files_path = path*"data/"
  i_am_main(ranks) && mkpath(files_path)

  ## Geo info
  L = 2.0;
  H = 0.5;
  x0 = 0.5;
  l = 0.4;
  w = 0.025;
  a = 0.3;
  b = 0.01;
  vol_D = 2.0*0.5

  ## Params
  γ_evo = 0.2
  max_steps = floor(Int,min(L*n,H*n)/10)
  vf = 0.025
  α_coeff = γ_evo*max_steps
  iter_mod = 1
  D = 2

  _model = CartesianDiscreteModel(ranks,mesh_partition,(0,L,0,H),(floor(Int,L*n),floor(Int,H*n)))
  base_model = UnstructuredDiscreteModel(_model)
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)

  f_Γ_Top(x) = x[2] == H
  f_Γ_Bottom(x) = x[2] == 0.0
  f_Γ_D(x) = x[1] == 0.0
  f_Γ_N(x) = x[1] == L
  f_NonDesign(x) = ((x0 - w/2 - eps() <= x[1] <= x0 + w/2 + eps() && 0.0 <= x[2] <= a + eps()) ||
    (x0 - l/2 - eps() <= x[1] <= x0 + l/2 + eps() && 0.0 <= x[2] <= b + eps()))

  update_labels!(1,model,f_Γ_Top,"Gamma_Top")
  update_labels!(2,model,f_Γ_Bottom,"Gamma_Bottom")
  update_labels!(3,model,f_Γ_D,"Gamma_f_D")
  update_labels!(4,model,f_Γ_N,"Gamma_f_N")
  update_labels!(5,model,f_NonDesign,"Omega_NonDesign")
  update_labels!(6,model,x->f_NonDesign(x) && f_Γ_Bottom(x),"Gamma_s_D")
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
  f0((x,y),W,H) = max(2/W*abs(x-x0),1/(H/2+1)*abs(y-H/2+1))-1
  f1((x,y),q,r) = - cos(q*π*x)*cos(q*π*y)/q - r/q
  fin(x) = f0(x,l*(1+5_e),a*(1+5_e))
  fsolid(x) = min(f0(x,l*(1+_e),b*(1+_e)),f0(x,w*(1+_e),a*(1+_e)))
  fholes((x,y),q,r) = max(f1((x,y),q,r),f1((x-1/q,y),q,r))
  φf(x) = min(max(fin(x),fholes(x,25,0.2)),fsolid(x))
  # φf(x) = min(max(fin(x),fholes(x,22,0.6)),fsolid(x))
  φh = interpolate(φf,V_φ)
  writevtk(get_triangulation(φh),path*"initial_lsf",cellfields=["φ"=>φh,"h"=>hₕ])

  φh_nondesign = interpolate(fsolid,V_φ)

  # Setup integration meshes and measures
  order = 1
  degree = 2*(order+1)

  dΩ_act = Measure(Ω_act,degree)
  Γf_D = BoundaryTriangulation(model,tags="Gamma_f_D")
  Γf_N = BoundaryTriangulation(model,tags="Gamma_f_N")
  dΓf_D = Measure(Γf_D,degree)
  dΓf_N = Measure(Γf_N,degree)
  Ω = EmbeddedCollection(model,φh) do cutgeo,_,_
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
      :n_Γ        => get_normal_vector(Γ), # Note, need to recompute inside obj/constraints to compute derivs
    )
  end

  # Setup spaces
  uin(x) = VectorValue(16x[2]*(H-x[2]),0.0)
  uin_dot_e1(x) = uin(x)⋅VectorValue(1.0,0.0)

  reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
  reffe_p = ReferenceFE(lagrangian,Float64,order,space=:P)
  reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

  # Test spaces
  V = TestFESpace(Ω_act,reffe_u,conformity=:H1,
    dirichlet_tags=["Gamma_f_D","Gamma_Top","Gamma_Bottom","Gamma_s_D"])
  Q = TestFESpace(Ω_act,reffe_p,conformity=:H1)
  T = TestFESpace(Ω_act,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_s_D"])

  # Trial spaces
  U = TrialFESpace(V,[uin,VectorValue(0.0,0.0),VectorValue(0.0,0.0),VectorValue(0.0,0.0)])
  P = TrialFESpace(Q)
  R = TrialFESpace(T)

  # Multifield spaces
  UP = MultiFieldFESpace([U,P])
  VQ = MultiFieldFESpace([V,Q])

  ### Weak form

  ## Fluid
  NS = 1 # Turn convection on and off
  SUPG = 1 # Turn SUPG on and off

  # Properties
  Re = 50 # Reynolds number
  ρ = 1.0 # Density
  cl = a # Characteristic length
  u0_max = sum(∫(uin_dot_e1)dΓf_D)/sum(∫(1)dΓf_D)
  μ = ρ*cl*u0_max/Re # Viscosity
  ν = μ/ρ # Kinematic viscosity

  # Stabilization parameters
  α_Nu   = 2.5
  α_SUPG = 1/3

  γ_Nu         = α_Nu*(μ/0.001^2)
  τ_SUPG(h,u)  = α_SUPG*(SUPG*(2norm(u)/h)^2 + 9*(4ν/h^2)^2)^-0.5 # (Eqn. 31, Peterson et al., 2018)
  τ_PSPG(h,u)  = τ_SUPG(h,u) # (Sec. 3.2.2, Peterson et al., 2018)

  # Terms
  δ = one(SymTensorValue{D,Float64})
  σ_f(ε,p) = -p*δ + 2μ*ε
  σ_f_β(ε,p) = -βp*p*δ + βμ*2μ*ε

  conv(u,∇u) = (∇u') ⋅ u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

  r_conv(u,v) = NS*ρ*v ⋅ (conv∘(u,∇(u)))
  r_Ωf((u,p),(v,q)) = ε(v) ⊙ (σ_f ∘ (ε(u),p)) + q*(∇⋅u)

  # Additional Brinkmann terms in SUPG based on 10.1002/nme.3151
  r_SUPG((u,p),(v,q),w;IN_Ωf=1) = (NS*SUPG*IN_Ωf*(τ_SUPG ∘ (hₕ,w))*(conv ∘ (u,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅
  (NS*SUPG*ρ*(conv∘(u,∇(u))) + ∇(p) - μ*Δ(u) + (1-IN_Ωf)*γ_Nu*u)
  r_SUPG_picard((u,p),(v,q),w;IN_Ωf=1) = (NS*SUPG*IN_Ωf*(τ_SUPG ∘ (hₕ,w))*(conv ∘ (w,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅
  (NS*SUPG*ρ*(conv∘(w,∇(u))) + ∇(p) - μ*Δ(u) + (1-IN_Ωf)*γ_Nu*u)

  dr_conv(u,du,v) = NS*ρ*v ⋅ (dconv∘(du,∇(du),u,∇(u)))
  dr_SUPG((u,p),(du,dp),(v,q),w;IN_Ωf=1) =
  (NS*SUPG*IN_Ωf*(τ_SUPG ∘ (hₕ,w))*(conv ∘ (du,∇(v))))⋅(NS*SUPG*ρ*(conv∘(u,∇(u))) + ∇(p) - μ*Δ(u) + (1-IN_Ωf)*γ_Nu*u) +
  (NS*SUPG*IN_Ωf*(τ_SUPG ∘ (hₕ,w))*(conv ∘ (u,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅(NS*SUPG*ρ*(dconv∘(du,∇(du),u,∇(u))) + ∇(dp) - μ*Δ(du) + (1-IN_Ωf)*γ_Nu*du)

  function res_fluid((),(u,p),(v,q),φ)
  return ∫(r_conv(u,v) + r_Ωf((u,p),(v,q)) + r_SUPG((u,p),(v,q),u))Ω.dΩf +
    ∫(r_conv(u,v) + r_Ωf((u,p),(v,q)) + γ_Nu*(u⋅v) + r_SUPG((u,p),(v,q),u;IN_Ωf=0))Ω.dΩs
  end

  function jac_fluid_newton((),(u,p),(du,dp),(v,q),φ)
  return ∫(dr_conv(u,du,v) + r_Ωf((du,dp),(v,q)) + dr_SUPG((u,p),(du,dp),(v,q),u))Ω.dΩf +
    ∫(dr_conv(u,du,v) + r_Ωf((du,dp),(v,q)) + γ_Nu*(du⋅v) + dr_SUPG((u,p),(du,dp),(v,q),u;IN_Ωf=0))Ω.dΩs
  end

  function jac_fluid_picard((),(u,p),(du,dp),(v,q),φ)
  return ∫(NS*ρ*v ⋅ (conv∘(u,∇(du))) + r_Ωf((du,dp),(v,q)) + r_SUPG_picard((du,dp),(v,q),u))Ω.dΩs +
    ∫(NS*ρ*v ⋅ (conv∘(u,∇(du))) + r_Ωf((du,dp),(v,q)) + γ_Nu*(du⋅v) + r_SUPG_picard((du,dp),(v,q),u;IN_Ωf=0))Ω.dΩf
  end

  jac_fluid_AD((),x,dx,y,φ) = jacobian((_x,_y,_φ)->res_fluid((),_x,_y,_φ),[x,y,φ],1)

  ## Structure
  # Material parameters
  function lame_parameters(E,ν)
    λ = (E*ν)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    (λ, μ)
  end
  λs, μs = lame_parameters(0.1,0.05)
  # Ersatz parameter
  ϵ = (λs + 2μs)*1e-3
  # Terms
  σ(ε) = λs*tr(ε)*one(ε) + 2*μs*ε
  a_s_Ω(s,d) = ε(s) ⊙ (σ ∘ ε(d))

  function a_solid(((u,p),),d,s,φ)
    return ∫(a_s_Ω(s,d))Ω.dΩs + ∫(ϵ*a_s_Ω(s,d))Ω.dΩf
  end
  _n(∇φ) = ∇φ/(10^-20+norm(∇φ))
  function l_solid(((u,p),),s,φ)
    # n = get_normal_vector(Ω.Γ)  # TODO: This is currently broken in distributed when using AD
    n = _n ∘ ∇(φ)
    return ∫(s ⋅ (σ_f(ε(u),p) ⋅ n))Ω.dΓ
  end

  res_solid(((u,p),),d,s,φ) = a_solid(((u,p),),d,s,φ) - l_solid(((u,p),),s,φ)
  jac_solid(((u,p),),d,dd,s,φ) = a_solid(((u,p),),dd,s,φ)

  ## Optimisation functionals
  J_comp(((u,p),d),φ) = ∫(ε(d) ⊙ (σ ∘ ε(d)))Ω.dΩs
  Vol(((u,p),d),φ) = ∫(vol_D)Ω.dΩs - ∫(vf/vol_D)dΩ_act
  dVol(q,(u,p,d),φ) = ∫(-1/vol_D*q/(abs(Ω.n_Γ ⋅ ∇(φ))))Ω.dΓ

  ## Staggered operators
  fluid_nls = NewtonSolver(MUMPSSolver();maxiter=10,rtol=1.e-8,verbose=i_am_main(ranks))
  elast_nls = NewtonSolver(ElasticitySolver(R;rtol=1.e-8,maxits=200);maxiter=1,verbose=i_am_main(ranks))
  solver = StaggeredFESolver([fluid_nls,elast_nls]);

  op = StaggeredNonlinearFEOperator([res_fluid,res_solid],[jac_fluid_AD,jac_solid],[UP,R],[VQ,T])
  state_map = StaggeredNonlinearFEStateMap(op,V_φ,U_reg,φh;adjoint_jacobians=[jac_fluid_AD,jac_solid],solver,adjoint_solver=solver)
  pcfs = PDEConstrainedFunctionals(J_comp,[Vol],state_map;analytic_dC=[dVol])

  ## Evolution Method
  evolve_ls = MUMPSSolver()
  evolve_nls = NewtonSolver(evolve_ls;maxiter=1,verbose=i_am_main(ranks))
  reinit_nls = NewtonSolver(MUMPSSolver();maxiter=20,rtol=1.e-14,verbose=i_am_main(ranks))

  evo = CutFEMEvolve(V_φ,Ω,dΩ_act,hₕ;max_steps,γg=γg_evo,ode_ls=evolve_ls,ode_nl=evolve_nls)
  reinit1 = StabilisedReinit(V_φ,Ω,dΩ_act,hₕ;stabilisation_method=ArtificialViscosity(1.0),nls=reinit_nls)
  reinit2 = StabilisedReinit(V_φ,Ω,dΩ_act,hₕ;stabilisation_method=GridapTopOpt.InteriorPenalty(V_φ,γg=1.0),nls=reinit_nls)
  reinit = GridapTopOpt.MultiStageStabilisedReinit([reinit1,reinit2])
  ls_evo = UnfittedFEEvolution(evo,reinit)

  reinit!(ls_evo,φh_nondesign)

  ## Hilbertian extension-regularisation problems
  _α(hₕ) = (α_coeff*hₕ)^2
  a_hilb(p,q) =∫((_α ∘ hₕ)*∇(p)⋅∇(q) + p*q)dΩ_act;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg;solver=CGAMGSolver())

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
  optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
    γ=γ_evo,verbose=i_am_main(ranks),constraint_names=[:Vol],converged,has_oscillations)
  for (it,(uh,ph,dh),φh) in optimiser
    GC.gc()
    if iszero(it % iter_mod)
      writevtk(Ω_act,files_path*"Omega_act_$it",
        cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ph"=>ph,"dh"=>dh])
      writevtk(Ω.Ωf,files_path*"Omega_f_$it",
        cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
      writevtk(Ω.Ωs,files_path*"Omega_s_$it",
        cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
    end
    write_history(path*"/history.txt",optimiser.history;ranks)

    _φ = get_free_dof_values(φh)
    _φ_nondesign = get_free_dof_values(φh_nondesign)
    map(local_views(_φ),local_views(_φ_nondesign)) do φ,φ_nondesign
      φ .= min.(φ,φ_nondesign)
    end
    reinit!(ls_evo,φh)
  end
  it = get_history(optimiser).niter; uh,ph,dh = get_state(pcfs)
  for _dir in (path,files_path)
    writevtk(Ω_act,_dir*"Omega_act_$it",
      cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ph"=>ph,"dh"=>dh])
    writevtk(Ω.Ωf,_dir*"Omega_f_$it",
      cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
    writevtk(Ω.Ωs,_dir*"Omega_s_$it",
      cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
  end
end

with_mpi() do distribute
  n = 200
  mesh_partition = (2,2)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  petsc_options = "-ksp_converged_reason -ksp_error_if_not_converged true -ksp_gmres_modifiedgramschmidt"
  GridapPETSc.with(;args=split(petsc_options)) do
    main(n,ranks,mesh_partition)
  end
end