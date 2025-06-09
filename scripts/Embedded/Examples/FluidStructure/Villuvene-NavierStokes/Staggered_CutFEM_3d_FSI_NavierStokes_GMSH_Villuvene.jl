using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt

using GridapDistributed,PartitionedArrays,GridapPETSc

using LinearAlgebra
LinearAlgebra.norm(x::VectorValue,p::Real) = norm(x.data,p)

if isassigned(ARGS,1)
  global γg_evo =  parse(Float64,ARGS[1])
else
  global γg_evo =  0.1
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

function main(ranks)
  path = "./results/Staggered_CutFEM_3d_FSI_NavierStokes_GMSH_Villuvene_gammag$γg_evo/"
  i_am_main(ranks) && mkpath(path)

  γ_evo = 0.2
  max_steps = 10 # Based on number of elements in vertical direction divided by 10
  vf = 0.025
  α_coeff = γ_evo*max_steps
  iter_mod = 10
  D = 3

  # Load gmsh mesh (Currently need to update mesh.geo and these values concurrently)
  L = 4.0;
  H = 1.0;
  x0 = 2;
  l = 1.0;
  w = 0.05;
  a = 0.7;
  b = 0.05;
  cw = 0.1;
  vol_D = L*H

  model = GmshDiscreteModel(ranks,(@__DIR__)*"/Meshes/mesh_3d_finer.msh")
  map(test_mesh,local_views(model))
  writevtk(model,path*"model")

  Ω_act = Triangulation(model)
  hₕ = get_element_diameter_field(model)
  hmin = minimum(get_element_diameters(model))

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Omega_NonDesign","Gamma_s_D"])
  U_reg = TrialFESpace(V_reg)

  _e = 5e-3
  f0((x,y,z),a,b) = max(2/a*abs(x-x0),1/(b/2+1)*abs(y-b/2+1),2/(H-2cw)*abs(z-H/2))-1
  f1((x,y,z),q,r) = - cos(q*π*x)*cos(q*π*y)*cos(q*π*z)/q - r/q
  fin(x) = f0(x,l*(1+5_e),a*(1+5_e))
  fsolid(x) = min(f0(x,l*(1+_e),b*(1+_e)),f0(x,w*(1+_e),a*(1+_e)))
  fholes((x,y,z),q,r) = max(f1((x,y,z),q,r),f1((x-1/q,y,z),q,r))
  lsf(x) = min(max(fin(x),fholes(x,5,0.5)),fsolid(x))
  φh = interpolate(lsf,V_φ)
  writevtk(get_triangulation(φh),path*"initial_lsf",cellfields=["φ"=>φh,"h"=>hₕ])

  # φh_nondesign = interpolate(fsolid,V_φ)

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
      :n_Γ        => get_normal_vector(Γ), # Note, need to recompute inside obj/constraints to compute derivs
      :Ω_act_s => Ω_act_s,
      :dΩ_act_s => Measure(Ω_act_s,degree),
      :Ω_act_f => Ω_act_f,
      :dΩ_act_f => Measure(Ω_act_f,degree),
      :ψ_s     => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_s_D"];groups=((GridapTopOpt.CUT,IN),OUT)),
      :ψ_f     => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_f_D"];groups=((GridapTopOpt.CUT,OUT),IN))
    )
  end
  writevtk(get_triangulation(φh),path*"initial_islands",cellfields=["ψ_s"=>Ω.ψ_s,"ψ_f"=>Ω.ψ_f])

  # Setup spaces
  uin(x) = VectorValue(x[2],0.0,0.0)
  uin_dot_e1(x) = uin(x)⋅VectorValue(1.0,0.0,0.0)

  reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
  reffe_p = ReferenceFE(lagrangian,Float64,order,space=:P)
  reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

  function build_spaces(Ω_act_s,Ω_act_f)
    # Test spaces
    V = TestFESpace(Ω_act_f,reffe_u,conformity=:H1,
      dirichlet_tags=["Gamma_f_D","Gamma_s_D","Gamma_Bottom","Gamma_Top",
      "Gamma_Left","Gamma_Right","Gamma_TopCorners"],
      dirichlet_masks=[(true,true,true),(true,true,true),(true,true,true),
        (false,true,false),(false,false,true),(false,false,true),(false,true,true)])
    Q = TestFESpace(Ω_act_f,reffe_p,conformity=:H1)
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
  βp = -1; βμ = 1;

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
    ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (du,∇(v))))⋅(ρ*(conv∘(u,∇(u))) + ∇(p) - μ*Δ(u)) +
    ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (u,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅(ρ*(dconv∘(du,∇(du),u,∇(u))) + ∇(dp) - μ*Δ(du))

  function res_fluid((),(u,p),(v,q),φ)
    n_Γ = -get_normal_vector(Ω.Γ)
    return ∫(r_conv(u,v) + r_Ωf((u,p),(v,q)))Ω.dΩf +
      ∫(r_SUPG((u,p),(v,q),u))Ω.dΩf + #Ω.dΩ_act_f +
      ∫(r_ψ(p,q))Ω.dΩf +
      ∫(r_Γ((u,p),(v,q),n_Γ,u))Ω.dΓ +
      ∫(r_GP_μ(u,v) + r_GP_p(p,q,u) + r_GP_u(u,v,u,Ω.n_Γg) + 0mean(φ))Ω.dΓg
  end

  function jac_fluid_picard((),(u,p),(du,dp),(v,q),φ)
    n_Γ = -get_normal_vector(Ω.Γ)
    return ∫(ρ*v ⋅ (conv∘(u,∇(du))) + r_Ωf((du,dp),(v,q)))Ω.dΩf +
      ∫(r_SUPG_picard((du,dp),(v,q),u))Ω.dΩf + #Ω.dΩ_act_f +
      ∫(r_ψ(dp,q))Ω.dΩf +
      ∫(r_Γ((du,dp),(v,q),n_Γ,u))Ω.dΓ +
      ∫(r_GP_μ(du,v) + r_GP_p(dp,q,u) + r_GP_u(du,v,u,Ω.n_Γg) + 0mean(φ))Ω.dΓg
  end

  function jac_fluid_newton((),(u,p),(du,dp),(v,q),φ)
    n_Γ = -get_normal_vector(Ω.Γ)
    return ∫(dr_conv(u,du,v) + r_Ωf((du,dp),(v,q)))Ω.dΩf +
      ∫(dr_SUPG((u,p),(du,dp),(v,q),u))Ω.dΩf + #Ω.dΩ_act_f +
      ∫(r_ψ(dp,q))Ω.dΩf +
      ∫(r_Γ((du,dp),(v,q),n_Γ,u))Ω.dΓ +
      ∫(r_GP_μ(du,v) + r_GP_p(dp,q,u) + r_GP_u(du,v,u,Ω.n_Γg) + 0mean(φ))Ω.dΓg
  end

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
  # Terms
  σ(ε) = λs*tr(ε)*one(ε) + 2*μs*ε
  a_s_Ω(s,d) = ε(s) ⊙ (σ ∘ ε(d)) # Elasticity
  j_s_k(s,d) = mean(γ_Gd ∘ hₕ)*jump(Ω.n_Γg ⋅ ∇(s)) ⋅ jump(Ω.n_Γg ⋅ ∇(d)) # (Eqn. 3.11, Burman et al., 2018)
  v_s_ψ(s,d) = k_d*Ω.ψ_s*d⋅s # Isolated volume term

  function a_solid(((u,p),),d,s,φ)
    return ∫(a_s_Ω(s,d))Ω.dΩs + ∫(j_s_k(s,d) + 0mean(φ))Ω.dΓg + ∫(v_s_ψ(s,d))Ω.dΩs
  end
  _n(∇φ) = -∇φ/(10^-20+norm(∇φ))
  function l_solid(((u,p),),s,φ)
    # n = -get_normal_vector(Ω.Γ)  # TODO: This is currently broken in distributed when using AD
    n = _n ∘ ∇(φ)
    # return ∫(-s ⋅ ((1-Ω.ψ_s)*σ_f(ε(u),p) ⋅ n))Ω.dΓ
    return ∫(-s ⋅ (σ_f(ε(u),p) ⋅ n))Ω.dΓ
  end

  res_solid(((u,p),),d,s,φ) = a_solid(((u,p),),d,s,φ) - l_solid(((u,p),),s,φ)
  jac_solid(((u,p),),d,dd,s,φ) = a_solid(((u,p),),dd,s,φ)

  ∂R2∂xh1((du,dp),((u,p),),d,s,φ) = -1*l_solid(((du,dp),),s,φ)
  ∂Rk∂xhi = ((∂R2∂xh1,),)

  # ## Optimisation functionals
  J_comp(((u,p),d),φ) = ∫(ε(d) ⊙ (σ ∘ ε(d)))Ω.dΩs
  ∂Jcomp∂up((du,dp),((u,p),d),φ) = ∫(0dp)Ω.dΩs
  # ∂Jcomp∂d(dd,((u,p),d),φ) = ∫(2ε(dd) ⊙ (σ ∘ ε(d)))Ω.dΩs
  ∂Jcomp∂d(dd,((u,p),d),φ) = ∫(ε(dd) ⊙ (σ ∘ ε(d)))Ω.dΩs + ∫(ε(d) ⊙ (σ ∘ ε(dd)))Ω.dΩs
  ∂Jpres∂xhi = (∂Jcomp∂up,∂Jcomp∂d)

  Vol(((u,p),d),φ) = ∫(vol_D)Ω.dΩs - ∫(vf/vol_D)dΩ_act
  dVol(q,(u,p,d),φ) = ∫(-1/vol_D*q/(abs(Ω.n_Γ ⋅ ∇(φ))))Ω.dΓ
  ∂Vol∂up((du,dp),((u,p),d),φ) = ∫(0dp)dΩ_act
  ∂Vol∂d(dd,((u,p),d),φ) = ∫(0dd ⋅ d)dΩ_act
  ∂Vol∂xhi = (∂Vol∂up,∂Vol∂d)

  ## Staggered operators
  fluid_nls = NewtonSolver(MUMPSSolver();maxiter=10,rtol=1.e-8,verbose=i_am_main(ranks))

  state_collection = GridapTopOpt.EmbeddedCollection_in_φh(model,φh) do _φh
    update_collection!(Ω,_φh)
    (UP,VQ),(R,T) = build_spaces(Ω.Ω_act_s,Ω.Ω_act_f)
    elast_nls = NewtonSolver(ElasticitySolver(R;rtol=1.e-8,maxits=200);maxiter=1,verbose=i_am_main(ranks))
    solver = StaggeredFESolver([fluid_nls,elast_nls]);
    op = StaggeredNonlinearFEOperator([res_fluid,res_solid],[jac_fluid_picard,jac_solid],[UP,R],[VQ,T])
    state_map = StaggeredNonlinearFEStateMap(op,∂Rk∂xhi,V_φ,U_reg,_φh;
      adjoint_jacobians=[jac_fluid_newton,jac_solid],solver,adjoint_solver=solver)
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

  # reinit!(ls_evo,φh_nondesign)

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
  optimiser = AugmentedLagrangian(pcf,ls_evo,vel_ext,φh;
    γ=γ_evo,verbose=i_am_main(ranks),constraint_names=[:Vol],converged,has_oscillations)
  try
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

      # # Geometric operation to re-add the non-designable region # TODO: Move to a function
      # _φ = get_free_dof_values(φh)
      # _φ_nondesign = get_free_dof_values(φh_nondesign)
      # map(local_views(_φ),local_views(_φ_nondesign)) do φ,φ_nondesign
      #   φ .= min.(φ,φ_nondesign)
      # end
      # reinit!(ls_evo,φh)
    end
  catch e
    i_am_main(ranks) && println("Error: $e\nPrinting history and exiting...")
    writevtk(Ω_act,path*"Omega_act_errored",
      cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),
        "ψ_s"=>Ω.ψ_s,"ψ_f"=>Ω.ψ_f])
      writevtk(Ω.Ωf,path*"Omega_f_errored")
      writevtk(Ω.Ωs,path*"Omega_s_errored")
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
  ncpus = 96
  ranks = distribute(LinearIndices((ncpus,)))
  petsc_options = "-ksp_converged_reason -ksp_error_if_not_converged true -ksp_gmres_modifiedgramschmidt"
  GridapPETSc.with(;args=split(petsc_options)) do
    main(ranks)
  end
end