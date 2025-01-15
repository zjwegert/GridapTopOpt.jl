using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt

using GridapDistributed,PartitionedArrays,GridapPETSc

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

function main(ranks)
  path = "./results/Staggered_FCM_3d_FSI_NavierStokes_GMSH/"
  i_am_main(ranks) && mkpath(path)

  γ_evo = 0.2
  max_steps = 24 # TODO: check this
  vf = 0.025
  α_coeff = γ_evo*max_steps
  iter_mod = 1
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
  vol_D = 2.0*0.5

  model = GmshDiscreteModel(ranks,(@__DIR__)*"/fsi/gmsh/mesh_3d.msh")
  # test_mesh(model) # Only in serial for now
  map(test_mesh,local_views(model))
  writevtk(model,path*"model")
  # return model

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
      :χ_s     => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_s_D"];IN_is=IN),
      :χ_f     => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_f_D"];IN_is=OUT)
    )
  end
  writevtk(get_triangulation(φh),path*"initial_islands",cellfields=["χ_s"=>Ω.χ_s,"χ_f"=>Ω.χ_f])
  writevtk(Ω.Ωs,path*"Omega_s_initial")

  # Setup spaces
  uin(x) = VectorValue(x[2],0.0,0.0)

  reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
  reffe_p = ReferenceFE(lagrangian,Float64,order,space=:P)
  reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

  # Test spaces
  V = TestFESpace(Ω_act,reffe_u,conformity=:H1,
    dirichlet_tags=["Gamma_f_D","Gamma_s_D","Gamma_Bottom","Gamma_Top",
      "Gamma_Left","Gamma_Right","Gamma_TopCorners"],
    dirichlet_masks=[(true,true,true),(true,true,true),(true,true,true),
      (false,true,false),(false,false,true),(false,false,true),(false,true,true)])
  Q = TestFESpace(Ω_act,reffe_p,conformity=:H1)
  T = TestFESpace(Ω_act,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_s_D"])

  # Trial spaces
  U = TrialFESpace(V,[uin,[VectorValue(0.0,0.0,0.0) for _ = 1:6]...])
  P = TrialFESpace(Q)
  R = TrialFESpace(T)

  # Multifield spaces
  mfs = BlockMultiFieldStyle()
  UP = MultiFieldFESpace([U,P])#;style=mfs)
  VQ = MultiFieldFESpace([V,Q])#;style=mfs)
  # mfs = BlockMultiFieldStyle(2,(2,1))
  # X = MultiFieldFESpace([U,P,R];style=mfs)
  # Y = MultiFieldFESpace([V,Q,T];style=mfs)

  ### Weak form

  ## Fluid
  # Properties
  Re = 60 # Reynolds number
  ρ = 1.0 # Density
  cl = a # Characteristic length
  u0_max = 1.0
  μ = ρ*cl*u0_max/Re # Viscosity
  ν = μ/ρ # Kinematic viscosity

  # Stabilization parameters
  α_Nu    = 2.5#5.0#2.5
  α_PSPG = 1/3

  γ_Nu(h)    = α_Nu*μ/0.0001^2
  # τ_PSPG(h) = α_PSPG*(h^2/4ν) # (Eqn. 32, Peterson et al., 2018)
  τ_PSPG(h,u) = α_PSPG*((2norm(u)/h)^2 + 9*(4ν/h^2)^2)^-0.5 # (Eqn. 31, Peterson et al., 2018)

  # Terms
  σf_n(u,p,n) = μ*∇(u) ⋅ n - p*n
  a_Ω(u,v) = μ*(∇(u) ⊙ ∇(v)) # (Eqn. 3.3, Massing et al., 2014)
  b_Ω(v,p) = - (∇ ⋅ v)*p # (Eqn. 3.4, Massing et al., 2014)
  c_Ω(p,q,u) = (τ_PSPG ∘ (hₕ,u))*1/ρ*(∇(p) ⋅ ∇(q)) # (Eqn. 3.7, Massing et al., 2014)

  a_fluid((u,p),(v,q),φ) =
    ∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p) )Ω.dΩf + # Volume terms
    ∫( a_Ω(u,v)+b_Ω(u,q)+b_Ω(v,p)+(γ_Nu ∘ hₕ)*(u⋅v) )Ω.dΩs # Stabilization terms

  a_PSPG((u,p),(v,q),φ) = ∫( -c_Ω(p,q,u) )Ω.dΩf + ∫( -c_Ω(p,q,u) )Ω.dΩs
  jac_PSPG((u,p),(du,dp),(v,q),φ) = ∫( -c_Ω(dp,q,u) )Ω.dΩf + ∫( -c_Ω(dp,q,u) )Ω.dΩs # Shouldn't diff through u in τ_PSPG

  conv(u,∇u) = ρ*(∇u') ⋅ u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
  c(u,v,φ) = ∫( v ⋅ (conv∘(u,∇(u))) )Ω.dΩf + ∫( v ⋅ (conv∘(u,∇(u))) )Ω.dΩs
  dc(u,du,v,φ) = ∫( v ⋅ (dconv∘(du,∇(du),u,∇(u))) )Ω.dΩf + ∫( v ⋅ (dconv∘(du,∇(du),u,∇(u))) )Ω.dΩs

  res_fluid((),(u,p),(v,q),φ) = a_fluid((u,p),(v,q),φ) + a_PSPG((u,p),(v,q),φ) + c(u,v,φ)
  jac_fluid((),(u,p),(du,dp),(v,q),φ) = a_fluid((du,dp),(v,q),φ) + jac_PSPG((u,p),(du,dp),(v,q),φ) + dc(u,du,v,φ)

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
  function l_solid(((u,p),),s,φ)
    n = get_normal_vector(Ω.Γ)
    return ∫(σf_n(u,p,n) ⋅ s)Ω.dΓ
  end

  res_solid(((u,p),),d,s,φ) = a_solid(((u,p),),d,s,φ) - l_solid(((u,p),),s,φ)
  jac_solid(((u,p),),d,dd,s,φ) = a_solid(((u,p),),dd,s,φ)

  # ## Optimisation functionals
  # J_comp(((u,p),d),φ) = ∫(ε(d) ⊙ (σ ∘ ε(d)))Ω.dΩs
  # Vol(((u,p),d),φ) = ∫(vol_D)Ω.dΩs - ∫(vf/vol_D)dΩ_act

  ## Staggered operators

  _res_fluid((),(u,p),(v,q)) = res_fluid((),(u,p),(v,q),φh)
  _jac_fluid((),(u,p),(du,dp),(v,q)) = jac_fluid((),(u,p),(du,dp),(v,q),φh)
  _res_solid(((u,p),),d,s) = res_solid(((u,p),),d,s,φh)
  _jac_solid(((u,p),),d,dd,s) = jac_solid(((u,p),),d,dd,s,φh)
  op = StaggeredNonlinearFEOperator([_res_fluid,_res_solid],[_jac_fluid,_jac_solid],[UP,R],[VQ,T])#X,Y);

  ## Solvers
  fluid_ls = MUMPSSolver()

  # Elasticity
  elast_ls = ElasticitySolver(R;rtol=1.e-8,maxits=200)

  fluid_nls = NewtonSolver(fluid_ls;maxiter=10,rtol=1.e-8,verbose=i_am_main(ranks))
  elast_nls = NewtonSolver(elast_ls;maxiter=1,verbose=i_am_main(ranks))

  solver = StaggeredFESolver([fluid_nls,elast_nls]);
  xh = zero(op.trial);
  solve!(xh,solver,op);

  uh,ph,dh = xh
  writevtk(Ω.Ωf,path*"Omega_f_res",cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
  writevtk(Ω.Ωs,path*"Omega_s_res",cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
end

with_mpi() do distribute
  mesh_partition = (2,2,2)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  petsc_options = "-ksp_monitor -ksp_error_if_not_converged true"
  GridapPETSc.with(;args=split(petsc_options)) do
    main(ranks)
  end
end
