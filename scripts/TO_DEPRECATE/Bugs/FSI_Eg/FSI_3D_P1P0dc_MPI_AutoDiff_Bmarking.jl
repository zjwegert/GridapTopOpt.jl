using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using Gridap.FESpaces
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt

using GridapDistributed,PartitionedArrays,GridapPETSc

using Gridap.Arrays, Gridap.Helpers

if isassigned(ARGS,1)
  global optim_wf =  parse(Bool,ARGS[1])
else
  global optim_wf =  false
end

function main(ranks,mesh_name;vtk_output=false,debug=false)
  if vtk_output
    path = "./results/FSI_3D_Burman_P1P0dc_MPI_bmark/"
    i_am_main(ranks) && mkpath(path)
  end
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

  model = GmshDiscreteModel(ranks,(@__DIR__)*"/$mesh_name")#mesh_low_res_3d.msh")
  model = UnstructuredDiscreteModel(model)

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
  lsf(x) = fin(x)
  φh = interpolate(lsf,V_φ)
  GridapTopOpt.correct_ls!(φh)
  vtk_output && writevtk(get_triangulation(φh),path*"initial_lsf",cellfields=["φh"=>φh])

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
    ψ_s,_ = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Gamma_s_D"])
    _,ψ_f = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Gamma_f_D"])
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

  # Test spaces
  V = TestFESpace(Ω.Ω_act_f,reffe_u,conformity=:H1,
    dirichlet_tags=["Gamma_f_D","Gamma_s_D","Gamma_Bottom","Gamma_Top",
    "Gamma_Left","Gamma_Right","Gamma_TopCorners"],
    dirichlet_masks=[(true,true,true),(true,true,true),(true,true,true),
      (false,true,false),(false,false,true),(false,false,true),(false,true,true)])
  Q = TestFESpace(Ω.Ω_act_f,reffe_p,conformity=:L2)

  # Trial spaces
  U = TrialFESpace(V,[uin,[VectorValue(0.0,0.0,0.0) for _ = 1:6]...])
  P = TrialFESpace(Q)

  # Multifield spaces
  UP = MultiFieldFESpace([U,P])
  VQ = MultiFieldFESpace([V,Q])

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
  a_Ω(u,v) = μ*(∇(u) ⊙ ∇(v))
  b_Ω(v,p) = -p*(∇⋅v)
  a_Γ(u,v,n) = - μ*(n⋅∇(u)) ⋅ v - μ*(n⋅∇(v)) ⋅ u + (γ_Nu ∘ hₕ)*(u⋅v)
  b_Γ(v,p,n) = (n⋅v)*p
  ju(u,v) = mean(γ_u ∘ hₕ)*(jump(Ω.n_Γg ⋅ ∇(u)) ⋅ jump(Ω.n_Γg ⋅ ∇(v)))
  jp(p,q) = mean(γ_p ∘ hₕ)*(jump(p) * jump(q))
  v_ψ(p,q) = (k_p * Ω.ψ_f)*(p*q) # (Isolated volume term, Eqn. 15, Villanueva and Maute, 2017)

  function a_fluid((),(u,p),(v,q),φ)
    n_Γ = -get_normal_vector(Ω.Γ)
    return ∫(a_Ω(u,v) + b_Ω(v,p) + b_Ω(u,q) + v_ψ(p,q))Ω.dΩf +
      ∫(a_Γ(u,v,n_Γ) + b_Γ(v,p,n_Γ) + b_Γ(u,q,n_Γ))Ω.dΓ +
      ∫(ju(u,v))Ω.dΓg - ∫(jp(p,q))Ω.dΓi
  end

  l_fluid((),(v,q),φ) =  ∫(0q)Ω.dΩf

  xdh = zero(UP)
  t = PTimer(ranks);

  assem_deriv = SparseMatrixAssembler(U_reg,V_reg)

  function time_calls(t,name::String="")
    tic!(t;barrier=true)
    λᵀ1_∂R1∂φ = ∇(φ -> a_fluid((),xdh,xdh,φ) - l_fluid((),xdh,φ),φh)
    toc!(t,"Compute grad $name")

    tic!(t;barrier=true)
    vecdata = collect_cell_vector(U_reg,λᵀ1_∂R1∂φ)
    toc!(t,"Collect cell vec $name")

    tic!(t;barrier=true)
    Σ_λᵀs_∂Rs∂φ = allocate_vector(assem_deriv,vecdata)
    toc!(t,"Alloc vec $name")
    return Σ_λᵀs_∂Rs∂φ
  end

  function time_calls(name::String="")
    @time "λᵀ1_∂R1∂φ $name" λᵀ1_∂R1∂φ = ∇(φ -> a_fluid((),xdh,xdh,φ) - l_fluid((),xdh,φ),φh)
    @time "Collect cell vec $name" vecdata = collect_cell_vector(U_reg,λᵀ1_∂R1∂φ)
    @time "Σ_λᵀs_∂Rs∂φ $name" Σ_λᵀs_∂Rs∂φ = allocate_vector(assem_deriv,vecdata)
    return Σ_λᵀs_∂Rs∂φ
  end

  if ~debug
    time_calls(t,"(prealloc)")
    time_calls(t)
    display(t)
  else
    time_calls("(prealloc)")
    time_calls()
  end
end

function main_optimised_wf(ranks,mesh_name;vtk_output=false,debug=false)
  if vtk_output
    path = "./results/FSI_3D_Burman_P1P0dc_MPI_bmark_opt/"
    i_am_main(ranks) && mkpath(path)
  end
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

  model = GmshDiscreteModel(ranks,(@__DIR__)*"/$mesh_name")
  model = UnstructuredDiscreteModel(model)

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
  lsf(x) = fin(x)
  φh = interpolate(lsf,V_φ)
  GridapTopOpt.correct_ls!(φh)
  vtk_output && writevtk(get_triangulation(φh),path*"initial_lsf",cellfields=["φh"=>φh])

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
    ψ_s,_ = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Gamma_s_D"])
    _,ψ_f = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Gamma_f_D"])
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

  # Test spaces
  V = TestFESpace(Ω.Ω_act_f,reffe_u,conformity=:H1,
    dirichlet_tags=["Gamma_f_D","Gamma_s_D","Gamma_Bottom","Gamma_Top",
    "Gamma_Left","Gamma_Right","Gamma_TopCorners"],
    dirichlet_masks=[(true,true,true),(true,true,true),(true,true,true),
      (false,true,false),(false,false,true),(false,false,true),(false,true,true)])
  Q = TestFESpace(Ω.Ω_act_f,reffe_p,conformity=:L2)

  # Trial spaces
  U = TrialFESpace(V,[uin,[VectorValue(0.0,0.0,0.0) for _ = 1:6]...])
  P = TrialFESpace(Q)

  # Multifield spaces
  UP = MultiFieldFESpace([U,P])
  VQ = MultiFieldFESpace([V,Q])

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
  a_Ω(∇u,∇v) = μ*(∇u ⊙ ∇v)
  b_Ω(div_v,p) = -p*(div_v)
  a_Γ(u,∇u,v,∇v,n) = - μ*n⋅(∇u ⋅ v + ∇v⋅ u) + γ_Nu_h*(u⋅v)
  b_Γ(v,p,n) = (n⋅v)*p
  ju(∇u,∇v) = γ_u_h*(jump(Ω.n_Γg ⋅ ∇u) ⋅ jump(Ω.n_Γg ⋅ ∇v))
  jp(p,q) = γ_p_h*(jump(p) * jump(q))
  v_ψ(p,q) = (k_p * Ω.ψ_f)*(p*q) # (Isolated volume term, Eqn. 15, Villanueva and Maute, 2017)

  function a_fluid((),(u,p),(v,q),φ)
    ∇u = ∇(u); ∇v = ∇(v);
    div_u = ∇⋅u; div_v = ∇⋅v
    n_Γ = -get_normal_vector(Ω.Γ)
    return ∫(a_Ω(∇u,∇v) + b_Ω(div_v,p) + b_Ω(div_u,q) + v_ψ(p,q))Ω.dΩf +
      ∫(a_Γ(u,∇u,v,∇v,n_Γ) + b_Γ(v,p,n_Γ) + b_Γ(u,q,n_Γ))Ω.dΓ +
      ∫(ju(∇u,∇v))Ω.dΓg - ∫(jp(p,q))Ω.dΓi
  end

  l_fluid((),(v,q),φ) =  ∫(0q)Ω.dΩf

  xdh = zero(UP)
  t = PTimer(ranks);

  assem_deriv = SparseMatrixAssembler(U_reg,V_reg)

  function time_calls(t,name::String="")
    tic!(t;barrier=true)
    λᵀ1_∂R1∂φ = ∇(φ -> a_fluid((),xdh,xdh,φ) - l_fluid((),xdh,φ),φh)
    toc!(t,"Compute grad $name")

    tic!(t;barrier=true)
    vecdata = collect_cell_vector(U_reg,λᵀ1_∂R1∂φ)
    toc!(t,"Collect cell vec $name")

    tic!(t;barrier=true)
    Σ_λᵀs_∂Rs∂φ = allocate_vector(assem_deriv,vecdata)
    toc!(t,"Alloc vec $name")
    return Σ_λᵀs_∂Rs∂φ
  end

  function time_calls(name::String="")
    @time "λᵀ1_∂R1∂φ $name" λᵀ1_∂R1∂φ = ∇(φ -> a_fluid((),xdh,xdh,φ) - l_fluid((),xdh,φ),φh)
    @time "Collect cell vec $name" vecdata = collect_cell_vector(U_reg,λᵀ1_∂R1∂φ)
    @time "Σ_λᵀs_∂Rs∂φ $name" Σ_λᵀs_∂Rs∂φ = allocate_vector(assem_deriv,vecdata)
    return Σ_λᵀs_∂Rs∂φ
  end

  if ~debug
    time_calls(t,"(prealloc)")
    time_calls(t)
    display(t)
  else
    time_calls("(prealloc)")
    time_calls()
  end
end

with_debug() do distribute
  ncpus = 1
  ranks = distribute(LinearIndices((ncpus,)))
  mesh_name = "mesh_ultra_low_res_3d.msh"
  vtk_output=false
  debug=true
  if optim_wf
    main_optimised_wf(ranks,mesh_name;vtk_output,debug)
  else
    main(ranks,mesh_name;vtk_output,debug)
  end
end