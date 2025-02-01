using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt

using PartitionedArrays, GridapDistributed

function create_mesh(fetype,path,mesh_size)
  if fetype==:bodyfitted
    if isfile("$(path)body_fitted_$mesh_size.msh")
      return "$(path)/body_fitted_$mesh_size.msh"
    else
      error("Please create msh file")
      return "$(path)/body_fitted_$mesh_size.msh"
    end
  elseif fetype==:cutfem
    if isfile("$(path)/cutfem_$mesh_size.msh")
      return "$(path)/cutfem_$mesh_size.msh"
    else
      error("Please create msh file")
      return "$(path)/cutfem_$mesh_size.msh"
    end
  else
    error("Invalid `fetype`, should be :bodyfitted or :cutfem")
  end
end

function main_bodyfitted(ranks,params;R=0.1,c=(0.5,0.2))
  Re,mesh_size,α_Nu,α_u,α_p = params

  path = "./results/Stokes-2D-Verification-MPI/"
  data_path = "$path/vtk_data/"
  mkpath(data_path)

  # Create the mesh
  mesh_file =create_mesh(:bodyfitted,path,mesh_size)

  # Load mesh
  model = GmshDiscreteModel(ranks,mesh_file)

  Ω = Triangulation(model)

  # Setup integration meshes and measures
  order = 2
  degree = 2*(order+1)

  dΩ = Measure(Ω,degree)
  Γf_D = BoundaryTriangulation(model,tags="Gamma_f_D")
  Γf_N = BoundaryTriangulation(model,tags="Gamma_f_N")
  dΓf_D = Measure(Γf_D,degree)
  dΓf_N = Measure(Γf_N,degree)
  Γ = BoundaryTriangulation(model,tags="Gamma_fsi")
  dΓ = Measure(Γ,degree)
  n_Γ = get_normal_vector(Γ)

  # Setup spaces
  uin(x) = VectorValue(16x[2]*(1/2-x[2]),0.0)
  uin_dot_e1(x) = uin(x)⋅VectorValue(1.0,0.0)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order,space=:P)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1,space=:P)

  V = TestFESpace(Ω,reffe_u,conformity=:H1,
    dirichlet_tags=["Gamma_f_D","Gamma_NoSlipTop","Gamma_NoSlipBottom","Gamma_fsi"])
  Q = TestFESpace(Ω,reffe_p,conformity=:H1)

  # Trial spaces
  U = TrialFESpace(V,[uin,VectorValue(0.0,0.0),VectorValue(0.0,0.0),VectorValue(0.0,0.0)])
  P = TrialFESpace(Q)

  # Multifield spaces
  X = MultiFieldFESpace([U,P])
  Y = MultiFieldFESpace([V,Q])

  ### Weak form

  ## Fluid
  # Properties
  Re = 60 # Reynolds number
  ρ = 1.0 # Density
  cl = 2R # Characteristic length
  u0_max = sum(∫(uin_dot_e1)dΓf_D)/sum(∫(1)dΓf_D)
  μ = ρ*cl*u0_max/Re # Viscosity
  ν = μ/ρ # Kinematic viscosity

  # Terms
  σf_n(u,p,n) = μ*∇(u) ⋅ n - p*n
  a((u,p),(v,q)) = ∫(μ*(∇(u) ⊙ ∇(v)) - p*(∇⋅v) - q*(∇⋅u))dΩ
  l((v,q)) = 0.0

  op = AffineFEOperator(a,l,X,Y)
  uh,ph = solve(op);

  fname="bodyfitted_Re$(Re)_msh$(round(mesh_size,sigdigits=2))_aNU$(α_Nu)_au$(α_u)_ap$(α_p)"
  writevtk(Ω,data_path*fname,
    cellfields=["uh"=>uh,"ph"=>ph])
  fname="bodyfitted_GAMMA_Re$(Re)_msh$(round(mesh_size,sigdigits=2))_aNU$(α_Nu)_au$(α_u)_ap$(α_p)"
  writevtk(Γ,data_path*fname,
    cellfields=["uh"=>uh,"ph"=>ph,"surface_force"=>σf_n(uh,ph,n_Γ)])
  nothing
end

function main_cutfem(ranks,params;R=0.1,c=(0.5,0.2))
  Re,mesh_size,α_Nu,α_u,α_p = params

  path = "./results/Stokes-2D-Verification-MPI/"
  data_path = "$path/vtk_data/"
  mkpath(data_path)

  # Create the mesh
  mesh_file =create_mesh(:cutfem,path,mesh_size)

  # Load mesh
  model = GmshDiscreteModel(ranks,mesh_file)

  Ω_act = Triangulation(model)
  hₕ = get_element_diameter_field(model)

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)

  x0,y0 = c # Disk centroid
  φh = interpolate(((x,y),)->sqrt((x-x0)^2+(y-y0)^2)-R,V_φ)

  # Setup integration meshes and measures
  order = 1
  degree = 2*(order+1)

  dΩ_act = Measure(Ω_act,degree)
  Γf_D = BoundaryTriangulation(model,tags="Gamma_f_D")
  Γf_N = BoundaryTriangulation(model,tags="Gamma_f_N")
  dΓf_D = Measure(Γf_D,degree)
  dΓf_N = Measure(Γf_N,degree)

  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)
  cutgeo_facets = cut_facets(model,geo)

  Ωs = Triangulation(cutgeo,PHYSICAL)
  Ωf = Triangulation(cutgeo,PHYSICAL_OUT)
  Γ  = EmbeddedBoundary(cutgeo)
  Γg = GhostSkeleton(cutgeo)
  Ω_act_s = Triangulation(cutgeo,ACTIVE)
  Ω_act_f = Triangulation(cutgeo,ACTIVE_OUT)
  Γi = SkeletonTriangulation(cutgeo_facets,ACTIVE_OUT)
  Ω = (;
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
      :n_Γi => get_normal_vector(Γi),
      # :ψ_s => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_s_D"];IN_is=IN),
      # :ψ_f => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_f_D"];IN_is=OUT),
    )
  # writevtk(Ω.Γg,path*"Gamma_g")
  # writevtk(Ω.Γi,path*"Gamma_i")
  # error()

  # Setup spaces
  uin(x) = VectorValue(16x[2]*(1/2-x[2]),0.0)
  uin_dot_e1(x) = uin(x)⋅VectorValue(1.0,0.0)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order,space=:P)
  reffe_p = ReferenceFE(lagrangian,Float64,order,space=:P)

  V = TestFESpace(Ω.Ω_act_f,reffe_u,conformity=:H1,
    dirichlet_tags=["Gamma_f_D","Gamma_NoSlipTop","Gamma_NoSlipBottom"])
  Q = TestFESpace(Ω.Ω_act_f,reffe_p,conformity=:H1)

  # Trial spaces
  U = TrialFESpace(V,[uin,VectorValue(0.0,0.0),VectorValue(0.0,0.0)])
  P = TrialFESpace(Q)

  # Multifield spaces
  X = MultiFieldFESpace([U,P])
  Y = MultiFieldFESpace([V,Q])

  ### Weak form

  ## Fluid
  # Properties
  # Re = 60 # Reynolds number
  ρ = 1.0 # Density
  cl = 2R # Characteristic length
  u0_max = sum(∫(uin_dot_e1)dΓf_D)/sum(∫(1)dΓf_D)
  μ = ρ*cl*u0_max/Re # Viscosity
  ν = μ/ρ # Kinematic viscosity

  # Stabilization parameters
  γ_Nu(h) = α_Nu*μ/h
  γ_u(h) = α_u*μ*h
  γ_p(h) = α_p*h^3/μ

  # Terms
  σf_n(u,p,n) = μ*∇(u) ⋅ n - p*n
  a_Ω(u,v) = μ*(∇(u) ⊙ ∇(v))
  b_Ω(v,p) = -p*(∇⋅v)
  a_Γ(u,v,n) = - μ*(n⋅∇(u)) ⋅ v - μ*(n⋅∇(v)) ⋅ u + (γ_Nu ∘ hₕ)*(u⋅v)
  b_Γ(v,p,n) = (n⋅v)*p
  ju(u,v)  = mean(γ_u ∘ hₕ)*jump(Ω.n_Γg ⋅ ∇(u)) ⋅ jump(Ω.n_Γg ⋅ ∇(v))
  jp(p,q)  = mean(γ_p ∘ hₕ)*jump(Ω.n_Γi ⋅ ∇(p)) * jump(Ω.n_Γi ⋅ ∇(q))

  function A((u,p),(v,q))
    n_Γ = -get_normal_vector(Ω.Γ)
    return ∫(a_Ω(u,v) + b_Ω(v,p) + b_Ω(u,q))Ω.dΩf +
      ∫(a_Γ(u,v,n_Γ) + b_Γ(v,p,n_Γ) + b_Γ(u,q,n_Γ))Ω.dΓ +
      ∫(ju(u,v))Ω.dΓg - ∫(jp(p,q))Ω.dΓi
  end

  L((v,q)) = 0

  op = AffineFEOperator(A,L,X,Y)
  uh,ph = solve(op);

  n_Γ = -get_normal_vector(Ω.Γ)

  fname="cutfem_P1P1_Re$(Re)_msh$(mesh_size)_aNU$(α_Nu)_au$(α_u)_ap$(α_p)"
  writevtk(Ω.Ωf,data_path*fname,
    cellfields=["uh"=>uh,"ph"=>ph])
  fname="cutfem_P1P1_GAMMA_Re$(Re)_msh$(mesh_size)_aNU$(α_Nu)_au$(α_u)_ap$(α_p)"
  writevtk(Ω.Γ,data_path*fname,
    cellfields=["uh"=>uh,"ph"=>ph,"surface_force"=>σf_n(uh,ph,n_Γ)])
end

with_mpi() do distribute
  ncpus = 4
  ranks = distribute(LinearIndices((ncpus,)))
  params = (;Re=60,mesh_size=0.01,α_Nu=0,α_u=0,α_p=0)
  main_bodyfitted(ranks,params)
  params = (;Re=60,mesh_size=0.01,α_Nu=10,α_u=0.1,α_p=0.1)
  main_cutfem(ranks,params)
end