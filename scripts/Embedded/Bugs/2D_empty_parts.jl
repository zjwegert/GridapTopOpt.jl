using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt

using GridapDistributed,PartitionedArrays,GridapPETSc

function main(ranks,mesh_partition,n;x0=(0.1,0.1))
  path = "./results/Testing AD with empty parts/"
  i_am_main(ranks) && mkpath(path)

  domain = (0,1,0,1,0,1)
  cell_partition = (n,n)
  base_model = UnstructuredDiscreteModel(CartesianDiscreteModel(ranks,mesh_partition,domain,cell_partition))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)
  Ω_bg = Triangulation(model)
  writevtk(model,path*"model")

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)

  lsf(x) = sqrt((x[1]-x0[1])^2+(x[2]-x0[2])^2)-0.2 # Sphere
  φh = interpolate(lsf,V_φ)
  writevtk(get_triangulation(φh),path*"initial_lsf",cellfields=["φh"=>φh])

  # Correct LSF
  _φ = get_free_dof_values(φh)
  map(local_views(_φ)) do φ
    idx = findall(isapprox(0.0;atol=eps()),φ)
    if !isempty(idx)
      i_am_main(ranks) && println("    Correcting level values at $(length(idx)) nodes")
    end
    φ[idx] .+= 100*eps(eltype(φ))
  end
  consistent!(_φ) |> wait

  # Cut
  order = 1
  degree = 2*(order+1)

  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)
  cutgeo_facets = cut_facets(model,geo)
  # IN + CUT
  Ω_IN = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
  Ω_act_IN = Triangulation(cutgeo,ACTIVE)
  # OUT + CUT
  Ω_OUT = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
  Ω_act_OUT = Triangulation(cutgeo,ACTIVE_OUT)
  # INTERFACE
  Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
  # GHOST SKEL
  Γg = GhostSkeleton(cutgeo)
  n_Γg = get_normal_vector(Γg)
  # OUT SKEL
  Γi_OUT = SkeletonTriangulation(cutgeo_facets,ACTIVE_OUT)
  n_Γi_OUT  = get_normal_vector(Γi_OUT)
  # IN SKEL
  Γi_IN = SkeletonTriangulation(cutgeo_facets,ACTIVE)
  n_Γi_IN  = get_normal_vector(Γi_IN)
  # MEAS
  dΩ_bg = Measure(Ω_bg,degree)
  dΩ_IN = Measure(Ω_IN,degree)
  dΩ_OUT = Measure(Ω_OUT,degree)
  dΓg = Measure(Γg,degree)
  dΓ = Measure(Γ,degree)
  dΓi_IN = Measure(Γi_IN,degree)
  dΓi_OUT = Measure(Γi_OUT,degree)

  writevtk(Γ,path*"Gamma")
  writevtk(Γi_OUT,path*"Gammai_OUT")
  writevtk(Γi_IN,path*"Gammai_IN")
  writevtk(Γg,path*"Gammag")

  # Some other spaces
  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order,space=:P)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1,space=:P)
  reffe_d = ReferenceFE(lagrangian,VectorValue{2,Float64},order)

  V = TestFESpace(Ω_act_OUT,reffe_u,conformity=:H1,dirichlet_tags=[5,6])
  Q = TestFESpace(Ω_act_OUT,reffe_p,conformity=:L2)
  VQ = MultiFieldFESpace([V,Q])
  S = TestFESpace(Ω_act_IN,reffe_d,conformity=:H1,dirichlet_tags=[5,6,7,8])

  U = TrialFESpace(V,[VectorValue(0.0,0.1),VectorValue(0.1,0.1)])
  P = TrialFESpace(Q)
  UP = MultiFieldFESpace([U,P])
  D = TrialFESpace(S,[VectorValue(0.0,0.1),VectorValue(0.1,0.1),
    VectorValue(0.1,0.0),VectorValue(0.0,0.0)])

  xh = interpolate([x->VectorValue(x[1],x[2]),x->x[1]],UP)
  uh,ph = xh
  dh = interpolate(x->VectorValue(x[2],x[1]),D)

  writevtk(Ω_IN,path*"Omega_IN",cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
  writevtk(Ω_OUT,path*"Omega_OUT",cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
  writevtk(Ω_act_OUT,path*"Omega_act_OUT")
  writevtk(Ω_act_IN,path*"Omega_act_IN")

  σf_n(u,p,n) = ∇(u) ⋅ n - p*n
  a_Ω(u,v) = ∇(u) ⊙ ∇(v)
  b_Ω(v,p) = -p*(∇⋅v)
  a_Γ(u,v,n) = - (n⋅∇(u)) ⋅ v - (n⋅∇(v)) ⋅ u + u⋅v
  b_Γ(v,p,n) = (n⋅v)*p
  ju(u,v) = jump(n_Γg ⋅ ∇(u)) ⋅ jump(n_Γg ⋅ ∇(v))
  jp(p,q) = jump(p) * jump(q)

  σ(ε) = tr(ε)*one(ε) + 2*ε
  a_s_Ω(d,s) = ε(s) ⊙ (σ ∘ ε(d))
  j_s_k(d,s) = jump(n_Γg ⋅ ∇(s)) ⋅ jump(n_Γg ⋅ ∇(d))

  #### AD through fluid weak form [PASSES]
  function a_fluid((u,p),(v,q),φ)
    n_Γ = -get_normal_vector(Γ)
    return ∫(a_Ω(u,v) + b_Ω(v,p) + b_Ω(u,q))dΩ_OUT +
      ∫(a_Γ(u,v,n_Γ) + b_Γ(v,p,n_Γ) + b_Γ(u,q,n_Γ))dΓ +
      ∫(ju(u,v))dΓg - ∫(jp(p,q))dΓi_OUT
  end
  l_fluid((),(v,q),φ) =  ∫(0q)dΩ_OUT
  R_fluid((u,p),(v,q),φ) = a_fluid((u,p),(v,q),φ) - l_fluid((),(v,q),φ)

  # sum(R_fluid(xh,xh,φh))
  # ∇(φ -> R_fluid(xh,xh,φ),φh)
  # ∇(φ -> R_fluid(zero(UP),zero(UP),φ),φh)
  # ∇(φ -> R_fluid(zero(UP),zero(VQ),φ),φh)
  # ∇(φ -> R_fluid(zero(VQ),zero(VQ),φ),φh)

  #### AD through elast weak form [FAILS]
  function a_solid(((u,p),),d,s,φ)
    return ∫(a_s_Ω(d,s))dΩ_IN +
      ∫(j_s_k(d,s))dΓg
  end
  function l_solid(((u,p),),s,φ)
    n = -get_normal_vector(Γ)
    return ∫(-σf_n(u,p,n) ⋅ s)dΓ
  end
  R_solid(((u,p),),d,s,φ) = a_solid(((u,p),),d,s,φ) - l_solid(((u,p),),s,φ)

  sum(∫(dh)dΩ_IN) # This works
  sum(∫(∇(dh))dΩ_IN) # This fails

  ## Untested due to above
  sum(R_solid((xh,),dh,dh,φh))
  ∇(φ -> R_solid((zero(UP),),dh,dh,φ),φh)
  ∇(φ -> R_solid((xh,),zero(D),zero(D),φ),φh)
  ∇(φ -> R_solid((xh,),zero(S),zero(S),φ),φh)
  ∇(φ -> R_solid((xh,),zero(D),zero(S),φ),φh)

  #### AD through objective/constraint functionals [FAILS]
  J_comp(d,φ) = ∫(ε(d) ⊙ (σ ∘ ε(d)))dΩ_IN
  Vol(d,φ) = ∫(1.0)dΩ_IN - ∫(0.1)dΩ_bg

  sum(J_comp(dh,φh)) # Fails due to above issue
  ∇(φ -> J_comp(dh,φ),φh) # Untested
  ∇(d -> J_comp(d,φh),dh) # Untested
  sum(Vol(dh,φh))
  ∇(φ -> Vol(dh,φ),φh) # Passes
  ∇(d -> Vol(d,φh),dh) # Fails for different LSFs, see serial version below for easier debug.
                       #  This one seems unrelated to empty parts issue.
end

with_debug() do distribute
  mesh_partition = (2,2)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  # main(ranks,mesh_partition,30;x0=(0.5,0.5))
  main(ranks,mesh_partition,30;x0=(0.1,0.1))
end

################## Last error in serial for easier debug ##################

function main_serial_for_last_error(n;x0=(0.1,0.1))
  domain = (0,1,0,1,0,1)
  cell_partition = (n,n)
  base_model = UnstructuredDiscreteModel(CartesianDiscreteModel(domain,cell_partition))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)
  Ω_bg = Triangulation(model)

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)

  lsf(x) = sqrt((x[1]-x0[1])^2+(x[2]-x0[2])^2)-0.2 # FAILS
  # lsf(x) = - cos(4π*x[1])*cos(4π*x[2])-0.4 # WORKS
  φh = interpolate(lsf,V_φ)

  # Cut
  order = 1
  degree = 2*(order+1)

  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)
  # IN + CUT
  Ω_IN = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
  Ω_act_IN = Triangulation(cutgeo,ACTIVE)
  # MEAS
  dΩ_bg = Measure(Ω_bg,degree)
  dΩ_IN = Measure(Ω_IN,degree)

  # Some other spaces
  reffe_d = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  S = TestFESpace(Ω_act_IN,reffe_d,conformity=:H1)
  dh = zero(S)
  Vol(d,φ) = ∫(1.0)dΩ_IN - ∫(0.1)dΩ_bg

  ∇(d -> Vol(d,φh),dh)
end

main_serial_for_last_error(30;x0=(0.5,0.5))