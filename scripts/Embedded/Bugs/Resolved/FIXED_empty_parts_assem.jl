using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt

using GridapDistributed,PartitionedArrays,GridapPETSc

function main(ranks,mesh_partition,n;x0=(0.1,0.1))
  domain = (0,1,0,1,0,1)
  cell_partition = (n,n)
  base_model = UnstructuredDiscreteModel(CartesianDiscreteModel(ranks,mesh_partition,domain,cell_partition))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)
  Ω_bg = Triangulation(model)

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)

  lsf(x) = sqrt((x[1]-x0[1])^2+(x[2]-x0[2])^2)-0.2 # Sphere
  φh = interpolate(lsf,V_φ)

  # Correct LSF
  GridapTopOpt.correct_ls!(φh)

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

  i_am_main(ranks) && println(" -- Compute gradients")
  uh0 = zero(UP);
  grad = ∇(φ -> R_fluid(uh0,uh0,φ),φh)
  i_am_main(ranks) && println(" -- Collect cell vec")
  vecdata = Gridap.FESpaces.collect_cell_vector(V_φ,grad)
  assem_deriv = SparseMatrixAssembler(V_φ,V_φ)
  i_am_main(ranks) && println(" -- Alloc vec")
  grad_vec = Gridap.FESpaces.allocate_vector(assem_deriv,vecdata)

  #### AD through elast weak form
  function a_solid(((u,p),),d,s,φ)
    return ∫(a_s_Ω(d,s))dΩ_IN +
      ∫(j_s_k(d,s))dΓg
  end
  function l_solid(((u,p),),s,φ)
    n = -get_normal_vector(Γ)
    return ∫(-σf_n(u,p,n) ⋅ s)dΓ
  end
  R_solid(((u,p),),d,s,φ) = a_solid(((u,p),),d,s,φ) - l_solid(((u,p),),s,φ)

  ## Untested due to above
  i_am_main(ranks) && println(" -- Compute gradients")
  grad = ∇(φ -> R_solid((xh,),dh,dh,φ),φh)
  i_am_main(ranks) && println(" -- Collect cell vec")
  vecdata = Gridap.FESpaces.collect_cell_vector(V_φ,grad)
  assem_deriv = SparseMatrixAssembler(V_φ,V_φ)
  i_am_main(ranks) && println(" -- Alloc vec")
  grad_vec = Gridap.FESpaces.allocate_vector(assem_deriv,vecdata)

  return grad_vec
end

with_mpi() do distribute
  mesh_partition = (2,2)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  # main(ranks,mesh_partition,30;x0=(0.5,0.5))
  main(ranks,mesh_partition,30;x0=(0.1,0.1))
end