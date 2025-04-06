using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt

using GridapDistributed,PartitionedArrays,GridapPETSc

function main(ranks)
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

  model = GmshDiscreteModel(ranks,(@__DIR__)*"/../Meshes/mesh_3d_finer.msh")
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
  lsf(x) = min(max(fin(x),fholes(x,5,0.5)),fsolid(x))
  φh = interpolate(lsf,V_φ)
  φh_nondesign = interpolate(fsolid,V_φ)

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
      :n_Γ        => get_normal_vector(Γ), # Note, need to recompute inside obj/constraints to compute derivs
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

  function a_fluid((),(u,p),(v,q),φ)
    return ∫(∇(u) ⊙ ∇(v) + p*q)Ω.dΩf
  end
  l_fluid((),(v,q),φ) = ∫(0*q)Ω.dΩf
  
  function a_solid2(d,s,φ)
    return ∫(∇(d) ⊙ ∇(s))Ω.dΩs
  end
  _g = VectorValue(0,0,0)
  function l_solid2(s,φ)
    return ∫(_g ⋅ s)Ω.dΓ
  end

  (_UP,_VQ),(_R,_T) = build_spaces(Ω.Ω_act_s,Ω.Ω_act_f)
  
  op_singlefield = AffineFEOperator((x,y)->a_solid2(x,y,φh),y->l_solid2(y,φh),_R,_T)
  
  op_multifield = AffineFEOperator((x,y)->a_fluid((),x,y,φh),y->l_fluid((),y,φh),_UP,_VQ)
  
  return "Yay?"
end

with_mpi() do distribute
  ncpus = 256
  ranks = distribute(LinearIndices((ncpus,)))
  petsc_options = "-ksp_converged_reason -ksp_error_if_not_converged true -pc_type lu -pc_factor_mat_solver_type superlu_dist -mat_superlu_dist_printstat"
  GridapPETSc.with(;args=split(petsc_options)) do
    main(ranks)
  end
end
