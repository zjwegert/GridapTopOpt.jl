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

function initial_lsf(lsf::Symbol,geo_info)
  L,H,x0,l,w,a,b,cw = geo_info
  _e = 0.0#1e-3 #TODO: Test with _e = 0.0 or different values to see if this is culprit
  if lsf == :box
    return ((x,y,z),) -> max(abs(x-x0),abs(y),abs(z-H/2)) - 0.4+_e
  elseif lsf == :sphere
    return ((x,y,z),) -> sqrt((x-x0)^2 + y^2 + (z-H/2)^2) - 0.4+_e
  elseif lsf == :wall
    f0((x,y,z),a,b) = max(2/a*abs(x-x0),1/(b/2+1)*abs(y-b/2+1),2/(H-2cw)*abs(z-H/2))-1
    return x -> min(f0(x,l*(1+_e),b*(1+_e)),f0(x,w*(1+_e),a*(1+_e)))
  elseif lsf == :initial
    f0((x,y,z),a,b) = max(2/a*abs(x-x0),1/(b/2+1)*abs(y-b/2+1),2/(H-2cw)*abs(z-H/2))-1
    f1((x,y,z),q,r) = - cos(q*π*x)*cos(q*π*y)*cos(q*π*z)/q - r/q
    fin(x) = f0(x,l*(1+5_e),a*(1+5_e))
    fsolid(x) = min(f0(x,l*(1+_e),b*(1+_e)),f0(x,w*(1+_e),a*(1+_e)))
    fholes((x,y,z),q,r) = max(f1((x,y,z),q,r),f1((x-1/q,y,z),q,r))
    return x -> min(max(fin(x),fholes(x,5,0.5)),fsolid(x))
  else
    error("Geometry not undefined")
  end
end

function main(ranks)
  path = "./results/Staggered_FCM_3d_FSI_NavierStokes_GMSH_MWEBroken/"
  i_am_main(ranks) && mkpath(path)

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
  writevtk(model,path*"model")

  Ω_act = Triangulation(model)

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)

  lsf = initial_lsf(:wall,(L,H,x0,l,w,a,b,cw))
  φh = interpolate(lsf,V_φ)

  # Setup integration meshes and measures
  order = 1
  degree = 2*(order+1)

  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)
  Ωs = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
  Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
  dΩs = Measure(Ωs,degree)
  dΩf = Measure(Ωf,degree)

  # Setup spaces

  reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
  reffe_p = ReferenceFE(lagrangian,Float64,order,space=:P)

  # Test spaces
  V = TestFESpace(Ω_act,reffe_u,conformity=:H1)
  Q = TestFESpace(Ω_act,reffe_p,conformity=:H1)

  # Trial spaces
  U = TrialFESpace(V)
  P = TrialFESpace(Q)

  # Multifield spaces
  UP = MultiFieldFESpace([U,P])
  VQ = MultiFieldFESpace([V,Q])

  ### Weak form
  # Terms
  a_Ω(u,v) = 1.0*(∇(u) ⊙ ∇(v))

  a_fluid((u,p),(v,q),φ) =
  ∫( a_Ω(u,v) )dΩf + # Volume terms
  ∫( a_Ω(u,v) )dΩs # Stabilization terms

  return a_fluid(get_trial_fe_basis(UP),get_fe_basis(VQ),φh)
end

_test = with_debug() do distribute
  mesh_partition = (2,2,2)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  petsc_options = "-ksp_monitor -ksp_error_if_not_converged true"
  GridapPETSc.with(;args=split(petsc_options)) do
    main(ranks)
  end
end