using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapGmsh
using GridapTopOpt

using GridapDistributed,PartitionedArrays

function main(ranks)
  path = "./results/IsolatedGmsh_BiteTest_MPI/"
  files_path = path*"data/"
  i_am_main(ranks) && mkpath(files_path)

  # Load gmsh mesh (Currently need to update mesh.geo and these values concurrently)
  L = 2.0;
  H = 0.5;
  x0 = 0.5;
  l = 0.4;
  w = 0.025;
  a = 0.3;
  b = 0.01;

  model = GmshDiscreteModel(ranks,(@__DIR__)*"/mesh_finer.msh")
  writevtk(model,path*"model")

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)

  _e = 1e-3
  f0((x,y),W,H) = max(2/W*abs(x-x0),1/(H/2+1)*abs(y-H/2+1))-1
  f1((x,y),q,r) = - cos(q*π*x)*cos(q*π*y)/q - r/q
  fin(x) = f0(x,l*(1+5_e),a*(1+5_e))
  fsolid(x) = min(f0(x,l*(1+_e),b*(1+_e)),f0(x,w*(1+_e),a*(1+_e)))
  fholes((x,y),q,r) = max(f1((x,y),q,r),f1((x-1/q,y),q,r))
  φf(x) = min(max(fin(x),fholes(x,25,0.2)),fsolid(x))

  _φf2(x) = max(φf(x),-(max(2/0.2*abs(x[1]-0.319),2/0.2*abs(x[2]-0.3))-1))
  φf2(x) = min(_φf2(x),sqrt((x[1]-0.35)^2+(x[2]-0.26)^2)-0.025)
  φh = interpolate(φf2,V_φ)

  # Setup integration meshes and measures
  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)

  Ωs = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
  Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)

  ψ_s =  GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_s_D"];groups=((GridapTopOpt.CUT,IN),OUT)) # Good one
  ψ_f =  GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_f_D"];groups=(IN,(GridapTopOpt.CUT,OUT)))

  writevtk(get_triangulation(φh),path*"initial_islands",cellfields=["φh"=>φh,"ψ_f"=>ψ_f,"ψ_s"=>ψ_s];append=false)
  writevtk(Ωs,path*"Omega_s_initial";append=false)
  writevtk(Ωf,path*"Omega_f_initial";append=false)
end

with_debug() do distribute
  ncpus = 4
  ranks = distribute(LinearIndices((ncpus,)))
  main(ranks)
end

