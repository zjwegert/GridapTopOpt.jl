module HeatReintialiserGmshTest

using Gridap, Gridap.Adaptivity, Gridap.Geometry, Gridap.Helpers
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt, GridapSolvers
using GridapGmsh
using Test

function main_gmsh(;vtk=false)
  path = "./results/HeatReinitialiser_gmsh/"
  files_path = path*"data/"
  mkpath(files_path)

  # Load gmsh mesh (Currently need to update mesh.geo and these values concurrently)
  H = 0.5;
  x0 = 0.5;
  l = 0.4;
  w = 0.025;
  a = 0.3;
  b = 0.01;

  model = GmshDiscreteModel(pathof(GridapTopOpt)*"/../../test/meshes/mesh_finer.msh")
  vtk && writevtk(model,path*"model")

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
  φh_old = FEFunction(V_φ,copy(get_free_dof_values(φh)))

  reinit_method = HeatReinitialiser(V_φ,model;
    boundary_tags=["Gamma_NoSlipBottom","Gamma_NoSlipTop","Gamma_f_D","Gamma_f_N","Gamma_s_D"],
    t = get_element_diameter_field(model)*get_element_diameter_field(model))
    # t = minimum(get_element_diameters(model))^2)
  reinit!(reinit_method,φh);
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2)
  vtk && writevtk(Ω,path*"gmsh_heat_sdf",cellfields=["φh_old"=>φh_old,"φh"=>φh,"|∇(φh)|"=>(norm ∘ ∇(φh))])

  L2error(u) = sqrt(sum(∫(u ⋅ u)dΩ))
  # Check |∇(φh)|
  @test abs(L2error(norm ∘ ∇(φh))-1) < 1e-2 # <- this is a difficult SDF to create, 1e-2 is pretty good...
end

function main_adaptive(;vtk=false)
  path = "./results/HeatReinitialiser_gmsh/"
  files_path = path*"data/"
  mkpath(files_path)

  model, geo_params = build_model(2*50,50,b=1/30,w=1/30)

  hₕ = get_element_diameter_field(model)
  hmin = minimum(get_element_diameters(model))

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)

  _e = 1/3*hmin
  L,H,x0,l,w,a,b = geo_params
  f0((x,y),W,H) = max(2/W*abs(x-x0),1/(H/2+1)*abs(y-H/2+1))-1
  f1((x,y),q,r) = - cos(q*π*x)*cos(q*π*y)/q - r/q
  fin(x) = f0(x,l*(1+5_e),a*(1+5_e))
  fsolid(x) = min(f0(x,l*(1+_e),b*(1+_e)),f0(x,w*(1+_e),a*(1+_e)))
  fholes((x,y),q,r) = max(f1((x,y),q,r),f1((x-1/q,y),q,r))
  lsf(x) = min(max(fin(x),fholes(x,18,0.6)),fsolid(x))
  φh = interpolate(lsf,V_φ)
  φh_old = FEFunction(V_φ,copy(get_free_dof_values(φh)))

  reinit_method = HeatReinitialiser(V_φ,model;
    boundary_tags=["Gamma_Bottom","Gamma_Top","Gamma_f_D","Gamma_f_N","Gamma_s_D"],
    t = get_element_diameter_field(model)*get_element_diameter_field(model))
    # t = mean(get_element_diameters(model))^2)
  reinit!(reinit_method,φh);
  Ω = Triangulation(model)
  vtk && writevtk(Ω,path*"adaptive_heat_sdf",cellfields=["φh_old"=>φh_old,"φh"=>φh,"|∇(φh)|"=>(norm ∘ ∇(φh))])
end

## Build refined model
function build_cells_to_refine(model)
  # Create refinement map
  tagged_to_marked_cells = get_face_tag_index(get_face_labeling(model),"RefineBox",2)
  return findall(convert(Vector{Bool},tagged_to_marked_cells))
end

function build_model(nx,ny;L=1.0,H=0.5,x0=0.5,l=0.4,w=0.05,a=0.3,b=0.05)
  geo_params = (;L,H,x0,l,w,a,b)
  base_model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,L,0,H),(nx,ny)))
  f_Γ_Top(x) = x[2] == H
  f_Γ_Bottom(x) = x[2] == 0.0
  f_Γ_D(x) = x[1] == 0.0
  f_Γ_N(x) = x[1] == L
  f_box(x) = 0.0 <= x[2] <= 1.1a + eps() && (x0 - 1.1*l/2 - eps() <= x[1] <= x0 + 1.1*l/2 + eps())
  f_NonDesign(x) = ((x0 - w/2 - eps() <= x[1] <= x0 + w/2 + eps() && 0.0 <= x[2] <= a + eps()) ||
    (x0 - l/2 - eps() <= x[1] <= x0 + l/2 + eps() && 0.0 <= x[2] <= b + eps()))
  update_labels!(1,base_model,f_box,"RefineBox")
  ref_model = refine(base_model, refinement_method = "barycentric")
  ref_model = refine(ref_model; cells_to_refine=build_cells_to_refine(ref_model))
  ref_model = refine(ref_model; cells_to_refine=build_cells_to_refine(ref_model))
  model = get_model(ref_model)
  update_labels!(2,model,f_Γ_Top,"Gamma_Top")
  update_labels!(3,model,f_Γ_Bottom,"Gamma_Bottom")
  update_labels!(4,model,f_Γ_D,"Gamma_f_D")
  update_labels!(5,model,f_Γ_N,"Gamma_f_N")
  update_labels!(6,model,f_NonDesign,"Omega_NonDesign")
  update_labels!(7,model,x->f_NonDesign(x) && f_Γ_Bottom(x),"Gamma_s_D")
  return model, geo_params
end

main_gmsh(;vtk=false)
# main_adaptive(;vtk=true)

end