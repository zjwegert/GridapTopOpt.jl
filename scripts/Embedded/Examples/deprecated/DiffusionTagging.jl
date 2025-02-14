using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers
using GridapGmsh
using GridapTopOpt

path = "./results/DiffusionTagging/"
mkpath(path)

γ_evo = 0.2
max_steps = 24 # Based on number of elements in vertical direction divided by 10
vf = 0.025
α_coeff = γ_evo*max_steps
iter_mod = 10
D = 2

# Load gmsh mesh (Currently need to update mesh.geo and these values concurrently)
L = 2.0;
H = 0.5;
x0 = 0.5;
l = 0.4;
w = 0.025;
a = 0.3;
b = 0.01;
vol_D = 2.0*0.5

model = GmshDiscreteModel("scripts/Embedded/Examples/FluidStructure/Meshes/mesh_finer.msh")
writevtk(model,path*"model")

Ω_act = Triangulation(model)
hₕ = CellField(get_element_diameters(model),Ω_act)
hmin = minimum(get_element_diameters(model))

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
# φf(x) = min(max(fin(x),fholes(x,22,0.6)),fsolid(x))
# φh = interpolate(φf,V_φ)

φf2(x) = max(φf(x),-(max(2/0.2*abs(x[1]-0.32),2/0.2*abs(x[2]-0.321))-1))
φh = interpolate(φf2,V_φ)

order = 1
degree = 2*(order+1)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)
Ωs = Triangulation(cutgeo,PHYSICAL)
Ωf = Triangulation(cutgeo,PHYSICAL_OUT)
dΩs = Measure(Ωs,degree)
dΩf = Measure(Ωf,degree)

Γg = GhostSkeleton(cutgeo)
n_Γg = get_normal_vector(Γg)
dΓg = Measure(Γg,degree)

h_ψ = 0.01
ψ_inf = 1.0
γ_GP(h) = 0.05h
k = 1.0
kw = 1000.0
kt = 0.99

reffe = ReferenceFE(lagrangian,Float64,1)
Ξ = TestFESpace(Triangulation(cutgeo,ACTIVE_OUT),reffe,conformity=:H1,dirichlet_tags=["Gamma_f_D"])
Ψ = TrialFESpace(Ξ)

J(c) = k*∇(c)
r_Ω(ψ,ξ) = ∫(∇(ξ)⋅J(ψ) - ξ*h_ψ*ψ)dΩf
r_GP(ψ,ξ) = ∫(mean(γ_GP ∘ hₕ)*jump(∇(ξ)⋅n_Γg)*jump(J(ψ)⋅n_Γg))dΓg
A(ψ,ξ) = r_Ω(ψ,ξ) + r_GP(ψ,ξ)
B(ξ) = ∫(-ξ*h_ψ*ψ_inf)dΩf

op = AffineFEOperator(A,B,Ξ,Ψ)
ψh = solve(op)

ψbar(ψ) = 1/2 + 1/2*tanh(kw*(ψ-kt*ψ_inf))

bgcell_to_inoutcut = compute_bgcell_to_inoutcut(cutgeo,geo)
cellfields = ["ψ"=>ψh,"ψbar"=>ψbar ∘(ψh),
  "ξ"=>GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_f_D"];groups=(OUT,(IN,GridapTopOpt.CUT)))]

writevtk(get_triangulation(model),path*"psih_bg",cellfields=cellfields,celldata=["bgcell_to_inoutcut"=>bgcell_to_inoutcut])
writevtk(Ωf,path*"psih_physical",cellfields=cellfields,celldata=["bgcell_to_inoutcut"=>bgcell_to_inoutcut])
writevtk(Triangulation(cutgeo,CUT_OUT),path*"psih_cut_out",cellfields=cellfields,celldata=["bgcell_to_inoutcut"=>bgcell_to_inoutcut])
writevtk(Triangulation(cutgeo,CUT_IN),path*"psih_cut_in",cellfields=cellfields,celldata=["bgcell_to_inoutcut"=>bgcell_to_inoutcut])
writevtk(Triangulation(cutgeo,ACTIVE_OUT),path*"psih_active_out",cellfields=cellfields,celldata=["bgcell_to_inoutcut"=>bgcell_to_inoutcut])
writevtk(Triangulation(cutgeo,ACTIVE_IN),path*"psih_active_in",cellfields=cellfields,celldata=["bgcell_to_inoutcut"=>bgcell_to_inoutcut])

writevtk(Triangulation(cutgeo,ACTIVE_OUT),path*"psih_out")