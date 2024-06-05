using Pkg; Pkg.activate()

using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData
import Gridap.Geometry: get_node_coordinates, collect1d

function cv_to_dof(cv,V)
	fv=zeros(eltype(eltype(cv)),num_free_dofs(V))
	gather_free_values!(fv,V,cv)
end

function field_to_cv(uh::FEFunction)
	get_cell_dof_values(uh)
end

function field_to_cv(cf::CellField)
	cv=cf.cell_field.args[1]
end

function get_geo_params(ϕₕ::FEFunction,Vbg)
  Ωbg = get_triangulation(Vbg)
  bgmodel = get_background_model(Ωbg)
  point_to_coords = collect1d(get_node_coordinates(bgmodel))
  ls_to_point_to_value_unmasked = field_to_cv(ϕₕ)
  p0 = cv_to_dof(ls_to_point_to_value_unmasked,Vbg)
  geo1 = DiscreteGeometry(p0,point_to_coords)
  geo2 = DiscreteGeometry(-1*p0,point_to_coords,name="")
  get_geo_params(geo1,geo2,bgmodel)
end

function get_geo_params(ϕₕ::CellField,Vbg)
  Ωbg = get_triangulation(Vbg)
  bgmodel = get_background_model(Ωbg)
  point_to_coords = collect1d(get_node_coordinates(bgmodel))
  ls_to_point_to_value_unmasked = field_to_cv(ϕₕ)
  p0 = cv_to_dof(ls_to_point_to_value_unmasked,Vbg)
  geo1 = DiscreteGeometry(p0,point_to_coords)
  geo2 = DiscreteGeometry(-1*p0,point_to_coords,name="")
  get_geo_params(geo1,geo2,bgmodel)
end

function get_geo_params(ϕ::AbstractVector,Vbg)
	Ωbg = get_triangulation(Vbg)
	bgmodel = get_background_model(Ωbg)
	point_to_coords = collect1d(get_node_coordinates(bgmodel))
	geo1 = DiscreteGeometry(ϕ,point_to_coords,name="")
	geo2 = DiscreteGeometry(-ϕ,point_to_coords,name="")
	get_geo_params(geo1,geo2,bgmodel)
end

function get_geo_params(geo1,geo2,bgmodel)
	cutgeo1= cut(bgmodel,geo1)
	cutgeo2= cut(bgmodel,geo2)
	# Setup interpolation meshes
	Ω1_act = Triangulation(cutgeo1,ACTIVE)
	Ω2_act = Triangulation(cutgeo2,ACTIVE)
	# Setup integration meshes
	Ω1 = Triangulation(cutgeo1,PHYSICAL)
	Ω2 = Triangulation(cutgeo2,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo1)
	# Setup Lebesgue measures
	order = 1
	degree = 2*order
	dΩ1 = Measure(Ω1,degree)
	dΩ2 = Measure(Ω2,degree)
	dΓ = Measure(Γ,degree)
	(;dΩ1,dΩ2,dΓ)#,debug=(;cutgeo1,cutgeo2,order, Ω1,Ω2,Ω1_act,Ω2_act,Γ))
end

model = CartesianDiscreteModel((0,1,0,1),(100,100));
Ω = Triangulation(model)
dΩ = Measure(Ω,2)
Γ_N = BoundaryTriangulation(model,tags=6)
dΓ_N = Measure(Γ_N,2)

reffe_scalar = FiniteElements(PhysicalDomain(),model,lagrangian,Float64,1) # ReferenceFE(lagrangian,Float64,1)
V = TestFESpace(model,reffe_scalar)
U = TrialFESpace(V)
V_φ = TestFESpace(model,reffe_scalar)

φh = interpolate(x->-cos(4π*x[1])*cos(4*pi*x[2])/4-0.2/4,V_φ)

# _Γ = get_triangulation(Gridap.CellData.get_cell_quadrature(get_geo_params(φh,V_φ).dΓ))
# writevtk(_Γ,"./results/Gamma",cellfields=["normal"=>get_normal_vector(_Γ)])

function a(u,v,φ)
  geo = get_geo_params(φ,V_φ)
  dΩ1 = geo.dΩ1; dΩ2 = geo.dΩ2
  ∫(∇(u)⋅∇(v))dΩ1 + ∫(10^-3*∇(u)⋅∇(v))dΩ2
end

l(v,φ) = ∫(v)dΓ_N

op = AffineFEOperator((u,v) -> a(u,v,φh),v -> l(v,φh),U,V)

function j(u,φ)
  geo = get_geo_params(φ,V_φ)
  dΩ1 = geo.dΩ1; dΩ2 = geo.dΩ2
  ∫(u)dΩ1 + ∫(u)dΩ2
end

∇(φ -> a(zero(U),zero(U),φ))(φh) # error calling _gradient_nd!
∇(φ -> j(zero(U),φ))(φh) # domain contribution not found error