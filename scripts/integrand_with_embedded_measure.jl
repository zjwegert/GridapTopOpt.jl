using Pkg; Pkg.activate()

using Gridap,GridapTopOpt

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

import GridapTopOpt: AbstractIntegrandWithMeasure

struct IntegrandWithEmbeddedMeasure{A,B,C} <: AbstractIntegrandWithMeasure
  F               :: A
  dΩ              :: B
  get_embedded_dΩ :: C
end

(F::IntegrandWithEmbeddedMeasure)(args...) = F.F(args...,F.dΩ...,F.get_embedded_dΩ(args[end])...)

function Gridap.gradient(F::IntegrandWithEmbeddedMeasure,uh::Vector{<:FEFunction},K::Int)
  # @check 0 < K <= length(uh)
  _f(uk) = if K == length(uh)
    F.F(uh[1:K-1]...,uk,uh[K+1:end]...,F.dΩ...,F.get_embedded_dΩ(uk)...)
  else
    F.F(uh[1:K-1]...,uk,uh[K+1:end]...,F.dΩ...,F.get_embedded_dΩ(uh[end])...)
  end
  # return Gridap.gradient(_f,uh[K])
  return Gridap.gradient(uh->∫(0)dΩ,uh[K]) # AD is currently disabled
end

# This currently does nothing and just ensures everything runs OK
function Gridap.gradient(F::IntegrandWithEmbeddedMeasure,uh::Vector,K::Int)
  # @check 0 < K <= length(uh)
  local_fields = map(local_views,uh) |> to_parray_of_arrays
  local_measures = map(local_views,F.dΩ) |> to_parray_of_arrays
  contribs = map(local_measures,local_fields) do dΩ,lf
    _f = u -> ∑(∫(0)dΩ[i] for i = 1:length(dΩ))
    return Gridap.Fields.gradient(_f,lf[K])
  end
  return DistributedDomainContribution(contribs)
end

Gridap.gradient(F::IntegrandWithEmbeddedMeasure,uh) = Gridap.gradient(F,[uh],1)

path="./results/CellFEM_thermal_compliance_ALM/"
n = 200
order = 1
γ = 0.2
γ_reinit = 0.5
max_steps = floor(Int,order*minimum(n)/10)
tol = 1/(1*order^2)/minimum(n)
κ = 1
vf = 0.4
α_coeff = 4max_steps*γ
iter_mod = 1

model = CartesianDiscreteModel((0,1,0,1),(n,n));
el_Δ = get_el_Δ(model)
f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= 0.2 + eps() || x[2] >= 0.8 - eps()))
f_Γ_N(x) = (x[1] ≈ 1 && 0.4 - eps() <= x[2] <= 0.6 + eps())
update_labels!(1,model,f_Γ_D,"Gamma_D")
update_labels!(2,model,f_Γ_N,"Gamma_N")

## Triangulations and measures
Ω = Triangulation(model)
Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
dΩ = Measure(Ω,2*order)
dΓ_N = Measure(Γ_N,2*order)
vol_D = sum(∫(1)dΩ)

## Spaces
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_D"])
U = TrialFESpace(V,0.0)
V_φ = TestFESpace(model,reffe_scalar)
V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
U_reg = TrialFESpace(V_reg,0)

φh = interpolate(x->-cos(4π*x[1])*cos(4*pi*x[2])/4-0.2/4,V_φ)

## Weak form
a(u,v,φ,dΩ,dΓ_N,dΩ1,dΩ2,dΓ) = ∫(∇(u)⋅∇(v))dΩ1 + ∫(10^-3*∇(u)⋅∇(v))dΩ2
l(v,φ,dΩ,dΓ_N,dΩ1,dΩ2,dΓ) = ∫(v)dΓ_N

## Optimisation functionals
J(u,φ,dΩ,dΓ_N,dΩ1,dΩ2,dΓ) = ∫(∇(u)⋅∇(u))dΩ1 #+ ∫(10^-3*∇(u)⋅∇(u))dΩ2
dJ(q,u,φ,dΩ,dΓ_N,dΩ1,dΩ2,dΓ) = ∫(∇(u)⋅∇(u)*q)dΓ;
Vol(u,φ,dΩ,dΓ_N,dΩ1,dΩ2,dΓ) = ∫(1/vol_D)dΩ1 - ∫(vf)dΩ;
dVol(q,u,φ,dΩ,dΓ_N,dΩ1,dΩ2,dΓ) = ∫(-1/vol_D*q)dΓ

a_iem = IntegrandWithEmbeddedMeasure(a,(dΩ,dΓ_N),φ->get_geo_params(φ,V_φ))
l_iem = IntegrandWithEmbeddedMeasure(l,(dΩ,dΓ_N),φ->get_geo_params(φ,V_φ))

J_iem = IntegrandWithEmbeddedMeasure(J,(dΩ,dΓ_N),φ->get_geo_params(φ,V_φ))
dJ_iem = IntegrandWithEmbeddedMeasure(dJ,(dΩ,dΓ_N),φ->get_geo_params(φ,V_φ))
Vol_iem = IntegrandWithEmbeddedMeasure(Vol,(dΩ,dΓ_N),φ->get_geo_params(φ,V_φ))
dVol_iem = IntegrandWithEmbeddedMeasure(dVol,(dΩ,dΓ_N),φ->get_geo_params(φ,V_φ))


ls_evo = HamiltonJacobiEvolution(FirstOrderStencil(2,Float64),model,V_φ,10^-3,max_steps)


## Setup solver and FE operators
state_map = AffineFEStateMap(a_iem,l_iem,U,V,V_φ,U_reg,φh,(dΩ,dΓ_N))
pcfs = PDEConstrainedFunctionals(J_iem,[Vol_iem],state_map,analytic_dJ=dJ_iem,analytic_dC=[dVol_iem])

## Hilbertian extension-regularisation problems
α = α_coeff*maximum(el_Δ)
a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

## Optimiser
rm(path,force=true,recursive=true)
mkpath(path)
optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
  γ,γ_reinit,verbose=true,constraint_names=[:Vol])
for (it,uh,φh) in optimiser
  dΩ1,dΩ2,dΓ = get_geo_params(φh,V_φ)
  if iszero(it % iter_mod)
    writevtk(dΩ1.quad.trian,path*"Omega_out$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
    writevtk(dΓ.quad.trian,path*"Gamma_out$it",cellfields=["normal"=>get_normal_vector(dΓ.quad.trian)])
  end
  write_history(path*"/history.txt",optimiser.history)
end
it = get_history(optimiser).niter; uh = get_state(pcfs)
_dΩ1,_dΩ2,_dΓ = get_geo_params(φh,V_φ)
writevtk(_dΩ1.quad.trian,path*"Omega_out$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
writevtk(_dΓ.quad.trian,path*"Gamma_out$it",cellfields=["normal"=>get_normal_vector(_dΓ.quad.trian)])