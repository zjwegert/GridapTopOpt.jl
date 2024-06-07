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
	(dΩ1,dΩ2,dΓ)
end

#######################

# For problems where the derivatives are known, we only want to update measures once
mutable struct EmbeddedMeasureCache
  const space
  measures

  function EmbeddedMeasureCache(φ,space)
    measures = get_geo_params(φ,space)
    new(space,measures)
  end
end

function update_embedded_measures!(φ,s::EmbeddedMeasureCache)
  s.measures = get_geo_params(φ,s.space)
  return s.measures
end

function get_embedded_measures(φ,s::EmbeddedMeasureCache)
  return s.measures
end

#######################

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
  return Gridap.gradient(u -> ∑(∫(0)F.dΩ[i] for i = 1:length(F.dΩ)),uh[K]) # AD is currently disabled
end

Gridap.gradient(F::IntegrandWithEmbeddedMeasure,uh) = Gridap.gradient(F,[uh],1)