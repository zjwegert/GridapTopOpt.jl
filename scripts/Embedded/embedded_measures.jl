using GridapTopOpt: to_parray_of_arrays
using GridapDistributed
using GridapDistributed: DistributedDiscreteModel, DistributedDomainContribution
using GridapEmbedded.LevelSetCutters
# using GridapEmbedded.LevelSetCutters: DiscreteGeometry
using GridapEmbedded.Distributed: DistributedDiscreteGeometry
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Helpers
import Gridap.Geometry: get_node_coordinates, collect1d

# function cv_to_dof(cv,V)
# 	fv=zeros(eltype(eltype(cv)),num_free_dofs(V))
# 	gather_free_values!(fv,V,cv)
# end

# function field_to_cv(uh::FEFunction)
# 	get_cell_dof_values(uh)
# end

# function field_to_cv(cf::CellField)
# 	cf.cell_field.args[1]
# end

# function get_geo_params(φh::FEFunction,Vbg)
#   Ωbg = get_triangulation(Vbg)
#   bgmodel = get_background_model(Ωbg)
#   point_to_coords = collect1d(get_node_coordinates(bgmodel))
#   ls_to_point_to_value_unmasked = field_to_cv(φh)
#   p0 = cv_to_dof(ls_to_point_to_value_unmasked,Vbg)
#   geo1 = DiscreteGeometry(p0,point_to_coords)
#   geo2 = DiscreteGeometry(-1*p0,point_to_coords,name="")
#   get_geo_params(geo1,geo2,bgmodel)
# end

# function get_geo_params(φh::CellField,Vbg)
#   Ωbg = get_triangulation(Vbg)
#   bgmodel = get_background_model(Ωbg)
#   point_to_coords = collect1d(get_node_coordinates(bgmodel))
#   ls_to_point_to_value_unmasked = field_to_cv(φh)
#   p0 = cv_to_dof(ls_to_point_to_value_unmasked,Vbg)
#   geo1 = DiscreteGeometry(p0,point_to_coords)
#   geo2 = DiscreteGeometry(-1*p0,point_to_coords,name="")
#   get_geo_params(geo1,geo2,bgmodel)
# end

# function get_geo_params(φ::AbstractVector,Vbg)
# 	Ωbg = get_triangulation(Vbg)
# 	bgmodel = get_background_model(Ωbg)
# 	point_to_coords = collect1d(get_node_coordinates(bgmodel))
# 	geo1 = DiscreteGeometry(φ,point_to_coords,name="")
# 	geo2 = DiscreteGeometry(-φ,point_to_coords,name="")
# 	get_geo_params(geo1,geo2,bgmodel)
# end

# The above is more efficent for serial problems but does not work for periodic problems or distirbuted
#  problems. This is subject to change.

_DiscreteGeometry(φ,model::DistributedDiscreteModel;name::String="") =
  DistributedDiscreteGeometry(φ,model;name)

_DiscreteGeometry(φh::CellField,model::CartesianDiscreteModel;name::String="") =
  DiscreteGeometry(φh(collect1d(get_node_coordinates(model))),collect1d(get_node_coordinates(model));name)

function get_geo_params(ϕh::CellField,Vbg)#::GridapDistributed.DistributedFESpace)
  Ωbg = get_triangulation(Vbg)
  bgmodel = get_background_model(Ωbg)
  ϕhminus = FEFunction(Vbg,-get_free_dof_values(ϕh))
  geo1 = _DiscreteGeometry(ϕh,bgmodel)
  geo2 = _DiscreteGeometry(ϕhminus,bgmodel,name="")
  get_geo_params(geo1,geo2,bgmodel)
end

function get_geo_params(φ::AbstractVector,Vbg)#::GridapDistributed.DistributedFESpace)
	Ωbg = get_triangulation(Vbg)
	bgmodel = get_background_model(Ωbg)
	ϕh = FEFunction(Vbg,φ); ϕhminus = FEFunction(Vbg,-φ)
	geo1 = _DiscreteGeometry(ϕh,bgmodel,name="")
	geo2 = _DiscreteGeometry(ϕhminus,bgmodel,name="")
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
	(Ω1,Ω2,Γ),(dΩ1,dΩ2,dΓ)
end

#######################

# For problems where the derivatives are known, we only want to update measures once
mutable struct EmbeddedMeasureCache
  const space
  trians
  measures

  function EmbeddedMeasureCache(φ,space)
    measures = get_geo_params(φ,space)
    new(space,measures)
  end
end

function update_embedded_measures!(φ,s::EmbeddedMeasureCache)
  trians, measures = get_geo_params(φ,s.space)
  s.measures = measures
  s.trians = trians
  return s.measures
end

function get_embedded_measures(φ,s::EmbeddedMeasureCache)
  return s.measures
end

function get_embedded_triangulations(s::EmbeddedMeasureCache)
  return s.trians
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
  @warn "Automatic differentation is currently disabled for `IntegrandWithEmbeddedMeasure` types" maxlog=1
  @check 0 < K <= length(uh)
  _f(uk) = if K == length(uh)
    F.F(uh[1:K-1]...,uk,uh[K+1:end]...,F.dΩ...,F.get_embedded_dΩ(uk)...)
  else
    F.F(uh[1:K-1]...,uk,uh[K+1:end]...,F.dΩ...,F.get_embedded_dΩ(uh[end])...)
  end
  # return Gridap.gradient(_f,uh[K]) # AD is currently disabled due to error (under investigation)
  return Gridap.gradient(u -> ∑(∫(0)F.dΩ[i] for i = 1:length(F.dΩ)),uh[K])
end

# This doesn't currently work, we need a nice way to differentiate local_embedded_measures
#  like the above.
function Gridap.gradient(F::IntegrandWithEmbeddedMeasure,uh::Vector,K::Int)
  @warn "Automatic differentation is currently disabled for `IntegrandWithEmbeddedMeasure` types" maxlog=1
  @check 0 < K <= length(uh)
  local_fields = map(local_views,uh) |> to_parray_of_arrays
  local_measures = map(local_views,F.dΩ) |> to_parray_of_arrays

  # if K == length(uh)
  #   # Not sure how to do this just yet...
  # else
  #   local_embedded_measures = map(local_views,F.get_embedded_dΩ(uh[end])) |> to_parray_of_arrays
  #   contribs = map(local_measures,local_embedded_measures,local_fields) do dΩ,dΩ_embedded,lf
  #     _f = u -> F.F(lf[1:K-1]...,u,lf[K+1:end]...,dΩ...,dΩ_embedded...)
  #     return Gridap.Fields.gradient(_f,lf[K])
  #   end
  # end

  # Placeholder
  local_embedded_measures = map(local_views,F.get_embedded_dΩ(uh[end])) |> to_parray_of_arrays
  contribs = map(local_measures,local_embedded_measures,local_fields) do dΩ,dΩ_embedded,lf
    _f = u -> ∑(∫(0)dΩ[i] for i = 1:length(dΩ))
    return Gridap.Fields.gradient(_f,lf[K])
  end

  return DistributedDomainContribution(contribs)
end

Gridap.gradient(F::IntegrandWithEmbeddedMeasure,uh) = Gridap.gradient(F,[uh],1)