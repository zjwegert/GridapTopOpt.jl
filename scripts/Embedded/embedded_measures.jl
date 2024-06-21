using GridapTopOpt: to_parray_of_arrays
using GridapDistributed
using GridapDistributed: DistributedDiscreteModel, DistributedDomainContribution
using GridapEmbedded.LevelSetCutters
# using GridapEmbedded.LevelSetCutters: DiscreteGeometry
using GridapEmbedded.Distributed: DistributedDiscreteGeometry
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Helpers
import Gridap.Geometry: get_node_coordinates, collect1d

function get_geo_params(ϕh::CellField,Vbg)
  Ωbg = get_triangulation(Vbg)
  bgmodel = get_background_model(Ωbg)
  geo1 = DiscreteGeometry(ϕh,bgmodel)
  geo2 = DiscreteGeometry(-ϕh,bgmodel,name="")
  get_geo_params(geo1,geo2,bgmodel)
end

function get_geo_params(φ::AbstractVector,Vbg)
	Ωbg = get_triangulation(Vbg)
	bgmodel = get_background_model(Ωbg)
	ϕh = FEFunction(Vbg,φ); ϕhminus = FEFunction(Vbg,-φ)
	geo1 = DiscreteGeometry(ϕh,bgmodel,name="")
	geo2 = DiscreteGeometry(ϕhminus,bgmodel,name="")
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

function update_embedded_measures!(s::EmbeddedMeasureCache,φ)
  trians, measures = get_geo_params(φ,s.space)
  s.measures = measures
  s.trians = trians
  return s.measures
end

function update_embedded_measures!(s::EmbeddedMeasureCache,φ,space)
  trians, measures = get_geo_params(φ,space)
  s.measures = measures
  s.trians = trians
  return s.measures
end

function get_embedded_measures(s::EmbeddedMeasureCache,args...)
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
  @check 0 < K <= length(uh)
  _f(uk) = if K == length(uh)
    F.F(uh[1:K-1]...,uk,uh[K+1:end]...,F.dΩ...,F.get_embedded_dΩ(uk)...)
  else
    F.F(uh[1:K-1]...,uk,uh[K+1:end]...,F.dΩ...,F.get_embedded_dΩ(uh[end])...)
  end
  return Gridap.gradient(_f,uh[K])
end

function Gridap.gradient(F::IntegrandWithEmbeddedMeasure,uh::Vector,K::Int)
  @check 0 < K <= length(uh)
  local_fields = map(local_views,uh) |> to_parray_of_arrays
  local_measures = map(local_views,F.dΩ) |> to_parray_of_arrays

  if K == length(uh)
    # Need to test. My hope is that locally this will work - ghosts will be wrong but
    #  will be thrown away later.
    contribs = map(local_measures,local_views(uh[end]),local_fields) do dΩ,φh,lf
      _f = u -> F.F(lf[1:K-1]...,u,lf[K+1:end]...,dΩ...,F.get_embedded_dΩ(φh,get_fe_space(φh))...)
      return Gridap.Fields.gradient(_f,lf[K])
    end
  else
    local_embedded_measures = map(local_views,F.get_embedded_dΩ(uh[end])) |> to_parray_of_arrays
    contribs = map(local_measures,local_embedded_measures,local_fields) do dΩ,dΩ_embedded,lf
      _f = u -> F.F(lf[1:K-1]...,u,lf[K+1:end]...,dΩ...,dΩ_embedded...)
      return Gridap.Fields.gradient(_f,lf[K])
    end
  end

  return DistributedDomainContribution(contribs)
end
Gridap.gradient(F::IntegrandWithEmbeddedMeasure,uh) = Gridap.gradient(F,[uh],1)