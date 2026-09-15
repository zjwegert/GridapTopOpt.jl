# DiscreteGeometryFromFEFunction

function DiscreteGeometryFromFEFunction(φh::CellField,model::DistributedDiscreteModel;name::String="")
  geometries = map(local_views(φh),local_views(model)) do φh, model
    DiscreteGeometryFromFEFunction(φh,model;name)
  end
  DistributedDiscreteGeometry(geometries)
end

# Cut

function cut(
  bgmodel::DistributedDiscreteModel,
  geo::DistributedDiscreteGeometry{<:AbstractVector{<:DiscreteGeometryFromFEFunction}}
)
  cut(PolytopalLevelSetCutter(),bgmodel,geo)
end

function cut_facets(
  bgmodel::DistributedDiscreteModel,
  geo::DistributedDiscreteGeometry{<:AbstractVector{<:DiscreteGeometryFromFEFunction}}
)
  cut_facets(PolytopalLevelSetCutter(),bgmodel,geo)
end

# Differentiation, mostly constructors...

function FESpaces._change_argument(
  op,f,
  local_trians::AbstractArray{<:Union{<:DifferentiableCutPolyTriangulation,<:DifferentiableAppendedCutPolyTriangulation,<:DifferentiableCutPolyTriangulationView}},
  uh::GridapDistributed.DistributedADTypes
)
  function dist_cf(uh::DistributedCellField,cfs)
    DistributedCellField(cfs,get_triangulation(uh))
  end
  function dist_cf(uh::DistributedMultiFieldCellField,cfs)
    sf_cfs = map(DistributedCellField,
      [tuple_of_arrays(map(cf -> Tuple(cf.single_fields),cfs))...],
      map(get_triangulation,uh)
    )
    DistributedMultiFieldCellField(sf_cfs,cfs)
  end

  uhs = local_views(uh)
  spaces = map(get_fe_space,uhs)
  function g(cell_u)
    cfs = map(CellField,spaces,cell_u)
    cf = dist_cf(uh,cfs)
    map(update_trian!,local_trians,spaces,local_views(cf))
    cg = f(cf)
    map(local_trians,spaces) do Ω, V
      update_trian!(Ω,V,nothing)
    end
    map(get_contribution,local_views(cg),local_trians)
  end
  g
end

function DifferentiableTriangulation(
  trian::DistributedTriangulation,
  cutgeo::DistributedEmbeddedDiscretization,
  in_or_out,
  geo::DistributedDiscreteGeometry
)
  model = get_background_model(trian)
  trians = map(local_views(trian),local_views(cutgeo),local_views(geo)) do ltrian, lcutgeo, lgeo
    DifferentiableTriangulation(ltrian,lcutgeo,in_or_out,lgeo)
  end
  return DistributedTriangulation(trians,model)
end

function DifferentiableTriangulation(
  trian::DistributedTriangulation,
  cutgeo::DistributedEmbeddedDiscretization,
  in_or_out,
  name::String
)
  geo = get_geometry(get_distributed_geometry(cutgeo),name)
  DifferentiableTriangulation(trian,cutgeo,in_or_out,geo)
end

function DifferentiableTriangulation(
  trian::DistributedTriangulation,
  cutgeo::DistributedEmbeddedDiscretization
)
  model = get_background_model(trian)
  trians = map(local_views(trian),local_views(cutgeo)) do ltrian, lcutgeo
    DifferentiableTriangulation(ltrian,lcutgeo)
  end
  return DistributedTriangulation(trians,model)
end

function DifferentiableTriangulation(
  trian::DistributedTriangulation,
  cutgeo::DistributedEmbeddedDiscretization,
  in_or_out
)
  model = get_background_model(trian)
  trians = map(local_views(trian),local_views(cutgeo)) do ltrian, lcutgeo
    DifferentiableTriangulation(ltrian,lcutgeo,in_or_out)
  end
  return DistributedTriangulation(trians,model)
end

function DifferentiableTriangulation(
  trian::DistributedTriangulation,
  cutgeo::DistributedEmbeddedDiscretization,
  name::String
)
  geo = get_geometry(get_distributed_geometry(cutgeo),name)
  DifferentiableTriangulation(trian,cutgeo,PHYSICAL_IN,geo)
end

function DifferentiableTriangulation(
  trian::DistributedTriangulation,
  cutgeo::DistributedEmbeddedDiscretization,
  geo::DistributedDiscreteGeometry
)
  model = get_background_model(trian)
  trians = map(local_views(trian),local_views(cutgeo),local_views(geo)) do ltrian, lcutgeo, lgeo
    DifferentiableTriangulation(ltrian,lcutgeo,PHYSICAL_IN,lgeo)
  end
  return DistributedTriangulation(trians,model)
end

function DifferentiableTriangulation(
  cutgeo :: DistributedEmbeddedDiscretization,
  args...
)
  trian = Triangulation(cutgeo,args...)
  return DifferentiableTriangulation(trian,cutgeo,args...)
end

function DifferentiableEmbeddedBoundary(
  trian::DistributedTriangulation,
  cutgeo::DistributedEmbeddedDiscretization
)
  model = get_background_model(trian)
  trians = map(local_views(trian),local_views(cutgeo)) do ltrian, lcutgeo
    DifferentiableEmbeddedBoundary(ltrian,lcutgeo)
  end
  return DistributedTriangulation(trians,model)
end

function DifferentiableEmbeddedBoundary(
  trian::DistributedTriangulation,
  cutgeo::DistributedEmbeddedDiscretization,
  geo::DistributedDiscreteGeometry
)
  model = get_background_model(trian)
  trians = map(local_views(trian),local_views(cutgeo),local_views(geo)) do ltrian, lcutgeo, lgeo
    DifferentiableEmbeddedBoundary(ltrian,lcutgeo,lgeo)
  end
  return DistributedTriangulation(trians,model)
end

function DifferentiableEmbeddedBoundary(
  trian::DistributedTriangulation,
  cutgeo::DistributedEmbeddedDiscretization,
  name::String
)
  geo = get_geometry(get_distributed_geometry(cutgeo),name)
  DifferentiableEmbeddedBoundary(trian,cutgeo,geo)
end

function DifferentiableEmbeddedBoundary(
  trian::DistributedTriangulation,
  cutgeo::DistributedEmbeddedDiscretization,
  geo1::DistributedDiscreteGeometry,
  geo2::DistributedDiscreteGeometry
)
  model = get_background_model(trian)
  trians = map(local_views(trian),local_views(cutgeo),local_views(geo1),local_views(geo2)) do ltrian, lcutgeo, lgeo1, lgeo2
    DifferentiableEmbeddedBoundary(ltrian,lcutgeo,lgeo1,lgeo2)
  end
  return DistributedTriangulation(trians,model)
end

function DifferentiableEmbeddedBoundary(
  trian::DistributedTriangulation,
  cutgeo::DistributedEmbeddedDiscretization,
  name1::String,
  name2::String
)
  geo = get_distributed_geometry(cutgeo)
  geo1 = get_geometry(geo,name1)
  geo2 = get_geometry(geo,name2)
  DifferentiableEmbeddedBoundary(trian,cutgeo,geo1,geo2)
end

function DifferentiableEmbeddedBoundary(
  cutgeo :: DistributedEmbeddedDiscretization,
  args...
)
  trian = EmbeddedBoundary(cutgeo,args...)
  return DifferentiableEmbeddedBoundary(trian,cutgeo,args...)
end

# Missing # TODO: All go in DistributedDiscreteGeometries.jl

function get_distributed_geometry(a::DistributedEmbeddedDiscretization)
  geometries = map(local_views(a)) do a
    get_geometry(a)
  end
  DistributedDiscreteGeometry(geometries)
end

function Distributed.get_geometry(a::DistributedDiscreteGeometry,name::String)
  geometries = map(local_views(a)) do a
    get_geometry(a,name)
  end
  DistributedDiscreteGeometry(geometries)
end

function Base.union(a::DistributedDiscreteGeometry,b::DistributedDiscreteGeometry;name::String="")
  map(local_views(a),local_views(b)) do la, lb
    union(la,lb;name)
  end |> DistributedDiscreteGeometry
end

function Base.intersect(a::DistributedDiscreteGeometry,b::DistributedDiscreteGeometry;name::String="")
  map(local_views(a),local_views(b)) do la, lb
    intersect(la,lb;name)
  end |> DistributedDiscreteGeometry
end

function Base.setdiff(a::DistributedDiscreteGeometry,b::DistributedDiscreteGeometry;name::String="")
  map(local_views(a),local_views(b)) do la, lb
    setdiff(la,lb;name)
  end |> DistributedDiscreteGeometry
end

function Base.:!(a::DistributedDiscreteGeometry;name::String="")
  map(local_views(a)) do la
    Base.:!(la;name)
  end |> DistributedDiscreteGeometry
end