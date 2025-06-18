
"""
    struct EmbeddedCollection
      recipes :: Vector{<:Function}
      objects :: Dict{Symbol,Any}
      bgmodel :: DiscreteModel
    end

A collection of embedded objects on the same background model. This structure
provides a way to update all the stored objects at once.

## Constructor

- `EmbeddedCollection(recipes::Union{<:Function,Vector{<:Function}},bgmodel::DiscreteModel[,φh])`

If provided, `φh` will be used to compute the initial collection of objects. If not provided,
the collection will remain empty until `update_collection!` is called.

## API:

- `update_collection!(c::EmbeddedCollection,φh)`: Update the collection of objects using the level set function `φh`.
- `add_recipe!(c::EmbeddedCollection,r::Function[,φh])`: Add a recipe to the collection. Update the collection if `φh` is provided.

"""
struct EmbeddedCollection
  recipes :: Vector{<:Function}
  objects :: Dict{Symbol,Any}
  bgmodel :: Union{<:DiscreteModel,<:DistributedDiscreteModel}
  function EmbeddedCollection(recipes::Vector{<:Function},objects::Dict{Symbol,Any},
      bgmodel::Union{<:DiscreteModel,<:DistributedDiscreteModel})

    if ~check_polytopes(bgmodel)
      i_am_main(get_parts(bgmodel)) && @warn """
      Non-TET/TRI polytopes are simplexified by GridapEmbedded when cutting. As a result,
      derivative information from AD will not be correct when using a mesh that isn't made of TRI/TET.

      Please use a mesh with TRI/TET polytopes to ensure correctness of derivative results.
    """
    end
    new(recipes, objects, bgmodel)
  end
end

function check_polytopes(bgmodel)
  polytopes = get_polytopes(bgmodel)
  return all(p -> p == TRI || p == TET, polytopes)
end

function check_polytopes(bgmodel::DistributedDiscreteModel)
  map(check_polytopes, local_views(bgmodel))
end

function EmbeddedCollection(recipes::Function,bgmodel)
  EmbeddedCollection(Function[recipes],Dict{Symbol,Any}(),bgmodel)
end

function EmbeddedCollection(recipes::Vector{<:Function},bgmodel)
  EmbeddedCollection(recipes,Dict{Symbol,Any}(),bgmodel)
end

function EmbeddedCollection(recipes::Union{<:Function,Vector{<:Function}},bgmodel,φ0)
  c = EmbeddedCollection(recipes,bgmodel)
  update_collection!(c,φ0)
end

(c::EmbeddedCollection)(φh) = update_collection!(c,φh)

function update_collection!(c::EmbeddedCollection,φh)
  geo = DiscreteGeometry(φh,c.bgmodel)
  cutgeo = cut(c.bgmodel,geo)
  cutgeo_facet = cut_facets(c.bgmodel,geo)
  for r in c.recipes
    merge!(c.objects,pairs(r(cutgeo,cutgeo_facet,φh)))
  end
  return c
end

function add_recipe!(c::EmbeddedCollection,r::Function)
  push!(c.recipes,r)
  return c
end

function add_recipe!(c::EmbeddedCollection,r::Function,φh)
  push!(c.recipes,r)
  update_collection!(c,φh)
end

function Base.getindex(c::EmbeddedCollection,key)
  return c.objects[key]
end

function Base.getproperty(c::EmbeddedCollection,sym::Symbol)
  objects = getfield(c,:objects)
  if haskey(objects,sym)
    objects[sym]
  else
    getfield(c,sym)
  end
end

function Base.propertynames(c::EmbeddedCollection, private::Bool=false)
  (fieldnames(typeof(c))...,keys(getfield(c,:objects))...)
end
