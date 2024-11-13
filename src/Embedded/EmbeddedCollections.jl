
struct EmbeddedCollection
  recipes :: Vector{<:Function}
  objects :: Dict{Symbol,Any}
  bgmodel :: Union{<:DiscreteModel,<:DistributedDiscreteModel}
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
  for r in c.recipes
    merge!(c.objects,pairs(r(cutgeo)))
  end
  return c
end

function add_recipe!(c::EmbeddedCollection,r::Function)
  push!(c.recipes,r)
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
