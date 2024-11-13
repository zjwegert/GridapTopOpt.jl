
struct EmbeddedCollection
  generate :: Function
  objects  :: Dict{Symbol,Any}
  bgmodel  :: Union{<:DiscreteModel,<:DistributedDiscreteModel}

  function EmbeddedCollection(
    generate::Function,bgmodel,φ0
  )
    geo = DiscreteGeometry(φ0,bgmodel)
    cutgeo = cut(bgmodel,geo)
    objects = Dict{Symbol,Any}(pairs(generate(cutgeo)))
    new(generate,objects,bgmodel)
  end
end

(c::EmbeddedCollection)(φh) = update_collection!(c,φh)

function update_collection!(c::EmbeddedCollection,φh)
  geo = DiscreteGeometry(φh,c.bgmodel)
  cutgeo = cut(c.bgmodel,geo)
  merge!(c.objects,pairs(c.generate(cutgeo)))
  return c
end

function Base.getindex(c::EmbeddedCollection,key)
  return c.objects[key]
end
