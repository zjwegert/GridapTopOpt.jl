
using Gridap.CellData, Gridap.Geometry, Gridap.Helpers

function CellData.get_contribution(a::DomainContribution,trian::Geometry.AppendedTriangulation)
  if haskey(a.dict,trian)
    return a.dict[trian]
  elseif haskey(a.dict,trian.a)
    return a.dict[trian.a]
  elseif haskey(a.dict,trian.b)
    return a.dict[trian.b]
  else
    @unreachable """\n
    There is not contribution associated with the given mesh in this DomainContribution object.
    """
  end
end


# DifferentiableTriangulation

mutable struct DifferentiableTriangulation{Dc,Dp} <: Triangulation{Dc,Dp}
  recipe   :: Function
  state    :: Triangulation{Dc,Dp}
  children :: IdDict{UInt64,Measure}
end

function DifferentiableTriangulation(f::Function,φh)
  DifferentiableTriangulation(f,f(φh),IdDict{UInt64,Measure}())
end

(t::DifferentiableTriangulation)(φh) = update_trian!(t,φh)

function update_trian!(trian::DifferentiableTriangulation,φh)
  trian.state = trian.recipe(φh)
  for child in values(trian.children)
    update_measure!(child,trian.state)
  end
  return trian
end

function Gridap.FESpaces._change_argument(
  op,f,trian::DifferentiableTriangulation,uh::SingleFieldFEFunction
)
  U = get_fe_space(uh)
  function g(cell_u)
    cf = CellField(U,cell_u)
    update_trian!(trian,cf)
    cell_grad = f(cf)
    get_contribution(cell_grad,trian)
  end
  g
end

function GridapEmbedded.Interfaces.get_background_model(t::DifferentiableTriangulation)
  get_background_model(t.state)
end

Gridap.Geometry.get_glue(ttrian::DifferentiableTriangulation,val::Val{d}) where d = get_glue(ttrian.state,val)

# Mutable measure

mutable struct MutableMeasure <: Measure
  state :: Measure
  trian :: Triangulation
  params
end

function Measure(trian::DifferentiableTriangulation,args...;kwargs...)
  state = Measure(trian.state,args...;kwargs...)
  meas  = MutableMeasure(state, trian,(args,kwargs))
  push!(trian.children, objectid(meas) => meas)
  return meas
end

function update_measure!(meas::MutableMeasure,trian::Triangulation)
  args, kwargs = meas.params
  meas.state = Measure(trian,args...;kwargs...)
  return meas
end

function CellData.integrate(f,b::MutableMeasure)
  c = integrate(f,b.state.quad)
  cont = DomainContribution()
  add_contribution!(cont,b.trian,c)
  cont
end

## Fix for _array_cache used by Connor
using Gridap.Arrays: IndexItemPair,LazyArray,testitem,return_cache,return_value
function Gridap.Arrays._array_cache!(dict::Dict,a::LazyArray)
  @boundscheck begin
    if ! all(map(isconcretetype, map(eltype, a.args)))
      for n in 1:length(a.args)
        @notimplementedif ! all(map(isconcretetype, map(eltype, a.args[n])))
      end
    end
    if ! (eltype(a.maps) <: Function)
      @notimplementedif ! isconcretetype(eltype(a.maps))
    end
  end
  gi = testitem(a.maps)
  fi = map(testitem,a.args)
  cg = array_cache(dict,a.maps)
  cf = map(fi->array_cache(dict,fi),a.args)
  cgi = return_cache(gi, fi...)
  index = -1
  #item = evaluate!(cgi,gi,testargs(gi,fi...)...)
  item = return_value(gi,fi...)
  (cg, cgi, cf), IndexItemPair(index, item)
end