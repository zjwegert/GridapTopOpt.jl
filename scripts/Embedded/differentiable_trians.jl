
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

# Mutable measure

mutable struct MutableMeasure <: Measure
  state :: Measure
  params 
end

function Measure(trian::DifferentiableTriangulation,args...;kwargs...)
  state = Measure(trian.state,args...;kwargs...)
  meas  = MutableMeasure(state,(args,kwargs))
  push!(trian.children, objectid(meas) => meas)
  return meas
end

function update_measure!(meas::MutableMeasure,trian::Triangulation)
  args, kwargs = meas.params
  meas.state = Measure(trian,args...;kwargs...)
  return meas
end

CellData.integrate(f,b::MutableMeasure) = integrate(f,b.state)