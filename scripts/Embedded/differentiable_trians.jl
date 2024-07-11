
using Gridap.CellData, Gridap.Geometry, Gridap.Helpers, Gridap.Arrays

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

function FESpaces._compute_cell_ids(uh,ttrian::AppendedTriangulation)
  ids_a = FESpaces._compute_cell_ids(uh,ttrian.a)
  ids_b = FESpaces._compute_cell_ids(uh,ttrian.b)
  lazy_append(ids_a,ids_b)
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

function FESpaces._change_argument(
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

function FESpaces._compute_cell_ids(uh,ttrian::DifferentiableTriangulation)
  FESpaces._compute_cell_ids(uh,ttrian.state)
end

function Geometry.get_background_model(t::DifferentiableTriangulation)
  get_background_model(t.state)
end

Geometry.get_glue(ttrian::DifferentiableTriangulation,val::Val{d}) where d = get_glue(ttrian.state,val)

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
