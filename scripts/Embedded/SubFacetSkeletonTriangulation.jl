using Gridap
using Gridap.Geometry, Gridap.Adaptivity, Gridap.Algebra, Gridap.Arrays,
  Gridap.CellData, Gridap.Fields, Gridap.Helpers, Gridap.ReferenceFEs, Gridap.Polynomials

using GridapEmbedded
using GridapEmbedded.Interfaces
using GridapEmbedded.LevelSetCutters

using GridapEmbedded.Interfaces: SubFacetData, SubFacetTriangulation

Base.round(a::VectorValue{D,T};kwargs...) where {D,T} = VectorValue{D,T}(round.(a.data;kwargs...))

function SubFacetSkeletonTriangulation(trian::SubFacetTriangulation{Dc,Dp}) where {Dc,Dp}
  subfacets = trian.subfacets

  if Dp == 2
    Γ_facet_to_points = map(Reindex(subfacets.point_to_coords),subfacets.facet_to_points)
    Γ_pts = unique(round.(reduce(vcat,Γ_facet_to_points);sigdigits=12))
    Γ_facet_to_pts = convert(Vector{Vector{Int32}},map(v->indexin(round.(v;sigdigits=12),Γ_pts),Γ_facet_to_points))

    grid_top = GridTopology(Γ.subgrid,Table(Γ_facet_to_pts),IdentityVector(length(Γ_pts)))
    grid = UnstructuredGrid(
      Γ_pts,Table(Γ_facet_to_pts),
      Γ.subgrid.reffes,
      Γ.subgrid.cell_types,
      Γ.subgrid.orientation_style,
      get_facet_normal(Γ)
    )

    Γ_model = UnstructuredDiscreteModel(grid,grid_top,FaceLabeling(grid_top))
    Γs = SkeletonTriangulation(Γ_model)
  else
    @notimplemented
  end

  return Γs
end

# function Geometry.get_grid(trian::SubFacetSkeletonTriangulation)
#   trian.subgrid
# end

order = 1
n = 10
_model = CartesianDiscreteModel((0,1,0,1),(n,n))
cd = Gridap.Geometry.get_cartesian_descriptor(_model)
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = ref_model.model
# model = simplexify(_model)
# model = _model # NOTE: currently broken:
#  If we really want this, then need to check that point is on
#  boundary of bgcell when computing maps ..._Γ_facet_to_points.

Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)

reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe_scalar)
φh = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-0.52,V_φ) # Circle
x_φ = get_free_dof_values(φh)

if any(isapprox(0.0;atol=10^-10),x_φ)
  error("Issue with lvl set alignment")
end

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)
Ω_cut = Triangulation(cutgeo,PHYSICAL)
Γ = EmbeddedBoundary(cutgeo)
Γg = GhostSkeleton(cutgeo,CUT)
dΓ = Measure(Γ,2*order)

Γs = SubFacetSkeletonTriangulation(Γ)

_2d_cross(n) = VectorValue(n[2],-n[1]);

n1 = CellField(evaluate(get_facet_normal(Γg.plus),Point(0)),Γs)
n2 = CellField(evaluate(get_facet_normal(Γg.minus),Point(0)),Γs)
n_∂Ω_plus = Operation(_2d_cross)(get_normal_vector(Γs).plus)
n_∂Ω_minus = Operation(_2d_cross)(get_normal_vector(Γs).minus)
nˢ = Operation(v->(-_2d_cross(v)))(n1)

m = get_normal_vector(Γs)

# ∇φh_Γ = change_domain(∇(φh),Ω,ReferenceDomain(),Γ,ReferenceDomain())
# Γ_trian = Triangulation(get_background_model(Γs.plus))
# ∇φh_Γ_trian = GenericCellField(get_data(φh_Γ),Γ_trian,ReferenceDomain())
# ∇φh_Γs_plus = change_domain(φh_Γ_trian,Γ_trian,ReferenceDomain(),Γs.plus,ReferenceDomain())
# ∇φh_Γs = move_to_sub_facet_skeleton(∇(φh),Γ,Γs)

function move_to_sub_facet_skeleton(f::CellField,Γ::SubFacetTriangulation,Γs::SkeletonTriangulation)
  f_Γ = change_domain(f,get_triangulation(f),ReferenceDomain(),Γ,ReferenceDomain())
  Γ_trian = Triangulation(get_background_model(Γs.plus))
  f_Γ_trian = GenericCellField(get_data(f_Γ),Γ_trian,ReferenceDomain())
  f_Γs_plus = change_domain(f_Γ_trian,Γ_trian,ReferenceDomain(),Γs.plus,ReferenceDomain())
  f_Γs_minus = change_domain(f_Γ_trian,Γ_trian,ReferenceDomain(),Γs.minus,ReferenceDomain())
  return SkeletonPair(f_Γs_plus,f_Γs_minus)
end

∇φh_Γs = move_to_sub_facet_skeleton(∇(φh),Γ,Γs)

jump(∇φh_Γs)

∫(n1)Measure(Γs,2)

writevtk(
  Γs,
  "results/GammaSkel",
  cellfields=["n.plus"=>get_normal_vector(Γs).plus,"n.minus"=>get_normal_vector(Γs).minus,
  "n1"=>n1,"n2"=>n2,
  "n_∂Ω_plus"=>n_∂Ω_plus,"n_∂Ω_minus"=>n_∂Ω_minus,
  "ns"=>nˢ,
  "∇φh_Γs_plus"=>∇φh_Γs.plus,"∇φh_Γs_minus"=>∇φh_Γs.minus,
  "jump"=>jump(∇φh_Γs)]
)
writevtk(
  Ω,
  "results/Background",
  cellfields=["φh"=>∇(φh)]
)