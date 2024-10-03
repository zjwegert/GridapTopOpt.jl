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
    Γ_pts = unique(round.(reduce(vcat,Γ_facet_to_points);sigdigits=15))
    Γ_facet_to_pts = convert(Vector{Vector{Int32}},map(v->indexin(round.(v;sigdigits=15),Γ_pts),Γ_facet_to_points))

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

function move_to_sub_facet_skeleton(f::CellField,Γ::SubFacetTriangulation,Γs::SkeletonTriangulation)
  f_Γ = change_domain(f,get_triangulation(f),ReferenceDomain(),Γ,ReferenceDomain())
  Γ_trian = Triangulation(get_background_model(Γs.plus))
  f_Γ_trian = GenericCellField(get_data(f_Γ),Γ_trian,ReferenceDomain())
  f_Γs_plus = change_domain(f_Γ_trian,Γ_trian,ReferenceDomain(),Γs.plus,ReferenceDomain())
  f_Γs_minus = change_domain(f_Γ_trian,Γ_trian,ReferenceDomain(),Γs.minus,ReferenceDomain())
  return SkeletonPair(f_Γs_plus,f_Γs_minus)
end

function belongs_to_edge(q::Point{2,T},p::Vector{Point{2,T}}) where T
  p1, p2 = p
  tol = 10eps()
  @assert abs(p2[1] - p1[1]) > tol || abs(p2[2] - p1[2]) > tol

  if abs(p2[1] - p1[1]) > tol
    t = (q[1]-p1[1])/((p2[1]-p1[1]))
    0 < t < 1 && abs(q[2] - (p1[2] + t*(p2[2]-p1[2]))) < tol
  else
    t = (q[2]-p1[2])/((p2[2]-p1[2]))
    0 < t < 1 && abs(q[1] - (p1[1] + t*(p2[1]-p1[1]))) < tol
  end
end

function Γg_to_Γs_perm(Γs::SkeletonTriangulation{0,2},Γg::SkeletonTriangulation{1,2})
  grid_top = get_grid_topology(get_background_model(Γg.plus))
  # ghost skeleton edges to bg edges
  Γg_to_bg_edge = Γg.plus.glue.face_to_bgface
  # bg edges to bg vertices
  bg_edges_to_bg_vertices = grid_top.n_m_to_nface_to_mfaces[2,1]
  # points on ghost skeleton
  Γg_to_bg_vertices = bg_edges_to_bg_vertices[Γg_to_bg_edge]
  # points attached to ghost skeleton 
  Γ_ghost_skel_edges_to_bg_vertex_pts = map(Reindex(grid_top.vertex_coordinates),Γg_to_bg_vertices)
  # skeleton to vertices
  Γs_to_vertex = Γs.plus.glue.face_to_bgface
  idxperm = zeros(Int32,length(Γs_to_vertex))
  for (i,p) ∈ enumerate(Γ_pts[Γs_to_vertex])
    for (j,q) ∈ enumerate(Γ_ghost_skel_edges_to_bg_vertex_pts)
      if belongs_to_edge(p,q)
        idxperm[i] = j
        continue
      end
    end
  end
  idxperm
end

function orient(a::VectorValue{2,T},b::VectorValue{2,T}) where T
  if a ⋅ b <= 0
    -a
  else
    a
  end
end

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
dΓs = Measure(Γs,2)
fh = interpolate(x->1,V_φ)

_2d_cross(n) = VectorValue(n[2],-n[1]);

Γg_to_Γs = Γg_to_Γs_perm(Γs,Γg)
n1 = CellField(evaluate(get_facet_normal(Γg.plus),Point(0))[Γg_to_Γs],Γs.plus)
n2 = CellField(evaluate(get_facet_normal(Γg.minus),Point(0))[Γg_to_Γs],Γs.minus)
# n1 = GenericCellField(get_facet_normal(Γg.plus),Γs.plus,ReferenceDomain()) # This breaks
# n2 = GenericCellField(get_facet_normal(Γg.minus),Γs.minus,ReferenceDomain())

# n_∂Ω_plus = GenericCellField(get_facet_normal(get_triangulation(Γs.plus)),Γs,ReferenceDomain())
# n_∂Ω_minus = GenericCellField(get_facet_normal(get_triangulation(Γs.minus)),Γs,ReferenceDomain())
_Γ_trian = Triangulation(get_background_model(Γs))
n_∂Ω_plus = change_domain(get_normal_vector(_Γ_trian).plus,_Γ_trian,ReferenceDomain(),Γs.plus,ReferenceDomain())
n_∂Ω_minus = change_domain(get_normal_vector(_Γ_trian).minus,_Γ_trian,ReferenceDomain(),Γs.minus,ReferenceDomain())

nˢ = Operation(v->(-_2d_cross(v)))(n1)
# nˢ = Operation(orient)(nˢ,n_∂Ω_plus)

m1 = Operation(v->(_2d_cross(v)))(n_∂Ω_plus)
m2 = Operation(v->(-_2d_cross(v)))(n_∂Ω_minus)
m = SkeletonPair(m1,m2)

∇φh_Γs = move_to_sub_facet_skeleton(∇(φh),Γ,Γs)
fh_Γs = move_to_sub_facet_skeleton(fh,Γ,Γs)
w_Γs = move_to_sub_facet_skeleton(get_fe_basis(V_φ),Γ,Γs)
jump_fm = fh_Γs.⁺*m.⁺ + fh_Γs.⁻*m.⁻

# I'm not sure about the basis w_Γs as this is a SkeletonPair...
dJ2 = ∫((nˢ ⋅ (jump_fm*jump(w_Γs)/(abs ∘ (nˢ ⋅ ∇φh_Γs.⁺)))))dΓs
get_array(dJ2)

writevtk(
  Γs,
  "results/GammaSkel",
  cellfields=["n.plus"=>get_normal_vector(Γs).plus,"n.minus"=>get_normal_vector(Γs).minus,
  "n1"=>n1,"n2"=>n2,
  "n_∂Ω_plus"=>n_∂Ω_plus,"n_∂Ω_minus"=>n_∂Ω_minus,
  "ns"=>nˢ,
  "∇φh_Γs_plus"=>∇φh_Γs.plus,"∇φh_Γs_minus"=>∇φh_Γs.minus,
  "m.minus"=>m.minus,"m.plus"=>m.plus,
  "data"=>abs ∘ (∇φh_Γs.plus ⋅ nˢ)]
)
  writevtk(
  Ω,
  "results/Background",
  cellfields=["φh"=>∇(φh)]
)
writevtk(
  Γ,
  "results/Boundary"
)
writevtk(
  Γg.plus.trian,
  "results/GhostSkel"
)



# subfacets.facet_to_points
# Γg.plus.trian.tface_to_mface
bg_edges_to_bg_faces = Ω.model.grid_topology.n_m_to_nface_to_mfaces[2,3]
Γ_ghost_skel_edges_to_bg_edges = bg_edges_to_bg_faces[Γg.plus.glue.face_to_bgface]
subfacets.facet_to_bgcell
Γ_ghost_skel_edges_to_bg_edges
Γ_ghost_skel_edges_to_subfacets = map(v->indexin(v,subfacets.facet_to_bgcell),Γ_ghost_skel_edges_to_bg_edges)

subfacets_to_Γ_ghost_skel_edges = map(i->findall(v->i ∈ v, Γ_ghost_skel_edges_to_subfacets),eachindex(subfacets.facet_to_bgcell))
findall(v->isone(length(v)),subfacets_to_Γ_ghost_skel_edges)

bg_edges_to_bg_faces = Ω.model.grid_topology.n_m_to_nface_to_mfaces[2,1]
Γ_ghost_skel_edges_to_bg_vertices = bg_edges_to_bg_faces[Γg.plus.glue.face_to_bgface]
Γ_ghost_skel_edges_to_bg_vertex_pts = getindex.((Ω.model.grid_topology.vertex_coordinates,),Γ_ghost_skel_edges_to_bg_vertices)

# map(v->v[2]-v[1],Γ_ghost_skel_edges_to_bg_vertex_pts)

# Γ_ghost_skel_edges_to_bg_faces[3]
# Γ.subfacets.facet_to_bgcell
# map(Reindex(Γ_facet_to_pts))

# getindex.((Γ_pts,),Γ_facet_to_pts)

subfacets = Γ.subfacets
Γ_facet_to_points = map(Reindex(subfacets.point_to_coords),subfacets.facet_to_points)
# Γ_facet_to_points = sort.(Γ_facet_to_points)

Γ_pts = unique(round.(reduce(vcat,Γ_facet_to_points);sigdigits=15))

Γ_facet_to_pts = convert(Vector{Vector{Int32}},map(v->indexin(round.(v;sigdigits=15),Γ_pts),Γ_facet_to_points))

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


# We need a map from points on ghost skeleton to points on skeleton as
#  there don't have the same ordering
Γ_pts[Γs.plus.glue.face_to_bgface] # points on skeleton
Γ_ghost_skel_edges_to_bg_vertex_pts # points attached to ghost skeleton 

function belongs_to_edge(q::Point{2,T},p::Vector{Point{2,T}}) where T
  p1, p2 = p
  tol = 10eps()
  @assert abs(p2[1] - p1[1]) > tol || abs(p2[2] - p1[2]) > tol

  if abs(p2[1] - p1[1]) > tol
    t = (q[1]-p1[1])/((p2[1]-p1[1]))
    0 < t < 1 && abs(q[2] - (p1[2] + t*(p2[2]-p1[2]))) < tol
  else
    t = (q[2]-p1[2])/((p2[2]-p1[2]))
    0 < t < 1 && abs(q[1] - (p1[1] + t*(p2[1]-p1[1]))) < tol
  end
end

idxperm = zeros(Int32,length(Γs.plus.glue.face_to_bgface))
for (i,p) ∈ enumerate(Γ_pts[Γs.plus.glue.face_to_bgface])
  for (j,q) ∈ enumerate(Γ_ghost_skel_edges_to_bg_vertex_pts)
    if belongs_to_edge(p,q)
      idxperm[i] = j
      continue
    end
  end
end
idxperm



# idxperm = findall.(map(q->belongs_to_edge.(q,Γ_ghost_skel_edges_to_bg_vertex_pts),Γ_pts[Γs.plus.glue.face_to_bgface]))
# @assert all(length.(idxperm) .== 1)
# idxperm = first.(idxperm)


writevtk(
  Γs,
  "results/GammaSkel",
  cellfields=["n.plus"=>get_normal_vector(Γs).plus,"n.minus"=>get_normal_vector(Γs).minus]
)