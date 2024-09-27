using GridapTopOpt

using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity
import Gridap.Geometry: get_node_coordinates, collect1d

include("../differentiable_trians.jl")

# Helpers
function orient(a::VectorValue{2,T},b::VectorValue{2,T}) where T
  if a ⋅ b <= 0
    -a
  else
    a
  end
end
# Instead of an array of with eltype Union{inds,nothing}, this puts a negative index in place of nothing.
function _unsafe_indexin(a, b::AbstractArray)
  inds = keys(b)
  bdict = Dict{eltype(b),eltype(inds)}()
  for (val, ind) in zip(b, inds)
      get!(bdict, val, ind)
  end
  return eltype(inds)[
      get(bdict, i, -one(eltype(inds))) for i in a
  ]
end
Base.round(a::VectorValue{D,T};kwargs...) where {D,T} = VectorValue{D,T}(round.(a.data;kwargs...))
uniqueidx(v) = unique(i -> Base.round(v[i];sigdigits=14), eachindex(v))
nonuniqueidx(v) = findfirst(i->~(i ∈ uniqueidx(v)),eachindex(v))
function find_intersect(a::Vector{VectorValue{D,T}},b::Vector{VectorValue{D,T}},aref::Vector{VectorValue{D,T}},bref::Vector{VectorValue{D,T}}) where {D,T}
  idx = indexin(Base.round.(a,sigdigits=14),Base.round.(b,sigdigits=14))
  idx_entry = findfirst(!isnothing,idx)
  return [aref[idx_entry],bref[idx[idx_entry]]]
end
# TODO: Avoid using these functions? I suspect some of this could be replaced with in-built Gridap
#   functionality, or Julia functions that I've forgotten!

######

order = 1
n = 101
N = 16
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

# φh = interpolate(x->max(abs(x[1]-0.5),abs(x[2]-0.5))-0.25,V_φ) # Square # NOTE: currently broken for this example (find_intersect), all others work as expected
# φh = interpolate(x->((x[1]-0.5)^N+(x[2]-0.5)^N)^(1/N)-0.25,V_φ) # Curved corner example
# φh = interpolate(x->abs(x[1]-0.5)+abs(x[2]-0.5)-0.25-0/n/10,V_φ) # Diamond
# φh = interpolate(x->(1+0.25)abs(x[1]-0.5)+0abs(x[2]-0.5)-0.25,V_φ) # Straight lines with scaling
# φh = interpolate(x->abs(x[1]-0.5)+0abs(x[2]-0.5)-0.25/(1+0.25),V_φ) # Straight lines without scaling
# φh = interpolate(x->tan(-pi/3)*(x[1]-0.5)+(x[2]-0.5),V_φ) # Angled interface
# φh = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-0.25,V_φ) # Circle
φh = interpolate(x->cos(2π*x[1])*cos(2π*x[2])-0.11,V_φ) # "Regular" LSF
x_φ = get_free_dof_values(φh)

if any(isapprox(0.0;atol=10^-10),x_φ)
  println("Issue with lvl set alignment")
  idx = findall(isapprox(0.0;atol=10^-10),x_φ)
  x_φ[idx] .+= 10eps()
end
any(x -> x < 0,x_φ)
any(x -> x > 0,x_φ)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)
Ω_cut = Triangulation(cutgeo,PHYSICAL)
Γ = EmbeddedBoundary(cutgeo)
Γg = GhostSkeleton(cutgeo,CUT)
dΓ = Measure(Γ,2*order)

fh = interpolate(x->1,V_φ)

## Surface functionals
# The analytic result is for differentiating J(φ)=∫_∂Ω(φ) f dS is given by:
#   dJ(φ)(w) = lim_{t->0} (J(φₜ)-J(φ))/t = -∫_∂Ω (n⋅∇f)w/|∂ₙφ| dS - ∑_S∈P {∫_∂Ω∩S nˢ ⋅ [[fm]] w/|nˢ ⋅ ∇φ| dS}
#  The first term is easy, the second term not so much...

#  Need [[fm]]=f₁m₁ + f₂m₂, where mₖ is the co-normal for τ = 0 with
#                   mₖ = tₖˢ×n_{∂Ω(φ(0))∩S}
#  where tₖˢ is the tangent vector along the edge ∂Ω(φ(0))∩S and fₖ is
#  the limit of f on S defined by fₖ(x)=lim_{ϵ->0⁺} f(x-ϵmₖ) for x ∈ ∂Ω∩S.

# In 2D:
# Take triangulation T with edges P. Let K₁ ∈ T and K₂ ∈ T be adjacent TRI elements
#  with intersection S = K₁ ∩ K₂. Let ∂Ω denote the boundary of the linear
#  cut that passes through K₁ ∪ K₂. We define n_{∂Ω ∩ Kₖ} as the normal along
#  the cut in the respective elements K₁ and K₂. We choose the tangent to S
#  tˢₖ to lie out of the page, namely tˢ₁ = -e₃ and t²₂ = e₃. The co-normal
#  (or bi-normal) is then defined as
#                         mₖ = tˢₖ × n_{∂Ω ∩ Kₖ}.
# Note that S ∩ ∂Ω is the skeleton of the cut cells and automatically excludes
#  the boundary as P ∩ ∂D ⊄ S.
#
# Some important facts:
# - In 2D, the derivative above simplifies to
#     dJ(φ)(w) = -∫_∂Ω (n⋅∇f)w/|∂ₙφ| dS - ∑_S∈P {nˢ ⋅ ([[fm]] w/|nˢ ⋅ ∇φ|)_∂Ω∩S},
#   where the second term is a point-wise evaluation along the ghost skeleton Γg.
#
# - ∂Ω ∩ S for each element S ∈ P\∂D is the `GhostSkeleton` of cut cells defined as
#  Γg = GhostSkeleton(cut cells) = background edge cut by Γ = {e ∈ ∂Ω ∩ (P\∂D)}.
#
# - We can compute the normal nˢ along each S using the normal nₖ along the ghost
#   skeleton where k denotes plus and minus values. The vector nˢ is computed via
#       nₖ×tₖˢ = (tₖˢ×nˢ)×tₖˢ = -(tₖˢ⋅nˢ)tₖˢ + (tₖˢ⋅tₖˢ)nˢ = 0tₖˢ + nˢ = +nˢ.

# map: bg edges (P) → bg faces (T)
bg_edges_to_bg_faces = Ω.model.grid_topology.n_m_to_nface_to_mfaces[2,3]
# map: S edge → bg edges (with points ∂Ω ∩ S)
Γ_ghost_skel_edges_to_bg_edges = Γg.plus.glue.face_to_bgface

## Construct map: S edge → facet (Γ₁,Γ₂ ∈ Γ) and map: S edge → facet normal (n_Γ₁,n_Γ₂)
# function: compute f(v) = e₃ × v where v = [v₁,v₂,0].
m(n) = VectorValue(n[2],-n[1]);
# map (plus): S edge → bg faces (T)
Γ_ghost_skel_edges_to_bg_faces = bg_edges_to_bg_faces[Γ_ghost_skel_edges_to_bg_edges]
# map: S edge → facet (Γ₁,Γ₂ ∈ Γ)
Γ_ghost_skel_edges_to_Γ_facets = _unsafe_indexin.(Γ_ghost_skel_edges_to_bg_faces,(Γ.subfacets.facet_to_bgcell,))
# map: S edge → facet normal (n_Γ₁,n_Γ₂)
Γ_ghost_skel_edges_to_Γ_normals = map(Reindex(Γ.subfacets.facet_to_normal),Γ_ghost_skel_edges_to_Γ_facets)
# map: S edge → Γ conormals (m_Γ₁,m_Γ₂)
Γ_ghost_skel_edges_to_Γ_conormals = map(v->[m(v[1]),-m(v[2])],Γ_ghost_skel_edges_to_Γ_normals)

## Construct map: S edge → nˢ (vector parallel to S, orthogonal to S normals, outward facing from ∂Ω)
S_normal = get_facet_normal(Γg.plus)
nˢ = lazy_map(Operation(v->(-m(v))),S_normal)
nˢ = map(v->v(Point(0)),nˢ)
# Reorient nˢ to point outwards from ∂Ω. TODO @Jordi: This has been disabled at the moment.
#  When re-enabled, results do not look correct (e.g., lack symmetry and weird jumps) I'm not
#  sure why nˢ doesn't have to be oriented here?
# nˢ = map((u,v)->orient(u,v[1]),nˢ,Γ_ghost_skel_edges_to_Γ_normals)

## Construct map: Γ facet → ∂Γ ∩ S (points)
# map: Γ facet → ∂Γ (end points)
Γ_facet_to_points = map(Reindex(Γ.subfacets.point_to_coords),Γ.subfacets.facet_to_points)
# map: S edge → ∂Γ₁ and ∂Γ₂ (end points)
Γ_ghost_skel_edges_to_Γ_facet_to_points = map(Reindex(Γ_facet_to_points),Γ_ghost_skel_edges_to_Γ_facets)
# # map: S edge → ∂Γ ∩ S (points) # Not really needed unless using PhysicalDomain
# flat_Γ_ghost_skel_edges_to_Γ_facet_to_points = map(x->reduce(vcat, x),Γ_ghost_skel_edges_to_Γ_facet_to_points)
# idxs = map(nonuniqueidx,flat_Γ_ghost_skel_edges_to_Γ_facet_to_points)
# Γ_ghost_skel_edges_to_Γ_facet_Γ_ghost_intersect = getindex.(flat_Γ_ghost_skel_edges_to_Γ_facet_to_points,idxs)

## Construct map: Γ facet → ∂Γ ∩ S (ref points)
# map: Γ facet → ∂Γ (end points)
Γ_facet_to_rpoints = map(Reindex(Γ.subfacets.point_to_rcoords),Γ.subfacets.facet_to_points)
Γ_ghost_skel_edges_to_Γ_facet_to_rpoints = map(Reindex(Γ_facet_to_rpoints),Γ_ghost_skel_edges_to_Γ_facets)
 # map: S edge → ∂Γ ∩ S (ref points)
Γ_ghost_skel_edges_to_Γ_facet_Γ_ghost_intersect_ref = map((ab,abref)->find_intersect(ab[1],ab[2],abref[1],abref[2]),
  Γ_ghost_skel_edges_to_Γ_facet_to_points,Γ_ghost_skel_edges_to_Γ_facet_to_rpoints)

# debugging
# for i ∈ eachindex(Γ_ghost_skel_edges_to_Γ_facet_to_points)
#   @show i
#   find_intersect(Γ_ghost_skel_edges_to_Γ_facet_to_points[i][1],Γ_ghost_skel_edges_to_Γ_facet_to_points[i][2],
#     Γ_ghost_skel_edges_to_Γ_facet_to_rpoints[i][1],Γ_ghost_skel_edges_to_Γ_facet_to_rpoints[i][2])
# end

## Construct map: S edge → ∇φh on adjacent faces and map: S edge → f on adjacent faces
∇φh_data = ∇(φh)
# map: S edge → ∇φh on adjacent faces (ref coords)
Γ_ghost_skel_edges_to_∇φh_field = map(Reindex(get_data(∇φh_data)),Γ_ghost_skel_edges_to_bg_faces)
∇φh_values_at_intersect = map(i->map.(Γ_ghost_skel_edges_to_∇φh_field[i],Γ_ghost_skel_edges_to_Γ_facet_Γ_ghost_intersect_ref[i]),
  eachindex(Γ_ghost_skel_edges_to_Γ_facet_Γ_ghost_intersect_ref))
# map: S edge → f on adjacent faces (ref coords)
Γ_ghost_skel_edges_to_fh_field = map(Reindex(get_data(fh)),Γ_ghost_skel_edges_to_bg_faces)
f_values_at_intersect = map(i->map.(Γ_ghost_skel_edges_to_fh_field[i],Γ_ghost_skel_edges_to_Γ_facet_Γ_ghost_intersect_ref[i]),
  eachindex(Γ_ghost_skel_edges_to_Γ_facet_Γ_ghost_intersect_ref))

# Construct D_S = nˢ ⋅ ([[fm]]/|nˢ ⋅ ∇φ|)_∂Ω∩S
# Note that ∇(φ) is continuous in direction nˢ so plus and minus are equal
abs_∂_nˢ_φ = map((u,v)->abs.(dot.(u,(v,))),∇φh_values_at_intersect,nˢ)
abs_∂_nˢ_φ⁺ = getindex.(abs_∂_nˢ_φ,1)
jump_m_f = Broadcasting(x->(reduce(+,x)))(Broadcasting(.*)(f_values_at_intersect,Γ_ghost_skel_edges_to_Γ_conormals))
D_S = @. nˢ ⋅ (jump_m_f/abs_∂_nˢ_φ⁺)

# Construct DomainContributions of ∑_S∈P{D_S*w} over Ω. NOTE: This is super hacky
dv = get_fe_basis(V_φ)
dv_data = get_data(dv)
nbasis = length(first(dv_data))
contrib = [zeros(nbasis) for _ ∈ eachindex(dv_data)]
for i ∈ Ω.tface_to_mface
  # Check if bgcell is cut and get relevant indices of reference coords
  dv_cell_data = get_data(dv)[i]
  Γg_contrib_idxs_loc = map(v->i .== v,Γ_ghost_skel_edges_to_bg_faces)
  Γg_contrib_idxs = map(any,Γg_contrib_idxs_loc)
  if ~any(Γg_contrib_idxs)
    continue
  end
  bgcell_rpoints = reduce(vcat,getindex.(Γ_ghost_skel_edges_to_Γ_facet_Γ_ghost_intersect_ref[Γg_contrib_idxs],Γg_contrib_idxs_loc[Γg_contrib_idxs]))
  data_contribs = D_S[Γg_contrib_idxs]
  # Evaluate basis for cell i at cut points
  basis_vals = map(v->evaluate(dv_cell_data,v),bgcell_rpoints)
  contrib[i] .= sum(map(*,data_contribs,basis_vals))
end
dom_contrib_t2 = DomainContribution()
Gridap.CellData.add_contribution!(dom_contrib_t2,Ω,contrib,-)

# Construct DomainContributions of -∫_∂Ω (n⋅∇f)w/|∂ₙφ| dS
_n = get_normal_vector(Γ)
dom_contrib_t1 = ∫(-(_n⋅∇(fh))*dv/(norm ∘ (∇(φh))))dΓ

# Construct DomainContribution and assemble
dom_contrib = dom_contrib_t1 + dom_contrib_t2
djΓ_exp_vec = assemble_vector(dom_contrib,V_φ)

### Visualisation and testing
# AD
diff_Γ = DifferentiableTriangulation(Γ)
dΓ2 = Measure(diff_Γ,2*order)
jΓ(φ) = ∫(1)dΓ2
djΓ = gradient(jΓ,φh)
djΓ_contrib = DomainContribution()
Gridap.CellData.add_contribution!(djΓ_contrib,diff_Γ.trian,get_array(djΓ),+)
djΓ_vec_out = assemble_vector(djΓ_contrib,V_φ)

bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo)
writevtk(
  Ω,"results/Result",
  cellfields=["φh"=>φh,"∇φ"=>∇(φh),"djΓ_analytic"=>FEFunction(V_φ,djΓ_exp_vec),"djΓ"=>FEFunction(V_φ,djΓ_vec_out)],
  celldata=["inoutcut"=>bgcell_to_inoutcut]
)

# Visualise jump quanities on Γg
abs_∂_nˢ_φh = CellField(getindex.(abs_∂_nˢ_φ,1),Γg)
fh_∂ΩcapS = SkeletonCellFieldPair(CellField(getindex.(f_values_at_intersect,1),Γg),CellField(getindex.(f_values_at_intersect,2),Γg))
mh = SkeletonCellFieldPair(CellField(getindex.(Γ_ghost_skel_edges_to_Γ_conormals,1),Γg),CellField(getindex.(Γ_ghost_skel_edges_to_Γ_conormals,2),Γg))
nˢh = CellField(nˢ,Γg)
datah = CellField(@.(nˢ ⋅ (jump_m_f/abs_∂_nˢ_φ⁺)),Γg)
writevtk(
  Γg,
  "results/GhostSkel",
  cellfields=["data"=>datah,"ns"=>nˢh,"abs_∂_nˢ_φh"=>abs_∂_nˢ_φh,"fh_∂ΩcapS"=>fh_∂ΩcapS,"mh_plus"=>mh.plus,"mh_minus"=>mh.minus]
)

# Boundary and normal vector
writevtk(
  # Ω_cut.a,
  Γ,
  "results/Boundary",
  cellfields=["n"=>get_normal_vector(Γ)]
)
