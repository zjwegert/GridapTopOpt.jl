using GridapTopOpt

using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity, Gridap.TensorValues
import Gridap.Geometry: get_node_coordinates, collect1d
import Gridap.Arrays: evaluate, evaluate!, return_cache

include("../differentiable_trians.jl")

# Helpers
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
function find_intersect(a::Vector{VectorValue{3,T}},b::Vector{VectorValue{3,T}},aref::Vector{VectorValue{3,T}},bref::Vector{VectorValue{3,T}}) where T
  idx = indexin(Base.round.(a,sigdigits=14),Base.round.(b,sigdigits=14))
  idx_entry = findall(!isnothing,idx)
  @assert length(idx_entry) > 1
  return [aref[idx_entry],bref[idx[idx_entry]]]
end

## AffineParameterisation

"""
A Field with this form
y = y0 + 1/2*(x+1)*G, y ∈ Rᴰ, x ∈ [-1,1]
"""
struct AffineParameterisation{D,T} <:Field
  gradient::Point{D,T}
  origin::Point{D,T}
end

function Arrays.evaluate(f::AffineParameterisation,x::Number)
  G = f.gradient
  y0 = f.origin
  1/2*(x+1)*G + y0
end

function Arrays.evaluate!(cache,f::AffineParameterisation,x::Number)
  G = f.gradient
  y0 = f.origin
  1/2*(x+1)*G + y0
end

function Arrays.return_cache(f::AffineParameterisation,x::AbstractVector{<:Number})
  T = return_type(f,testitem(x))
  y = similar(x,T,size(x))
  CachedArray(y)
end

function Arrays.evaluate!(cache,f::AffineParameterisation,x::AbstractVector{<:Number})
  setsize!(cache,size(x))
  y = cache.array
  G = f.gradient
  y0 = f.origin
  for i in eachindex(x)
    xi = x[i]
    yi = 1/2*(xi+1)*G + y0
    y[i] = yi
  end
  y
end

function Gridap.gradient(h::AffineParameterisation)
  ConstantField(1/2*h.gradient)
end

function Base.zero(::Type{<:AffineParameterisation{D,T}}) where {D,T}
  gradient = Point{D,T}(tfill(zero(T),Val{D}()))
  origin = Point{D,T}(tfill(zero(T),Val{D}()))
  AffineParameterisation(gradient,origin)
end

################################################################################

order = 1
n = 14
N = 16
_model = CartesianDiscreteModel((0,1,0,1,0,1),(n,n,n))
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

# φh = interpolate(x->max(abs(x[1]-0.5),abs(x[2]-0.5),abs(x[3]-0.5))-0.25,V_φ) # Square prism
# φh = interpolate(x->((x[1]-0.5)^N+(x[2]-0.5)^N+(x[3]-0.5)^N)^(1/N)-0.25,V_φ) # Curved corner example
# φh = interpolate(x->abs(x[1]-0.5)+abs(x[2]-0.5)+abs(x[3]-0.5)-0.25-0/n/10,V_φ) # Diamond prism
# φh = interpolate(x->(1+0.25)abs(x[1]-0.5)+0abs(x[2]-0.5)+0abs(x[3]-0.5)-0.25,V_φ) # Straight lines with scaling
# φh = interpolate(x->abs(x[1]-0.5)+0abs(x[2]-0.5)+0abs(x[3]-0.5)-0.25/(1+0.25),V_φ) # Straight lines without scaling
# φh = interpolate(x->tan(-pi/3)*(x[1]-0.5)+(x[2]-0.5),V_φ) # Angled interface
φh = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-0.25,V_φ) # Sphere
# φh = interpolate(x->a*(cos(2π*x[1])*cos(2π*x[2])*cos(2π*x[3])-0.11),V_φ) # "Regular" LSF
x_φ = get_free_dof_values(φh)

if any(isapprox(0.0;atol=10^-10),x_φ)
  println("Issue with lvl set alignment")
  # idx = findall(isapprox(0.0;atol=10^-10),x_φ)
  # x_φ[idx] .+= 10eps()
end
@assert !any(isapprox(0.0;atol=10^-10),x_φ)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)
Ω_cut = Triangulation(cutgeo,PHYSICAL)
Γ = EmbeddedBoundary(cutgeo)
Γg = GhostSkeleton(cutgeo,CUT)
dΓ = Measure(Γ,2*order)

fh = interpolate(x->1,V_φ)

bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo)
writevtk(
  Ω,"results/Result",
  cellfields=["φh"=>φh,"∇φ"=>∇(φh)],
  celldata=["inoutcut"=>bgcell_to_inoutcut]
)
writevtk(
  Γg,
  "results/GhostSkel",
  cellfields=["n"=>get_normal_vector(Γg).plus]
)
writevtk(
  Γ,
  "results/Boundary",
  cellfields=["n"=>get_normal_vector(Γ)]
)

## Surface functionals
# The analytic result is for differentiating J(φ)=∫_∂Ω(φ) f dS is given by:
#   dJ(φ)(w) = lim_{t->0} (J(φₜ)-J(φ))/t = -∫_∂Ω (n⋅∇f)w/|∂ₙφ| dS - ∑_S∈P {∫_∂Ω∩S nˢ ⋅ [[fm]] w/|nˢ ⋅ ∇φ| dS}
#  The first term is easy, the second term not so much...

#  Need [[fm]]=f₁m₁ + f₂m₂, where mₖ is the co-normal for τ = 0 with
#                   mₖ = tₖˢ×n_{∂Ω(φ(0))∩S}
#  where tₖˢ is the tangent vector along the edge ∂Ω(φ(0))∩S and fₖ is
#  the limit of f on S defined by fₖ(x)=lim_{ϵ->0⁺} f(x-ϵmₖ) for x ∈ ∂Ω∩S.

# In 3D:
# Take triangulation T with edges P. Let K₁ ∈ T and K₂ ∈ T be adjacent TRI elements
#  with intersection S = K₁ ∩ K₂. Let ∂Ω denote the boundary of the linear
#  cut that passes through K₁ ∪ K₂. We define n_{∂Ω ∩ Kₖ} as the normal along
#  the cut in the respective elements K₁ and K₂. The tangent in S to ∂Ω ∩ S
#  is given by tˢₖ for k = 1 or 2. The co-normal (or bi-normal) is then defined as
#                         mₖ = tˢₖ × n_{∂Ω ∩ Kₖ}.
# Note that S ∩ ∂Ω is the skeleton of the cut cells and automatically excludes
#  the boundary as P ∩ ∂D ⊄ S.
#
# Some important facts:
# - ∂Ω ∩ S for each element S ∈ P\∂D is the `GhostSkeleton` of cut cells defined as
#  Γg = GhostSkeleton(cut cells) = background edge cut by Γ = {e ∈ ∂Ω ∩ (P\∂D)}.
#
# - We can compute the normal nˢ along each S using the normal nₖ along the ghost
#   skeleton where k denotes plus and minus values. The vector nˢ is computed via
#       nₖ×tₖˢ = (tₖˢ×nˢ)×tₖˢ = -(tₖˢ⋅nˢ)tₖˢ + (tₖˢ⋅tₖˢ)nˢ = 0tₖˢ + nˢ = +nˢ.

## Maps
# map: bg edges (P) → bg faces (T)
bg_edges_to_bg_faces = Ω.model.grid_topology.n_m_to_nface_to_mfaces[3,4]
# map: S edge → bg edges (with points ∂Ω ∩ S)
Γ_ghost_skel_edges_to_bg_edges = Γg.plus.glue.face_to_bgface
# map: S edge → bg faces (T)
Γ_ghost_skel_edges_to_bg_faces = bg_edges_to_bg_faces[Γ_ghost_skel_edges_to_bg_edges]
# map: S edge → facet (Γ₁,Γ₂ ∈ Γ)
Γ_ghost_skel_edges_to_Γ_facets = _unsafe_indexin.(Γ_ghost_skel_edges_to_bg_faces,(Γ.subfacets.facet_to_bgcell,))
# map: Γ facet → ∂Γ (end points)
Γ_facet_to_points = map(Reindex(Γ.subfacets.point_to_coords),Γ.subfacets.facet_to_points)
# map: S edge → ∂Γ₁ and ∂Γ₂ (end points)
Γ_ghost_skel_edges_to_Γ_facet_to_points = map(Reindex(Γ_facet_to_points),Γ_ghost_skel_edges_to_Γ_facets)
# map: Γ facet → ∂Γ (end points)
Γ_facet_to_rpoints = map(Reindex(Γ.subfacets.point_to_rcoords),Γ.subfacets.facet_to_points)
Γ_ghost_skel_edges_to_Γ_facet_to_rpoints = map(Reindex(Γ_facet_to_rpoints),Γ_ghost_skel_edges_to_Γ_facets)
# map: S edge → ∂Γ ∩ S (ref points) - This gives the end points of each edge in
#  the reference coordinates for the neighbouring cells
Γ_ghost_skel_edges_to_Γ_facet_Γ_ghost_intersect_ref = map((ab,abref)->find_intersect(ab[1],ab[2],abref[1],abref[2]),
  Γ_ghost_skel_edges_to_Γ_facet_to_points,Γ_ghost_skel_edges_to_Γ_facet_to_rpoints)

# This is what causes issues, the skeleton
_Γ_ghost_skel_edges_to_bg_edges = Γg.plus.glue.face_to_bgface[1] # skel face 1 -> face in bg model
_Γ_ghost_skel_edges_to_bg_faces = bg_edges_to_bg_faces[_Γ_ghost_skel_edges_to_bg_edges] # face in bg model to connected cells
Γ.subfacets.facet_to_bgcell[1]
Γ.subfacets.facet_to_bgcell[298]
Γ.subfacets.facet_to_bgcell[299]

# For visualisation on sphere with n = 2
# # E.g., for ghost face 1, the corresponding PLUS face on Γ is 1
# _tst = get_facet_normal(Γg.plus)[1](Point(0,0)) # n1
# _tst2 = get_normal_vector(Γ).cell_field[1].value # n_∂Ω
# _t = _tst × _tst2; _t /= norm(_t); _t/4 + VectorValue(0.5,0.25,0.5) # End is for visualisation
# _ns = _t × _tst; _ns /= norm(_ns); _ns/4 +  + VectorValue(0.5,0.25,0.5)
# _m = _t × _tst2; _m /= norm(_m)#; _m/4 + VectorValue(0.5,0.25,0.5)
# # E.g., for ghost face 1, the corresponding PLUS face on Γ is 25
# _tst = get_facet_normal(Γg.minus)[1](Point(0,0)) # n2
# _tst2 = get_normal_vector(Γ).cell_field[25].value # n_∂Ω
# _t = _tst × _tst2; _t /= norm(_t); _t/4 + VectorValue(0.5,0.25,0.5) # End is for visualisation
# _ns = _t × _tst; _ns /= norm(_ns); _ns/4 +  + VectorValue(0.5,0.25,0.5)
# _m = _t × _tst2; _m /= norm(_m)#; _m/4 + VectorValue(0.5,0.25,0.5)

## Get normal nₖ to skeleton and normal n_∂Ω to interface. Take (k=1)≡Γg⁺ and (k=2)≡Γg⁻.
n1 = map(x->evaluate(x,Point(0,0)),get_facet_normal(Γg.plus))
n2 = map(x->evaluate(x,Point(0,0)),get_facet_normal(Γg.minus))
n_∂Ω = map(x->evaluate(x,Point(0,0)),get_data(get_normal_vector(Γ)))

## Compute n_{∂Ω∩Kₖ} for each face in the skeleton
n_∂Ω1 = n_∂Ω[getindex.(Γ_ghost_skel_edges_to_Γ_facets,1)]
n_∂Ω2 = n_∂Ω[getindex.(Γ_ghost_skel_edges_to_Γ_facets,2)]

## Compute tangents tₖˢ
t1 = n1 .× n_∂Ω1; @. t1 /= norm(t1)
t2 = n2 .× n_∂Ω2; @. t2 /= norm(t2)

## Compute nˢ
ns = t1 .× n1; @. ns /= norm(ns)
# ns_2 = t2 .× n2; @. ns_2 /= norm(ns_2); @assert ns ≈ ns_2

## Compute bi-normal mₖ
m1 = t1 .× n_∂Ω1; @. m1 /= norm(m1)
m2 = t2 .× n_∂Ω2; @. m1 /= norm(m1)

# Parameterise ∂Ω ∩ Kₖ
function affine_params_from_pts_plus(x)
  xplus = x[1];
  AffineParameterisation(xplus[2]-xplus[1],xplus[1])
end
function affine_params_from_pts_minus(x)
  xminus = x[2];
  AffineParameterisation(xminus[2]-xminus[1],xminus[1])
end

γ_plus = map(affine_params_from_pts_plus,Γ_ghost_skel_edges_to_Γ_facet_Γ_ghost_intersect_ref)
γ_minus = map(affine_params_from_pts_minus,Γ_ghost_skel_edges_to_Γ_facet_Γ_ghost_intersect_ref)

# Gaussian quadrature
ti = [-1/sqrt(3),1/sqrt(3)]
wi = [1,1];
xi_plus = map(γ -> evaluate(γ,ti),γ_plus)
xi_minus = map(γ -> evaluate(γ,ti),γ_minus)

## Construct map: S edge → ∇φh on adjacent faces and map: S edge → f on adjacent faces
∇φ_plus = map(Reindex(get_data(∇(φh))),getindex.(Γ_ghost_skel_edges_to_Γ_facets,1))
∇φ_minus = map(Reindex(get_data(∇(φh))),getindex.(Γ_ghost_skel_edges_to_Γ_facets,2))
f_plus = map(Reindex(get_data(fh)),getindex.(Γ_ghost_skel_edges_to_Γ_facets,1))
f_minus = map(Reindex(get_data(fh)),getindex.(Γ_ghost_skel_edges_to_Γ_facets,2))

# Construct D_S = nˢ ⋅ ([[fm]]/|nˢ ⋅ ∇φ|) over quadrature points
# Note that ∇(φ) is continuous in direction nˢ so plus and minus are equal
abs_∂_nˢ_φ = map((∇φ,xi,ns)->abs.(∇φ(xi) .⋅ (ns,)),∇φ_plus,xi_plus,ns)
m1f1 = map((f,m,x)->Broadcasting(*)(f(x),m),f_plus,m1,xi_plus)
m2f2 = map((f,m,x)->Broadcasting(*)(f(x),m),f_minus,m2,xi_minus)
D_S = map((nˢ,m₁f₁,m₂f₂,∂nˢφ)->Broadcasting(/)(Broadcasting(⋅)(nˢ,m₁f₁+m₂f₂),∂nˢφ),ns,m1f1,m2f2,abs_∂_nˢ_φ)

# Construct DomainContributions of ∑_S∈P∫{D_S*w}dγ over Ω
dv = get_fe_basis(V_φ)
dv_data = get_data(dv)
nbasis = length(first(dv_data))
contrib = [zeros(nbasis) for _ ∈ eachindex(dv_data)]
xi_both = map((a,b)->[a,b],xi_plus,xi_minus)
for i ∈ Ω.tface_to_mface
  # Check if bgcell is cut and get relevant indices of reference coords
  dv_cell_data = get_data(dv)[i]
  Γg_contrib_idxs_loc = map(v->i .== v,Γ_ghost_skel_edges_to_bg_faces)
  Γg_contrib_idxs = map(any,Γg_contrib_idxs_loc)
  if ~any(Γg_contrib_idxs)
    continue
  end
  ## Data on each edge of bg cell
  data_contribs = D_S[Γg_contrib_idxs]
  ## Quadrature points on each edge of bg cell
  bgcell_quad_rpoints = reduce(vcat,getindex.(xi_both[Γg_contrib_idxs],Γg_contrib_idxs_loc[Γg_contrib_idxs]))
  ## Value of each basis functions at quadrature points on each edge of bg cell
  basis_vals = map(v->[evaluate(dv_cell_data,v[j]) for j ∈ eachindex(v)],bgcell_quad_rpoints)
  ## Compute f_i*w_i in ∫(f)dt = ∑_i f(x_i)w_i over [-1,1]
  f_i = map(.*,basis_vals,data_contribs)
  f_i_w_i = map(fw -> fw.*wi,f_i)
  ## Add cell contribution
  contrib[i] .= sum(sum(f_i_w_i))
end
dom_contrib_t2 = DomainContribution()
Gridap.CellData.add_contribution!(dom_contrib_t2,Ω,contrib,-)

# Construct DomainContributions of -∫_∂Ω (n⋅∇f)w/|∂ₙφ| dS
_n = get_normal_vector(Γ)
dom_contrib_t1 = ∫(-(_n⋅∇(fh))*dv/(norm ∘ (∇(φh))))dΓ

# Construct DomainContribution and assemble
dom_contrib = dom_contrib_t1 + dom_contrib_t2
dj2_exp_vec = assemble_vector(dom_contrib,V_φ)

# Visualisation
bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo)
writevtk(
  Ω,"results/Result",
  cellfields=["φh"=>φh,"∇φ"=>∇(φh),"dj2"=>FEFunction(V_φ,dj2_exp_vec)],
  celldata=["inoutcut"=>bgcell_to_inoutcut]
)

# Visualise jump quanities on Γg
writevtk(
  Γg,
  "results/GhostSkel",
)

# Boundary and normal vector
writevtk(
  Γ,
  "results/Boundary",
  cellfields=["n"=>get_normal_vector(Γ)]
)