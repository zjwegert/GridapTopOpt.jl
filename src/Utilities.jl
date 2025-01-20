"""
    struct SmoothErsatzMaterialInterpolation{M<:Vector{<:Number},N<:Vector{<:Number}}

A wrapper holding parameters and methods for interpolating an
integrand across a single boundary ``\\partial\\Omega``.

E.g., ``\\int f~\\mathrm{d}\\Omega = \\int I(\\varphi)f~\\mathrm{d}D`` where ``\\Omega\\subset D`` is described by a level-set
function ``\\varphi`` and ``I`` is an indicator function.

# Properties

- `η::M`: the interpolation or smoothing radius across ∂Ω
- `ϵ::M`: the ersatz material density
- `H`: a smoothed Heaviside function
- `DH`: the derivative of `H`
- `I`: an indicator function
- `ρ`: a function describing the volume density of ``\\Omega``
  (e.g., ``\\mathrm{Vol}(\\Omega) = \\int \\rho(\\varphi))~\\mathrm{d}D)``

# Note
- We store η and ϵ as length-one vectors so that updating these values propagates through H, DH, etc.
- To update η and/or ϵ in an instance `m`, take `m.η .= <VALUE>`.
- A conviencence constructor is provided to create an instance given `η<:Number` and `ϵ<:Number`.
"""
struct SmoothErsatzMaterialInterpolation{M<:Vector{<:Number},N<:Vector{<:Number}}
  η::M
  ϵ::N
  H
  DH
  I
  ρ
end

function SmoothErsatzMaterialInterpolation(;η::M,ϵ::N = 10^-3) where {M<:Number,N<:Number}
  _η = [η,]
  _ϵ = [ϵ,]
  H(x) = H_η(x,first(_η))
  DH(x) = DH_η(x,first(_η))
  I(φ) = (1 - H(φ)) + first(_ϵ)*H(φ)
  ρ(φ) = 1 - H(φ)

  return SmoothErsatzMaterialInterpolation{Vector{M},Vector{N}}(_η,_ϵ,H,DH,I,ρ)
end

function H_η(t,η)
  M = typeof(η*t)
  if t<-η
    return zero(M)
  elseif abs(t)<=η
    return 1/2*(1+t/η+1/pi*sin(pi*t/η))
  elseif t>η
    return one(M)
  end
end

function DH_η(t,η)
  M = typeof(η*t)
  if t<-η
    return zero(M)
  elseif abs(t)<=η
    return 1/2/η*(1+cos(pi*t/η))
  elseif t>η
    return zero(M)
  end
end

## Helpers

"""
    update_labels!(e::Int,model,f_Γ::Function,name::String)

Given a tag number `e`, a `DiscreteModel` model, an indicator function `f_Γ`,
and a string `name`, label the corresponding vertices, edges, and faces
as `name`.

Note: `f_Γ` must recieve a Vector and return a Boolean depending on whether it indicates Γ
"""
function update_labels!(e::Integer,model::DiscreteModel,f_Γ::Function,name::String)
  mask = mark_nodes(f_Γ,model)
  _update_labels_locally!(e,model,mask,name)
  nothing
end

function update_labels!(e::Integer,model::DistributedDiscreteModel,f_Γ::Function,name::String)
  mask = mark_nodes(f_Γ,model)
  cell_to_entity = map(local_views(model),local_views(mask)) do model,mask
    _update_labels_locally!(e,model,mask,name)
  end
  cell_gids=get_cell_gids(model)
  cache=GridapDistributed.fetch_vector_ghost_values_cache(cell_to_entity,partition(cell_gids))
  GridapDistributed.fetch_vector_ghost_values!(cell_to_entity,cache)
  nothing
end

function _update_labels_locally!(e,model::DiscreteModel{2},mask,name)
  topo   = get_grid_topology(model)
  labels = get_face_labeling(model)
  cell_to_entity = labels.d_to_dface_to_entity[end]
  entity = maximum(cell_to_entity) + e
  # Vertices
  vtxs_Γ = findall(mask)
  vtx_edge_connectivity = Array(get_faces(topo,0,1)[vtxs_Γ])
  # Edges
  edge_entries = [findall(x->any(x .∈  vtx_edge_connectivity[1:end.!=j]),
    vtx_edge_connectivity[j]) for j = 1:length(vtx_edge_connectivity)]
  edge_Γ = unique(reduce(vcat,getindex.(vtx_edge_connectivity,edge_entries),init=[]))
  labels.d_to_dface_to_entity[1][vtxs_Γ] .= entity
  labels.d_to_dface_to_entity[2][edge_Γ] .= entity
  add_tag!(labels,name,[entity])
  return cell_to_entity
end

function _update_labels_locally!(e,model::DiscreteModel{3},mask,name)
  topo   = get_grid_topology(model)
  labels = get_face_labeling(model)
  cell_to_entity = labels.d_to_dface_to_entity[end]
  entity = maximum(cell_to_entity) + e
  # Vertices
  vtxs_Γ = findall(mask)
  vtx_edge_connectivity = Array(Geometry.get_faces(topo,0,1)[vtxs_Γ])
  vtx_face_connectivity = Array(Geometry.get_faces(topo,0,2)[vtxs_Γ])
  # Edges
  edge_entries = [findall(x->any(x .∈  vtx_edge_connectivity[1:end.!=j]),
    vtx_edge_connectivity[j]) for j = 1:length(vtx_edge_connectivity)]
  edge_Γ = unique(reduce(vcat,getindex.(vtx_edge_connectivity,edge_entries),init=[]))
  # Faces
  face_entries = [findall(x->count(x .∈  vtx_face_connectivity[1:end.!=j])>2,
    vtx_face_connectivity[j]) for j = 1:length(vtx_face_connectivity)]
  face_Γ = unique(reduce(vcat,getindex.(vtx_face_connectivity,face_entries),init=[]))
  labels.d_to_dface_to_entity[1][vtxs_Γ] .= entity
  labels.d_to_dface_to_entity[2][edge_Γ] .= entity
  labels.d_to_dface_to_entity[3][face_Γ] .= entity
  add_tag!(labels,name,[entity])
  return cell_to_entity
end

function mark_nodes(f,model::DistributedDiscreteModel)
  local_masks = map(local_views(model)) do model
    mark_nodes(f,model)
  end
  gids = get_face_gids(model,0)
  mask = PVector(local_masks,partition(gids))
  assemble!(|,mask) |> fetch  # Ghosts -> Owned with `or` applied
  consistent!(mask) |> fetch  # Owned  -> Ghost
  return mask
end

function mark_nodes(f,model::DiscreteModel)
  topo   = get_grid_topology(model)
  coords = get_vertex_coordinates(topo)
  mask = map(f,coords)
  return mask
end

"""
    initial_lsf(ξ,a;b)

Generate a function `f` according to
f(x) = -1/4 ∏ᵢ(cos(ξ*π*(xᵢ-bᵢ))) - a/4
where x is a vector with components xᵢ.
"""
initial_lsf(ξ,a;b=0) = x::VectorValue -> -1/4*prod(cos.(get_array(@.(ξ*pi*(x-b))))) - a/4

"""
    get_cartesian_element_sizes(model)

Given a CartesianDiscreteModel return the element size as a tuple.
"""
function get_cartesian_element_sizes(model::CartesianDiscreteModel)
  desc = get_cartesian_descriptor(model)
  return desc.sizes
end

function get_cartesian_element_sizes(model::DistributedDiscreteModel)
  local_Δ = map(local_views(model)) do model
    get_cartesian_element_sizes(model)
  end
  return getany(local_Δ)
end

const get_el_Δ = get_cartesian_element_sizes

"""
    isotropic_elast_tensor(D::Int,E::M,v::M)

Generate an isotropic `SymFourthOrderTensorValue` given
a dimension `D`, Young's modulus `E`, and Poisson's ratio `v`.
"""
function isotropic_elast_tensor(D::Int,E::Number,v::Number)
  if D == 2
    λ = E*v/((1+v)*(1-v)); μ = E/(2*(1+v))
    C = [
      λ+2μ  λ     0
      λ    λ+2μ   0
      0     0     μ
    ];
    return SymFourthOrderTensorValue(
      C[1,1], C[3,1], C[2,1],
      C[1,3], C[3,3], C[2,3],
      C[1,2], C[3,2], C[2,2]
    )
  elseif D == 3
    λ = E*v/((1+v)*(1-2v)); μ = E/(2*(1+v))
    C = [
      λ+2μ   λ      λ      0      0      0
      λ     λ+2μ    λ      0      0      0
      λ      λ     λ+2μ    0      0      0
      0      0      0      μ      0      0
      0      0      0      0      μ      0
      0      0      0      0      0      μ
    ];
    return SymFourthOrderTensorValue(
      C[1,1], C[6,1], C[5,1], C[2,1], C[4,1], C[3,1],
      C[1,6], C[6,6], C[5,6], C[2,6], C[4,6], C[3,6],
      C[1,5], C[6,5], C[5,5], C[2,5], C[4,5], C[3,5],
      C[1,2], C[6,2], C[5,2], C[2,2], C[4,2], C[3,2],
      C[1,4], C[6,4], C[5,4], C[2,4], C[4,4], C[3,4],
      C[1,3], C[6,3], C[5,3], C[2,3], C[4,3], C[3,3]
    )
  else
    @notimplemented
  end
end

#   function zerocrossing_period
#
# Find the period of oscillations based on a zero crossing algorithm.
#
# This is based on the zero crossing method from ChaosTools.jl
# See (https://github.com/JuliaDynamics/ChaosTools.jl)
function _zerocrossing_period(v;t=0:length(v)-1,line=sum(v)/length(v))
  inds = findall(@. ≥(line, $@view(v[2:end])) & <(line, $@view(v[1:end-1])))
  difft = diff(t[inds])
  sum(difft)/length(difft)
end

import Gridap.ReferenceFEs: get_order
function get_order(space::FESpace)
  return get_order(first(Gridap.CellData.get_data(get_fe_basis(space))))
end

function get_order(space::DistributedFESpace)
  order = map(get_order,local_views(space))
  return getany(order)
end

# Element diameter for general model

# According to Massing et al. (doi:10.1007/s10915-014-9838-9) the stabilisation terms
#   should use h_K denotes the diameter of element K, and h_F denotes the average
#   of the diameters of the elements sharing a facet K. The latter can be computed
#   as the mean on the facet triangulation.
#
# In this implementation, diameter is interpreted as the circumdiameter of the polytope.
"""
    get_element_diameters(model)

Given a general unstructured model return the element circumdiameters.
"""
function get_element_diameters(model)
  coords = get_cell_coordinates(model)
  polys = get_polytopes(model)
  @assert length(polys) == 1 "Only one cell type is currently supported"
  poly = first(polys)
  if poly == TRI
    return lazy_map(_get_tri_circumdiameter,coords)
  elseif poly == TET
    return lazy_map(_get_tet_circumdiameter,coords)
  else
    @notimplemented "Only triangles and tetrahedra are currently supported"
  end
end

"""
    get_element_diameter_field(model)

Given a general unstructured model return the element circumdiameters as a
CellField over the triangulation.
"""
function get_element_diameter_field(model)
  return CellField(get_element_diameters(model),Triangulation(model))
end

function get_element_diameters(model::DistributedDiscreteModel{Dc}) where Dc
  h = map(get_element_diameters,local_views(model))
  gids = get_face_gids(model,Dc)
  return PVector(h,partition(gids))
end

function get_element_diameter_field(model::DistributedDiscreteModel)
  Ω = Triangulation(model)
  fields = map(local_views(model)) do model
    h = get_element_diameters(model)
    CellField(h,Triangulation(model))
  end
  return DistributedCellField(fields,Ω)
end

# # Based on doi:10.1017/CBO9780511973611. C is the Cayley-Menger matrix.
# function _get_tri_circumdiameter(coords)
#   d12 = norm(coords[1]-coords[2])^2
#   d13 = norm(coords[1]-coords[3])^2
#   d23 = norm(coords[2]-coords[3])^2
#   # C = [
#   #   0  1   1   1
#   #   1  0  d12 d13
#   #   1 d12  0  d23
#   #   1 d13 d23  0
#   # ];
#   # M = -2inv(C);
#   # circumcentre = (M[1,2]*coords[1] + M[1,3]*coords[2] + M[1,4]*coords[3])/sum(M[1,2:end])
#   # circumdiameter = sqrt(M[1,1])
#   M11 = -((4*d12*d13*d23)/(d12^2+(d13-d23)^2-2*d12*(d13+d23)))
#   return sqrt(M11)
# end

# TODO: I have replaced the get diameter function with one based on the literature.
#   This is instead of circumdiameter. Once this is tested, all functions here should be
#   renamed before release!
function _get_tri_circumdiameter(coords)
  d12 = norm(coords[1]-coords[2])
  d13 = norm(coords[1]-coords[3])
  d23 = norm(coords[2]-coords[3])

  max(d12,d13,d23)
end


# function _get_tet_circumdiameter(coords)
#   d12 = norm(coords[1]-coords[2])^2
#   d13 = norm(coords[1]-coords[3])^2
#   d14 = norm(coords[1]-coords[4])^2
#   d23 = norm(coords[2]-coords[3])^2
#   d24 = norm(coords[2]-coords[4])^2
#   d34 = norm(coords[3]-coords[4])^2
#   # C = [
#   #   0  1   1   1   1
#   #   1  0  d12 d13 d14
#   #   1 d12  0  d23 d24
#   #   1 d13 d23  0  d34
#   #   1 d14 d24 d34  0
#   # ];
#   # M = -2inv(C);
#   # circumcentre = (M[1,2]*coords[1] + M[1,3]*coords[2] + M[1,4]*coords[3] + M[1,5]*coords[4])/sum(M[1,2:end])
#   # circumdiameter = sqrt(M[1,1])
#   M11 = (d14^2*d23^2+(d13*d24-d12*d34)^2-2*d14*d23*(d13*d24+d12*d34))/(
#     d13^2*d24+d12^2*d34+d23*(d14^2+d14*(d23-d24-d34)+d24*d34)-
#     d13*(d14*(d23+d24-d34)+d24*(d23-d24+d34))-d12*((d23+d24-d34)*d34+
#     d14*(d23-d24+d34)+d13*(-d23+d24+d34)))
#   return sqrt(M11)
# end

function _get_tet_circumdiameter(coords)
  d12 = norm(coords[1]-coords[2])
  d13 = norm(coords[1]-coords[3])
  d14 = norm(coords[1]-coords[4])
  d23 = norm(coords[2]-coords[3])
  d24 = norm(coords[2]-coords[4])
  d34 = norm(coords[3]-coords[4])

  max(d12,d13,d14,d23,d24,d34)
end

# Test that a distributed and serial field are the same.
#
# Note:
#   - This is only designed for small tests
#   - We require that the distributed model is generated with a global ordering
#     that matches the serial model. See function below.
function test_serial_and_distributed_fields(fhd::CellField,Vd,fhs::FEFunction,Vs)
  fhd_cell_values = map(local_views(Vd),local_views(fhd)) do Vd,fhd
    free = get_free_dof_values(fhd)
    diri = get_dirichlet_dof_values(Vd)
    scatter_free_and_dirichlet_values(Vd,free,diri)
  end

  free = get_free_dof_values(fhs)
  diri = get_dirichlet_dof_values(Vs)
  fhs_cell_values = scatter_free_and_dirichlet_values(Vs,free,diri)

  dmodel = get_background_model(get_triangulation(Vd))
  map(partition(get_cell_gids(dmodel)),fhd_cell_values) do gids,lfhd_cell_values
    lfhd_cell_values ≈ fhs_cell_values[local_to_global(gids)]
  end
end

# Generate a distributed model from a serial model with global ordering that
#  matches the serial model
function ordered_distributed_model_from_serial_model(ranks,model_serial)
  cell_to_part = reduce(vcat,[[i for j in 1:num_cells(model_serial)/length(ranks)] for i in 1:length(ranks)])
  append!(cell_to_part,[length(ranks) for i = 1: num_cells(model_serial) % length(ranks)]...)
  @assert length(cell_to_part) == num_cells(model_serial)
  DiscreteModel(ranks,model_serial,cell_to_part)
end

# MultiField
function test_serial_and_distributed_fields(fhd::DistributedMultiFieldCellField,Vd,fhs::MultiFieldFEFunction,Vs)
  @assert num_fields(fhd)==num_fields(Vd)==num_fields(fhs)==num_fields(Vs)
  result = map(i->test_serial_and_distributed_fields(fhd[i],Vd[i],fhs[i],Vs[i]),1:num_fields(fhd)) |> to_parray_of_arrays
  map(all,result)
end