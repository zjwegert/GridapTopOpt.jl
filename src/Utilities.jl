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
Base.@kwdef struct SmoothErsatzMaterialInterpolation{M<:Vector{<:Number},N<:Vector{<:Number}}
  η::M
  ϵ::N
  H = x -> H_η(x,first(η))
  DH = x -> DH_η(x,first(η))
  I = φ -> (1 - H(φ)) + first(ϵ)*H(φ)
  ρ = φ -> 1 - H(φ)
end

function SmoothErsatzMaterialInterpolation(;η::M,ϵ::N=10^-3) where {M<:Number,N<:Number}
  return SmoothErsatzMaterialInterpolation{Vector{M},Vector{N}}(η=[η],ϵ=[ϵ])
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

Given a tag number `e`, a `CartesianDiscreteModel` or `DistributedDiscreteModel` model,
an indicator function `f_Γ`, and a string `name`, label the corresponding vertices, edges, and faces
as `name`.

Note: `f_Γ` must recieve a Vector and return a Boolean depending on whether it indicates Γ
"""
function update_labels!(e::Integer,model::CartesianDiscreteModel,f_Γ::Function,name::String)
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

function _update_labels_locally!(e,model::CartesianDiscreteModel{2},mask,name)
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

function _update_labels_locally!(e,model::CartesianDiscreteModel{3},mask,name)
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
    get_el_Δ(model)

Given a CartesianDiscreteModel or DistributedDiscreteModel that is
uniform, return the element size as a tuple. 
"""
function get_el_Δ(model::CartesianDiscreteModel)
  desc = get_cartesian_descriptor(model)
  return desc.sizes
end

function get_el_Δ(model::DistributedDiscreteModel)
  local_Δ = map(local_views(model)) do model
    get_el_Δ(model)
  end
  return getany(local_Δ)
end

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