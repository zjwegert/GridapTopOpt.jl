## Initial LSF
gen_lsf(ξ,a;b=0) = x::VectorValue -> -1/4*prod(cos.(get_array(@.(ξ*pi*(x-b))))) - a/4

## Get element size Δ
function get_Δ(model::CartesianDiscreteModel)
  desc = get_cartesian_descriptor(model)
  desc.sizes
end

function get_Δ(model::DistributedDiscreteModel)
  local_Δ = map(local_views(model)) do model
    desc = get_cartesian_descriptor(model)
    desc.sizes
  end
  getany(local_Δ)
end

## Create label given function f_Γ. e is a count of added tags. TODO: Can this go in GridapExtensions.jl? @Jordi
function update_labels!(e::Int,model::CartesianDiscreteModel,f_Γ::Function,name::String)
  mask = mark_nodes(f_Γ,model)
  _update_labels_locally!(e,model,mask,name)
  nothing
end

function update_labels!(e::Int,model::DistributedDiscreteModel,f_Γ::Function,name::String)
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
  cell_to_entity
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
  cell_to_entity
end

function mark_nodes(f,model::DistributedDiscreteModel)
  local_masks = map(local_views(model)) do model
    topo   = get_grid_topology(model)
    coords = get_vertex_coordinates(topo)
    mask = map(f,coords)
    return mask
  end
  gids = get_face_gids(model,0)
  mask = PVector(local_masks,partition(gids))
  assemble!(|,mask) |> fetch  # Ghosts -> Owned with `or` applied
  consistent!(mask) |> fetch  # Owned  -> Ghost
  return mask
end

function mark_nodes(f,model::CartesianDiscreteModel)
  topo   = get_grid_topology(model)
  coords = get_vertex_coordinates(topo)
  mask = map(f,coords)
  return mask
end

# Isotropic elasticity tensors
function isotropic_2d(E::M,ν::M) where M<:AbstractFloat
  λ = E*ν/((1+ν)*(1-ν)); μ = E/(2*(1+ν))
  C = [λ+2μ  λ     0
        λ    λ+2μ   0
        0     0     μ];
  SymFourthOrderTensorValue(
      C[1,1], C[3,1], C[2,1],
      C[1,3], C[3,3], C[2,3],
      C[1,2], C[3,2], C[2,2])
end

function isotropic_3d(E::M,ν::M) where M<:AbstractFloat
  λ = E*ν/((1+ν)*(1-2ν)); μ = E/(2*(1+ν))
  C =[λ+2μ   λ      λ      0      0      0
      λ     λ+2μ    λ      0      0      0
      λ      λ     λ+2μ    0      0      0
      0      0      0      μ      0      0
      0      0      0      0      μ      0
      0      0      0      0      0      μ];
  return SymFourthOrderTensorValue(
      C[1,1], C[6,1], C[5,1], C[2,1], C[4,1], C[3,1],
      C[1,6], C[6,6], C[5,6], C[2,6], C[4,6], C[3,6],
      C[1,5], C[6,5], C[5,5], C[2,5], C[4,5], C[3,5],
      C[1,2], C[6,2], C[5,2], C[2,2], C[4,2], C[3,2],
      C[1,4], C[6,4], C[5,4], C[2,4], C[4,4], C[3,4],
      C[1,3], C[6,3], C[5,3], C[2,3], C[4,3], C[3,3])
end

# Logging
function make_dir(path;ranks=nothing) # TODO: Adjust to mkpath?
  if i_am_main(ranks)
    !isdir(path) ? mkdir(path) : rm(path,recursive=true);
    !isdir(path) ? mkdir(path) : 0;
  end
end

function print_history(it,entries::Vector{<:Pair};ranks=nothing) # TODO: Remove
  if i_am_main(ranks)
    str = "it: $it"
    for pair in entries
      val = round.(pair.second;digits=5)
      str*=" | $(pair.first): $val"
    end
    println(str)
  end
end

function write_vtk(Ω,path,it,entries::Vector{<:Pair};iter_mod=10) # TODO: Rename to writevtk
  if isone(it) || iszero(it % iter_mod) 
    writevtk(Ω,path,cellfields=entries)
  end

  if iszero(it % 10*iter_mod)
    GC.gc() # Garbage collection, due to memory leak in writevtk - TODO
  end 
end

"""
  write_vector(dir::String,φ::PVector,delim)

Write a PVector to an array of files corresponding to each partition.
"""
function write_vector(dir::String,φ::PVector,delim)
  i_am_main(get_parts(φ)) && mkpath(dir)
  map(local_views(φ),get_parts(φ)) do φ,id
    writedlm(joinpath(dir,basename(dir)*"_$id.txt"),φ,delim)
  end |> fetch
end

"""
  write_vector(dir::String,φ::Vector,delim)

Write a vector to a file.
"""
function write_vector(dir::String,φ::Vector,delim)
  writedlm(joinpath(dir*".txt"),φ,delim)
end

"""
  write_file_to_vector!(dir::String,φ::PVector,delim)

Given a group of files, attempt to write data to each partition of a 
PVector.
"""
function write_file_to_vector!(dir::String,φ::PVector,delim)
  map(local_views(φ),get_parts(φ)) do φ,id
    file_path = joinpath(dir,basename(dir)*"_$id.txt");
    write_file_to_vector!(file_path,φ,delim)
  end
  consistent!(φ) |> fetch
end

"""
  write_file_to_vector!(dir::String,φ::PVector,delim)

Given a file, attempt to write data to a PVector.
"""
function write_file_to_vector!(file_path::String,φ::Vector,delim)
  @check isfile(file_path) "File `$file_path` not found. Check file names and number of partition."
  file_data = readdlm(file_path,delim);
  @check isequal(length(file_data),length(φ)) "Vector lengths not consistent. $(size(file_data)) !== $(size(φ))"
  copyto!(φ,file_data)
end