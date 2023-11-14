using Gridap.ReferenceFEs

# Update label given function f_Γ. e should be the count of added tags.
function update_labels!(e::Int,model::D,f_Γ::F,::T,name::String) where {
        M<:AbstractFloat,F<:Function,T<:NTuple{2,M},D<:DistributedDiscreteModel}
    mask = mark_nodes(f_Γ,model)
    cell_to_entity = map(local_views(model),local_views(mask)) do model,mask
        labels = get_face_labeling(model)
        cell_to_entity = labels.d_to_dface_to_entity[end]
        entity = maximum(cell_to_entity) + e
        # Vertices
        vtxs_Γ = findall(mask)
        vtx_edge_connectivity = Array(Geometry.get_faces(model.grid_topology,0,1)[vtxs_Γ])
        # Edges
        edge_entries = [findall(x->any(x .∈  vtx_edge_connectivity[1:end.!=j]),
            vtx_edge_connectivity[j]) for j = 1:length(vtx_edge_connectivity)]
        edge_Γ = unique(reduce(vcat,getindex.(vtx_edge_connectivity,edge_entries),init=[]))
        labels.d_to_dface_to_entity[1][vtxs_Γ] .= entity
        labels.d_to_dface_to_entity[2][edge_Γ] .= entity
        add_tag!(labels,name,[entity])
        cell_to_entity
    end
    cell_gids=get_cell_gids(model)
    cache=GridapDistributed.fetch_vector_ghost_values_cache(cell_to_entity,partition(cell_gids))
    GridapDistributed.fetch_vector_ghost_values!(cell_to_entity,cache)
end

function update_labels!(e::Int,model::D,f_Γ::F,::T,name::String) where {
        M<:AbstractFloat,F<:Function,T<:NTuple{3,M},D<:DistributedDiscreteModel}
    mask = mark_nodes(f_Γ,model)
    cell_to_entity = map(local_views(model)) do model
        labels = get_face_labeling(model)
        cell_to_entity = labels.d_to_dface_to_entity[end]
        entity = maximum(cell_to_entity) + e
        # Vertices
        vtxs_Γ = findall(mask)
        vtx_edge_connectivity = Array(Geometry.get_faces(model.grid_topology,0,1)[vtxs_Γ])
        vtx_face_connectivity = Array(Geometry.get_faces(model.grid_topology,0,2)[vtxs_Γ])
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
    cell_gids=get_cell_gids(model)
    cache=GridapDistributed.fetch_vector_ghost_values_cache(cell_to_entity,partition(cell_gids))
    GridapDistributed.fetch_vector_ghost_values!(cell_to_entity,cache)
end

function mark_nodes(f,model)
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