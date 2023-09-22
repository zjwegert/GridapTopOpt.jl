# Heaviside function
function H_η(t::M;η::M) where M<:AbstractFloat
    if t<-η
        return zero(M)
    elseif abs(t)<=η
        return M(1/2*(1+t/η+1/pi*sin(pi*t/η)))
    elseif t>η
        return one(M)
    end
end

function DH_η(t::M;η::M) where M<:AbstractFloat
    if t<-η
        return zero(M)
    elseif abs(t)<=η
        return M(1/2/η*(1+cos(pi*t/η)))
    elseif t>η
        return zero(M)
    end
end

# Material interpolation
Base.@kwdef struct SmoothErsatzMaterialInterpolation{M<:AbstractFloat}
    η::M # Smoothing radius
    ϵₘ::M = 10^-3 # Void material multiplier
    H = x -> H_η(x,η=η)
    DH = x -> DH_η(x,η=η)
    I = φ -> (1 - H(φ)) + ϵₘ*H(φ)
    ρ = φ -> 1 - H(φ)
end

write_vtk(Ω,path,φh,uh,H) = writevtk(Ω,path,cellfields=["phi"=>φh,
    "H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh"=>uh])

# Update layout of ghost nodes to match columns of stiffness matrix
function correct_ghost_layout(xh::DistributedCellField,cols)
    x_tmp = get_free_dof_values(xh)
    x = pfill(0.0, cols)
    map(local_views(own_values(x)),local_views(own_values(x_tmp))) do x,x_tmp
        x .= x_tmp
    end
    consistent!(x) |> fetch
    return x
end

# Inf norm
function infnorm(a::PVector)
    contibs = map(own_values(a)) do oid_to_value
        norm(oid_to_value,Inf)
    end
    reduce(max,contibs;init=zero(eltype(contibs)))
end

# Update label given function f_Γ. e should be the count of added tags.
function update_labels!(e::Int,model::D,f_Γ::F,::T,name::String) where {
        M<:AbstractFloat,F<:Function,T<:NTuple{2,M},D<:DistributedDiscreteModel}
    cell_to_entity = map(local_views(model)) do model
        labels = get_face_labeling(model)
        cell_to_entity = labels.d_to_dface_to_entity[end]
        entity = maximum(cell_to_entity) + e
        # Vertices
        vtx_coords = model.grid_topology.vertex_coordinates
        vtxs_Γ = findall(isone,f_Γ.(vtx_coords))
        vtx_edge_connectivity = Array(model.grid_topology.n_m_to_nface_to_mfaces[1,2][vtxs_Γ]);
        # Edges
        edge_entires = [findall(x->any(x .∈  vtx_edge_connectivity[1:end.!=j]),
            vtx_edge_connectivity[j]) for j = 1:length(vtx_edge_connectivity)]
        edge_Γ = unique(reduce(vcat,getindex.(vtx_edge_connectivity,edge_entires),init=[]))
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
    cell_to_entity = map(local_views(model)) do model
        labels = get_face_labeling(model)
        cell_to_entity = labels.d_to_dface_to_entity[end]
        entity = maximum(cell_to_entity) + e
        # Vertices
        vtx_coords = model.grid_topology.vertex_coordinates
        vtxs_Γ = findall(isone,f_Γ.(vtx_coords))
        vtx_edge_connectivity = Array(model.grid_topology.n_m_to_nface_to_mfaces[1,2][vtxs_Γ]);
        vtx_face_connectivity = Array(model.grid_topology.n_m_to_nface_to_mfaces[1,3][vtxs_Γ]);
        # Edges
        edge_entires = [findall(x->any(x .∈  vtx_edge_connectivity[1:end.!=j]),
            vtx_edge_connectivity[j]) for j = 1:length(vtx_edge_connectivity)]
        edge_Γ = unique(reduce(vcat,getindex.(vtx_edge_connectivity,edge_entires),init=[]))
        # Faces
        face_entires = [findall(x->count(x .∈  vtx_face_connectivity[1:end.!=j])>2,
            vtx_face_connectivity[j]) for j = 1:length(vtx_face_connectivity)]
        face_Γ = unique(reduce(vcat,getindex.(vtx_face_connectivity,face_entires),init=[]))
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