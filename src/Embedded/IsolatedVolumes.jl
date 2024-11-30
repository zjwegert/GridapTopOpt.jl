

# TODO: Can be optimized for CartesianModels
function generate_neighbor_graph(model::DiscreteModel{Dc}) where Dc
  topo = get_grid_topology(model)
  cell_to_node = Geometry.get_faces(topo, Dc, 0)
  node_to_cell = Geometry.get_faces(topo, 0, Dc)
  cell_to_nbors = map(1:num_cells(model)) do cell
    unique(sort(vcat(map(n -> view(node_to_cell,n), view(cell_to_node,cell))...)))
  end
  return cell_to_nbors
end

function tag_volume!(
  cell::Int,color::Int16,group::Union{Integer,NTuple{N,Integer}},
  cell_to_nbors::Vector{Vector{Int32}},
  cell_to_inoutcut::Vector{Int8},
  cell_to_color::Vector{Int16},
  touched::BitVector
) where N
  @assert cell_to_inoutcut[cell] ∈ group
  
  q = Queue{Int}()
  enqueue!(q,cell)
  touched[cell] = true
  
  while !isempty(q)
    cell = dequeue!(q)
    cell_to_color[cell] = color
    
    state = cell_to_inoutcut[cell]
    nbors = cell_to_nbors[cell]
    for nbor in nbors
      if !touched[nbor] && state ∈ group
        enqueue!(q,nbor)
        touched[nbor] = true
      end
    end
  end
end

function tag_isolated_volumes(
  model::DiscreteModel{Dc},
  cell_to_inoutcut::Vector{<:Integer};
  groups = (CUT,IN,OUT)
) where Dc

  n_cells = num_cells(model)
  cell_to_nbors = generate_neighbor_graph(model)

  color_to_group = Int8[]
  cell_to_color = zeros(Int16, n_cells)
  touched  = falses(n_cells)

  cell = 1; n_color = zero(Int16)
  while cell <= n_cells
    if !touched[cell]
      n_color += one(Int16)
      state = cell_to_inoutcut[cell]
      group = findfirst(g -> state ∈ g, groups)
      push!(color_to_group, group)
      tag_volume!(cell, n_color, groups[group], cell_to_nbors, cell_to_inoutcut, cell_to_color, touched)
    end
    cell += 1
  end
  
  return cell_to_color, color_to_group
end

function find_tagged_volumes(
  model::DiscreteModel{Dc},tags,
  cell_to_color::Vector{Int16},
  color_to_group::Vector{Int8};
  Df = Dc - 1
) where Dc
  topo = get_grid_topology(model)
  faces = findall(get_face_mask(get_face_labeling(model),tags,Df))
  face_to_cell = Geometry.get_faces(topo, Df, Dc)

  is_tagged = falses(length(color_to_group))
  for face in faces
    for cell in view(face_to_cell,face)
      color = cell_to_color[cell]
      is_tagged[color] = true
    end
  end

  return is_tagged
end

function get_isolated_volumes_mask(
  cutgeo::EmbeddedDiscretization,dirichlet_tags
)
  model = get_background_model(cutgeo)
  geo = get_geometry(cutgeo)

  bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo)
  cell_to_color, color_to_group = tag_isolated_volumes(model,bgcell_to_inoutcut;groups=((CUT,IN),OUT))
  color_to_tagged = find_tagged_volumes(model,dirichlet_tags,cell_to_color,color_to_group)
  
  data = map(c -> !color_to_tagged[c] && isone(color_to_group[c]), cell_to_color)
  return CellField(collect(Float64,data),Triangulation(model))  
end
