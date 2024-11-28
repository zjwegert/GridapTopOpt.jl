using GridapTopOpt
using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity, Gridap.Arrays

using DataStructures

const CUT = 0

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
  cell::Int,color::Int16,
  cell_to_nbors::Vector{Vector{Int32}},
  cell_to_inoutcut::Vector{Int8},
  cell_to_color::Vector{Int16},
  touched::BitVector
)
  q = Queue{Int}()
  enqueue!(q,cell)
  touched[cell] = true
  state = cell_to_inoutcut[cell]
  
  while !isempty(q)
    cell = dequeue!(q)
    cell_to_color[cell] = color

    nbors = cell_to_nbors[cell]
    for nbor in nbors
      if !touched[nbor] && (cell_to_inoutcut[nbor] == state)
        enqueue!(q,nbor)
        touched[nbor] = true
      end
    end
  end
end

function tag_isolated_volumes(
  model::DiscreteModel{Dc}, cell_to_inoutcut::Vector{<:Integer}
) where Dc

  n_cells = num_cells(model)
  cell_to_nbors = generate_neighbor_graph(model)

  color_to_inout = Int8[]
  cell_to_color = zeros(Int16, n_cells)
  touched  = falses(n_cells)

  cell = 1; n_color = zero(Int16)
  while cell <= n_cells
    if !touched[cell]
      n_color += one(Int16)
      push!(color_to_inout, cell_to_inoutcut[cell])
      tag_volume!(cell, n_color, cell_to_nbors, cell_to_inoutcut, cell_to_color, touched)
    end
    cell += 1
  end
  
  return cell_to_color, color_to_inout
end

function find_tagged_volumes(
  model::DiscreteModel{Dc},tags,
  cell_to_color::Vector{Int16},
  color_to_inout::Vector{Int8};
  Df = Dc - 1
) where Dc
  topo = get_grid_topology(model)
  faces = findall(get_face_mask(get_face_labeling(model),tags,Df))
  face_to_cell = Geometry.get_faces(topo, Df, Dc)

  is_tagged = falses(length(color_to_inout))
  for face in faces
    for cell in view(face_to_cell,face)
      color = cell_to_color[cell]
      is_tagged[color] = true
    end
  end

  return is_tagged
end

order = 1
n = 100
model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1),(n,n)))
Ω = Triangulation(model)

reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe_scalar)

# φh = interpolate(x->cos(2π*x[1])*cos(2π*x[2])-0.11,V_φ)

# R = 0.195
R = 0.2 # This fails
f(x0,r) = x -> sqrt((x[1]-x0[1])^2 + (x[2]-x0[2])^2) - r
φh = interpolate(x->-f([0.5,0.5],R)(x),V_φ)
# φh = interpolate(x->min(f([0.25,0.5],R)(x),f([0.75,0.5],R)(x)),V_φ)


geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo)
colors, color_to_inout = tag_isolated_volumes(model,bgcell_to_inoutcut)

color_to_tagged = find_tagged_volumes(model,["tag_5","tag_7"],colors,color_to_inout)
cell_to_tagged = map(c -> color_to_tagged[c], colors)

writevtk(
  Ω,"results/background",
  cellfields=[
    "φh"=>φh
  ],
  celldata=[
    "inoutcut"=>bgcell_to_inoutcut,
    "volumes"=>colors,
    "tagged"=>cell_to_tagged,
  ];
  append=false
)
