using GridapTopOpt
using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Adaptivity, Gridap.Arrays

using DataStructures

const CUT = 0

# TODO: Can be optimized CartesianModels
function generate_neighbor_graph(model::DiscreteModel{Dc}) where Dc
  topo = get_grid_topology(model)
  cell_to_node = Geometry.get_faces(topo, Dc, 0)
  node_to_cell = Geometry.get_faces(topo, 0, Dc)
  cell_to_nbors = map(1:num_cells(model)) do cell
    unique(sort(vcat(map(n -> view(node_to_cell,n), view(cell_to_node,cell))...)))
  end
  return cell_to_nbors
end

"""
  Given an initial interface cell, enqueue all the CUT cells in the same interface
  inside the queue `q_cut` and mark them as touched in the `touched` array.
"""
function enqueue_interface!(q_cut,cell_to_nbors,cell_to_inoutcut,touched,cell)
  q = Queue{Int}(); enqueue!(q,cell)
  enqueue!(q_cut,cell)
  touched[cell] = true
  while !isempty(q)
    cell = dequeue!(q)
    nbors = cell_to_nbors[cell]
    for nbor in nbors
      if !touched[nbor] && (cell_to_inoutcut[nbor] == CUT)
        touched[nbor] = true
        enqueue!(q_cut,nbor)
        enqueue!(q,nbor)
      end
    end
  end
end

function tag_isolated_volumes(
  model::DiscreteModel{Dc}, cell_to_inoutcut::Vector{<:Integer}
) where Dc

  n_cells = num_cells(model)
  cell_to_nbors = generate_neighbor_graph(model)

  n_color = 0
  cell_color = zeros(Int16, n_cells)
  color_to_inout = Int8[]
  touched  = falses(n_cells)
  q, q_cut = Queue{Int}(), Queue{Int}()
  
  # First pass: Color IN/OUT cells
  #   - We assume that every IN/OUT transition can be bridged by a CUT cell
  first_cell = findfirst(state -> state != CUT, cell_to_inoutcut)
  enqueue!(q,first_cell); touched[first_cell] = true; # Queue first cell
  while !isempty(q)
    cell  = dequeue!(q)
    nbors = cell_to_nbors[cell]
    state = cell_to_inoutcut[cell]
    
    # Mark with color
    if state != CUT
      i = findfirst(!iszero,view(cell_color,nbors))
      if isnothing(i) # New patch
        n_color += 1
        cell_color[cell] = n_color
        push!(color_to_inout, state)
      else # Existing patch
        color = cell_color[nbors[i]]
        cell_color[cell] = color
      end
    end

    # Queue and touch unseen neighbors
    # We touch neighbors here to avoid enqueuing the same cell multiple times
    for nbor in nbors
      if !touched[nbor]
        touched[nbor] = true
        enqueue!(q,nbor)
        if cell_to_inoutcut[nbor] == CUT
          enqueue_interface!(q_cut,cell_to_nbors,cell_to_inoutcut,touched,nbor)
        end
      end
    end
  end

  # Second pass: Color CUT cells
  #   - We assume that every CUT cell has an IN neighbor
  #   - We assume all IN neighbors have the same color
  # Then we assign the same color to the CUT cell
  while !isempty(q_cut)
    cell  = dequeue!(q_cut)
    nbors = cell_to_nbors[cell]
    @assert cell_to_inoutcut[cell] == CUT

    i = findfirst(n -> cell_to_inoutcut[n] == IN, nbors)
    @assert !isnothing(i)
    cell_color[cell] = cell_color[nbors[i]]
  end

  return cell_color, color_to_inout
end

order = 1
n = 20
model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1),(n,n)))
Ω = Triangulation(model)

reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe_scalar)

φh = interpolate(x->cos(2π*x[1])*cos(2π*x[2])-0.11,V_φ)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo)

colors, color_to_inout = tag_isolated_volumes(model,bgcell_to_inoutcut)

writevtk(
  Ω,"results/background",
  cellfields=[
    "φh"=>φh
  ],
  celldata=["inoutcut"=>bgcell_to_inoutcut,"volumes"=>colors];
  append=false
)
