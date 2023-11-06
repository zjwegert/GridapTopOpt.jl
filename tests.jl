using FillArrays

using Gridap
using Gridap.Geometry, Gridap.Arrays
using Gridap.Geometry: get_faces

using GridapDistributed, PartitionedArrays

############################################################################################

function EdgeTriangulation(model::DiscreteModel{Dc}) where Dc
  EdgeTriangulation(model,Base.OneTo(Geometry.num_faces(model,1)))
end

function EdgeTriangulation(model::DiscreteModel{Dc},tags) where Dc
  labeling = get_face_labeling(model)
  face_to_mask = get_face_mask(labeling,tags,1)
  face_to_bgface = findall(face_to_mask)
  return EdgeTriangulation(model,face_to_bgface)
end

function EdgeTriangulation(
  model::DiscreteModel{Dc},
  face_to_bgface::AbstractVector{<:Integer}) where {Dc}

  topo = get_grid_topology(model)
  bgface_grid = Grid(Geometry.ReferenceFE{1},model)
  bgface_to_lcell = Fill(1,Geometry.num_faces(model,1))

  face_grid = view(bgface_grid,face_to_bgface)
  cell_grid = get_grid(model)
  glue = Geometry.FaceToCellGlue(topo,cell_grid,face_grid,face_to_bgface,bgface_to_lcell)
  trian = BodyFittedTriangulation(model,face_grid,face_to_bgface)

  BoundaryTriangulation(trian,glue)
end

function EdgeTriangulation(model::GridapDistributed.DistributedDiscreteModel)
  trians = map(EdgeTriangulation,local_views(model))
  return GridapDistributed.DistributedTriangulation(trians,model)
end

function get_edge_coordinates(model::DiscreteModel)
  topo = get_grid_topology(model)
  get_edge_coordinates(topo)
end

function get_edge_coordinates(topo::GridTopology)
  e2n_map = Geometry.get_faces(topo,1,0)
  node_coords = Geometry.get_vertex_coordinates(topo)
  edge_coords = lazy_map(nodes->lazy_map(Reindex(node_coords),nodes),e2n_map)
  return edge_coords
end

function get_tangent_vector(trian::Triangulation{1})
  function my_tangent(m)
    t = m(VectorValue(1.0)) - m(VectorValue(0.0))
    return t/norm(t)
  end
  cmaps = get_cell_map(trian)
  return CellField(lazy_map(my_tangent,cmaps),trian)
end

function get_tangent_vector(trian::GridapDistributed.DistributedTriangulation)
  fields = map(get_tangent_vector,local_views(trian))
  GridapDistributed.DistributedCellField(fields)
end

"""
  `edge_orientation(edge_coords)`

  Returns:
    1 : if the edge is in the x-direction
    2 : if the edge is in the y-direction
    3 : if the edge is in the z-direction (only 3D)
"""
function edge_orientation(edge_coords)
  v = edge_coords[1]-edge_coords[2]
  return findfirst(x -> abs(x) > 1.e-3,v.data)
end

"""
  Returns a table that for each node contains the adjacent edge ids. 
   - If the edge is missing (for instance in boundaries), the corresponding entry is -1.
   - For each node, the edges are ordered as follows:
     [x-outgoing, x-incoming, y-outgoing, y-incoming, z-outgoing, z-incoming]
"""
function get_node_edge_nbors(model::DiscreteModel{Dc}) where Dc
  num_edges_x_node = (Dc==2) ? 4 : 6
  ptrs = zeros(Int,num_faces(model,0)+1); ptrs[1] = 1; ptrs[2:end] .= num_edges_x_node; 
  Gridap.Algebra.length_to_ptrs!(ptrs)
  
  e2n_map = Geometry.get_faces(get_grid_topology(model),1,0)
  orientations = lazy_map(edge_orientation,get_edge_coordinates(model))

  data = fill(-1,ptrs[end]-1)
  e2n_map_cache = array_cache(e2n_map)
  orientations_cache = array_cache(orientations)
  for E in 1:num_faces(model,1)
    E_nodes = getindex!(e2n_map_cache,e2n_map,E)
    E_orientation = getindex!(orientations_cache,orientations,E)
    for (iN,N) in enumerate(E_nodes)
      # Edge ordering in each node:
      #  [x-outgoing, x-incoming, y-outgoing, y-incoming, z-outgoing, z-incoming]
      id = 2*(E_orientation-1) + iN
      data[ptrs[N]+id-1] = E
    end
  end
  return Table(data,ptrs)
end

function get_node_edge_nbors(model::GridapDistributed.DistributedDiscreteModel)
  return map(get_node_edge_nbors,local_views(model))
end

"""
  Returns a table that for each node contains the directional derivatives in adjacent edges:
    - If the edge is missing (for instance in boundaries), the corresponding entry is 0.0.
    - For each node, the derivatives are ordered as follows:
      [d/dx⁺, -d/dx⁻, d/dy⁺, -d/dy⁻, d/dz⁺, -d/dz]
"""
function evaluate_derivatives(φ::FEFunction,Λ::Triangulation,edge_nbors,t)
  quad = CellQuadrature(Λ,0)
  pts = get_cell_points(quad)

  vals = lazy_map(first,(∇(φ)⋅t)(pts))
  idx  = lazy_map(PosNegReindex(vals,[0.0]),edge_nbors.data)
  return Table(idx,edge_nbors.ptrs)
end

function evaluate_derivatives(φ::GridapDistributed.DistributedCellField,
                              Λ::GridapDistributed.DistributedTriangulation,
                              edge_nbors,
                              t)
  return map(evaluate_derivatives,local_views(φ),local_views(Λ),edge_nbors,local_views(t))
end

############################################################################################

parallel = true
D = 3

if D == 2
  np = (2,1)
  domain = (0,1,0,1)
  nc = (4,4)
else
  np = (2,1,1)
  domain = (0,1,0,1,0,1)
  nc = (4,4,4)
end

if parallel
  ranks = with_debug() do distribute
    distribute(LinearIndices((prod(np),)))
  end
  model = CartesianDiscreteModel(ranks,np,domain,nc)
else
  model = CartesianDiscreteModel(domain,nc)
end

Ω = Triangulation(model)
Λ = EdgeTriangulation(model)
t = get_tangent_vector(Λ)
edge_nbors = get_node_edge_nbors(model)

reffe = ReferenceFE(lagrangian,Float64,1)
V = FESpace(model,reffe)

x = !parallel ? randn(num_free_dofs(V)) : prandn(partition(get_free_dof_ids(V)))
u = FEFunction(V,x)

derivatives = evaluate_derivatives(u,Λ,edge_nbors,t)
