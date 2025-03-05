
using Gridap
using Gridap.Geometry, Gridap.Arrays, Gridap.TensorValues, Gridap.Helpers
using Gridap.ReferenceFEs
using Gridap.ReferenceFEs: get_graph, isactive

using STLCutters
using STLCutters: complete_graph!, disconnect_graph!, add_open_vertices!, OPEN

using GridapEmbedded, GridapEmbedded.Interfaces, GridapEmbedded.LevelSetCutters
using GridapTopOpt

Base.round(a::VectorValue{D,T};kwargs...) where {D,T} = VectorValue{D,T}(round.(a.data;kwargs...))

function get_node_reindex(p::Polytope{D}) where D
  if (D == 3) || (p == TRI)
    return Base.OneTo(num_vertices(p))
  elseif p == QUAD
    return [1,2,4,3]
  else
    @notimplemented
  end
end

function get_face_reindex(p::Polytope{D}, d::Int) where D
  n2o_node = get_node_reindex(p)
  iszero(d) && return n2o_node

  q = GeneralPolytope{D}(p)
  old_f2n = get_faces(p,d,0)
  new_f2n = get_faces(q,d,0)

  same_face(f,g) = all(n -> n ∈ g, f)
  same_face(f) = g -> same_face(f,g)
  n2o_face = zeros(Int,num_faces(p,d))
  for (new, new_face) in enumerate(new_f2n)
    old = findfirst(same_face(n2o_node[new_face]), old_f2n)
    @assert !isnothing(old)
    n2o_face[new] = old
  end

  return n2o_face
end

function compact!(graph)
  n_old = length(graph)
  old_to_new = fill(UNSET,n_old)
  new = 0
  for old in 1:n_old
    isempty(graph[old]) && continue
    new += 1
    old_to_new[old] = new
  end
  new_to_old = find_inverse_index_map(old_to_new,new)

  keepat!(graph,new_to_old)
  f(i) = ifelse(i ∈ (OPEN,UNSET), i, old_to_new[i])
  map!(i->map!(f,i,i),graph,graph)

  return graph, new_to_old
end

function split_postprocess!(graph,vertices,input_poly,values,(≶))
  complete_graph!(graph,num_vertices(input_poly))
  disconnect_graph!(graph,num_vertices(input_poly),values,(≶))
  graph, new_to_old = compact!(graph)
  D = num_cell_dims(input_poly)
  poly = GeneralPolytope{D}(vertices[new_to_old],graph)
  return poly, new_to_old
end

function split(p::Polyhedron,vertex_values)
  interpolate_values(v1,v2,w1,w2) = v1 + w1/(w1+w2)*(v2-v1)
  is_in(v) = v < 0

  all(is_in, vertex_values) && return p, nothing
  all(!is_in, vertex_values) && return nothing, p

  graph = get_graph(p)
  edge_nodes = get_faces(p,1,0)

  vertices = copy(get_vertex_coordinates(p))
  edges = Int[]
  in_graph = deepcopy(graph)
  out_graph = deepcopy(graph)
  
  D = 3
  n_vertices = num_vertices(p)
  for v in 1:n_vertices
    isactive(p,v) || continue
    vertex_values[v] < 0 && continue
    for (i,vneig) in enumerate(graph[v])
      w, wneig = vertex_values[v], vertex_values[vneig]
      w*wneig > 0 && continue
      
      vertex = interpolate_values(vertices[v],vertices[vneig],abs(w),abs(wneig))
      push!(vertices, vertex)

      e = findfirst(e -> e == [v,vneig] || e == [vneig,v], edge_nodes)
      push!(edges, e)

      push!(in_graph, fill(UNSET,D))
      in_graph[v][i] = length(vertices)
      in_graph[end][1] = v
      
      push!(out_graph, fill(UNSET,D))
      ineig = findfirst(isequal(v), graph[vneig])
      out_graph[vneig][ineig] = length(vertices)
      out_graph[end][1] = vneig
    end
  end

  new_vertices = vertices[num_vertices(p)+1:end]
  p_out = split_postprocess!(out_graph,vertices,p,vertex_values,(>))
  p_in = split_postprocess!(in_graph,vertices,p,vertex_values,(<))
  p_out, p_in, new_vertices, edges # TODO: Why are they reversed? 
end

function split(p::Polygon,vertex_values)
  interpolate_values(v1,v2,w1,w2) = v1 + (w1/(w1+w2))*(v2-v1)
  is_in(v) = v < 0

  all(is_in, vertex_values) && return p, nothing
  all(!is_in, vertex_values) && return nothing, p

  graph = get_graph(p)
  edge_nodes = get_faces(p,1,0)
  vertices = copy(get_vertex_coordinates(p))
  in_nodes = Int[]
  out_nodes = Int[]
  edges = Int[]

  start = 1
  current, next = start, first(graph[start])
  while true
    v_current, v_next = vertices[current], vertices[next]
    w_current, w_next = vertex_values[current], vertex_values[next]

    is_in(w_current) ? push!(in_nodes, current) : push!(out_nodes, current)

    if w_current*w_next < 0
      vertex = interpolate_values(v_current,v_next,abs(w_current),abs(w_next))
      
      push!(vertices, vertex)
      push!(in_nodes, length(vertices))
      push!(out_nodes, length(vertices))

      e = findfirst(e -> e == [current,next], edge_nodes)
      push!(edges, e)
    end

    current, next = next, first(graph[next])
    isequal(current,start) && break
  end

  new_vertices = vertices[num_vertices(p)+1:end]
  p_in = Polygon(vertices[in_nodes])
  p_out = Polygon(vertices[out_nodes])
  return (p_in, in_nodes), (p_out, out_nodes), new_vertices, edges
end

function cut_conforming(topo::UnstructuredGridTopology{D}, cell_values) where D
  @notimplementedif !isone(length(get_polytopes(topo)))

  p_ref = first(get_polytopes(topo))
  node_reindex = get_face_reindex(p_ref,0)
  edge_reindex = get_face_reindex(p_ref,1)
  cell_nodes = Geometry.get_faces(topo,D,0)
  cell_edges = Geometry.get_faces(topo,D,1)
  Tn = eltype(eltype(cell_nodes))
  
  is_cut(vals) = any(v->v<0,vals) && any(v->v>0,vals)
  cell_iscut = map(is_cut,cell_values)

  max_sc = ifelse(is_simplex(p_ref),2,ifelse(D==2,3,5))
  max_cn = ifelse(is_simplex(p_ref),2*(D-1),ifelse(D==2,4,12))
  max_subcells = sum(c -> ifelse(!c, 1, max_sc), cell_iscut)
  max_cutnodes = sum(c -> ifelse(!c, 0, max_cn), cell_iscut)
  max_nodes = num_faces(topo,0) + max_cutnodes
  
  subcell_nodes = Vector{Vector{Tn}}(undef,max_subcells)
  subcell_polys = Vector{GeneralPolytope{D,D,Float64,Nothing}}(undef,max_subcells)
  subcell_to_inout = Vector{Int8}(undef,max_subcells)
  subcell_to_cell = Vector{Int}(undef,max_subcells)

  coords = Geometry.get_vertex_coordinates(topo)
  vertex_coordinates = Vector{eltype(coords)}(undef,max_nodes)
  vertex_coordinates[1:num_faces(topo,0)] .= coords

  nodes_cache = array_cache(cell_nodes)
  edges_cache = array_cache(cell_edges)
  values_cache = array_cache(cell_values)
  
  n_subcells = 0
  n_nodes = num_faces(topo,0)
  edge_to_new_node = zeros(Tn,num_faces(topo,1))
  for cell in 1:num_cells(topo)
    nodes = getindex!(nodes_cache,cell_nodes,cell)
    values = getindex!(values_cache,cell_values,cell)
    vertices = vertex_coordinates[nodes]
    p = GeneralPolytope{D}(p_ref,vertices)
    nodes = nodes[node_reindex]
    if !cell_iscut[cell]
      n_subcells += 1
      subcell_nodes[n_subcells] = nodes
      subcell_polys[n_subcells] = p
      subcell_to_inout[n_subcells] = ifelse(all(v->v<0,values),IN,OUT)
      subcell_to_cell[n_subcells] = cell
    else
      (p_in, lnodes_in), (p_out, lnodes_out), new_vertices, ledges = split(p,values)
      edges = getindex!(edges_cache,cell_edges,cell)
      edges = edges[edge_reindex][ledges]
      
      for (ie,e) in enumerate(edges)
        if iszero(edge_to_new_node[e])
          edge_to_new_node[e] = (n_nodes += 1)
          vertex_coordinates[n_nodes] = new_vertices[ie]
        end
      end
      nodes = [nodes...,edge_to_new_node[edges]...]

      n_subcells += 1
      subcell_nodes[n_subcells] = nodes[lnodes_in]
      subcell_polys[n_subcells] = p_in
      subcell_to_inout[n_subcells] = IN
      subcell_to_cell[n_subcells] = cell

      n_subcells += 1
      subcell_nodes[n_subcells] = nodes[lnodes_out]
      subcell_polys[n_subcells] = p_out
      subcell_to_inout[n_subcells] = OUT
      subcell_to_cell[n_subcells] = cell
    end
  end
  resize!(subcell_nodes,n_subcells)
  resize!(subcell_polys,n_subcells)
  resize!(subcell_to_inout,n_subcells)
  resize!(subcell_to_cell,n_subcells)
  resize!(vertex_coordinates,n_nodes)
  
  subcell_nodes = Table(subcell_nodes)
  ptopo = Geometry.PolytopalGridTopology(vertex_coordinates,subcell_nodes,subcell_polys)
  return ptopo, subcell_to_inout, subcell_to_cell
end

function cut_conforming(model::UnstructuredDiscreteModel{D}, cell_values) where D
  ptopo, subcell_to_inout, subcell_to_cell = cut_conforming(get_grid_topology(model),cell_values)
  pgrid = Geometry.PolytopalGrid(
    Geometry.get_vertex_coordinates(ptopo),
    Geometry.get_faces(ptopo,D,0),
    get_polytopes(ptopo)
  )
  plabels = FaceLabeling(ptopo)
  pmodel = Geometry.PolytopalDiscreteModel(pgrid,ptopo,plabels)
  return pmodel, subcell_to_inout, subcell_to_cell
end

function get_isolated_volumes_mask_polytopal(
  model,cell_values,dirichlet_tags
)
  scmodel, cell_to_inout, cell_to_bgcell = cut_conforming(model,cell_values)
  cell_to_color, color_to_group = GridapTopOpt.tag_disconnected_volumes(scmodel,cell_to_inout;groups=(IN,OUT))

  bgcell_to_color_IN = GridapTopOpt.project_colors(model,cell_to_bgcell,cell_to_color,color_to_group,1)
  color_to_isolated_IN = GridapTopOpt.find_isolated_volumes(model,dirichlet_tags,bgcell_to_color_IN,color_to_group)
  data_IN = map(c -> color_to_isolated_IN[c] && isone(color_to_group[c]), bgcell_to_color_IN)
  cf_IN = CellField(collect(Float64,data_IN),Triangulation(model))

  bgcell_to_color_OUT = GridapTopOpt.project_colors(model,cell_to_bgcell,cell_to_color,color_to_group,2)
  color_to_isolated_OUT = GridapTopOpt.find_isolated_volumes(model,dirichlet_tags,bgcell_to_color_OUT,color_to_group)
  data_OUT = map(c -> color_to_isolated_OUT[c] && !isone(color_to_group[c]), bgcell_to_color_OUT)
  cf_OUT = CellField(collect(Float64,data_OUT),Triangulation(model))

  return cf_IN, cf_OUT
end

n = 20
#model = simplexify(CartesianDiscreteModel((0,1,0,1),(n,n)))
model = simplexify(CartesianDiscreteModel((0,1,0,1,0,1),(n,n,n)))

reffe = ReferenceFE(lagrangian,Float64,1)
V = TestFESpace(model,reffe)
# φh = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-0.35,V) # Circle
φh = interpolate(x->cos(2π*x[1])*cos(2π*x[2])*cos(2π*x[3])-0.11,V)
#φh = interpolate(x->cos(2π*x[1])*cos(2π*x[2])-0.11,V)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

cell_values = get_cell_dof_values(φh)
pmodel, subcell_to_inout, subcell_to_cell = cut_conforming(model,cell_values)

cf_IN, cf_OUT = get_isolated_volumes_mask_polytopal(
  model,cell_values,["boundary"]
)

writevtk(
  Triangulation(model),"results/polymodel",
  cellfields=[
    "φh"=>φh,
    "cf_IN"=>cf_IN,
    "cf_OUT"=>cf_OUT,
  ],
  append=false
)

writevtk(pmodel,"results/in_model";append=false)

in_model = Geometry.restrict(pmodel,findall(subcell_to_inout .== IN))
writevtk(in_model,"results/in_model";append=false)
