
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

function compact!(p)
  ids = findall(i->!isactive(p,i),1:num_vertices(p))
  old_to_new = fill(UNSET,num_vertices(p))
  new = 0
  for old in 1:num_vertices(p)
    isactive(p,old) || continue
    new += 1
    old_to_new[old] = new
  end
  vertices = get_vertex_coordinates(p)
  graph = get_graph(p)
  deleteat!(vertices,ids)
  deleteat!(graph,ids)
  f(i) = ifelse(i ∈ (OPEN,UNSET), i, old_to_new[i])
  map!(i->map!(f,i,i),graph,graph)
  new_to_old = find_inverse_index_map(old_to_new,new)
  return p, new_to_old
end

function split_postprocess!(graph,vertices,input_poly,values,(≶))
  complete_graph!(graph,num_vertices(input_poly))
  disconnect_graph!(graph,num_vertices(input_poly),values,(≶))
  poly = GeneralPolytope{num_cell_dims(input_poly)}(copy(vertices),graph)
  compact!(poly)
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
  
  subcell_nodes = Vector{Int}[]
  subcell_polys = GeneralPolytope{D,D,Float64,Nothing}[]
  vertex_coordinates = copy(Geometry.get_vertex_coordinates(topo))
  subcell_to_inout = Int8[]
  subcell_to_cell = Int[]
  
  n_nodes = num_faces(topo,0)
  edge_to_new_node = zeros(Int,num_faces(topo,1))
  for (cell,(nodes,values)) in enumerate(zip(cell_nodes,cell_values))
    iscut = any(v->v<0,values) && any(v->v>0,values)
  
    vertices = vertex_coordinates[nodes]
    p = GeneralPolytope{D}(p_ref,vertices)
    new_nodes = nodes[node_reindex]
    if !iscut
      push!(subcell_nodes,nodes)
      push!(subcell_polys,p)
      push!(subcell_to_inout,ifelse(all(v->v<0,values),IN,OUT))
      push!(subcell_to_cell,cell)
    else
      (p_in, lnodes_in), (p_out, lnodes_out), extra_vertices, ledges = split(p,values)
      edges = cell_edges[cell][edge_reindex][ledges]
      
      for (e,v) in zip(edges,extra_vertices)
        if iszero(edge_to_new_node[e])
          push!(vertex_coordinates,v)
          edge_to_new_node[e] = n_nodes + 1
          n_nodes += 1
        else
          v_ref = vertex_coordinates[edge_to_new_node[e]]
          @assert round(v;digits=10) ≈ round(v_ref;digits=10) "v_ref = $v_ref, v = $v"
        end
      end
      new_nodes = [nodes...,edge_to_new_node[edges]...]
  
      push!(subcell_nodes,new_nodes[lnodes_in])
      push!(subcell_polys,p_in)
      push!(subcell_to_inout,IN)
      push!(subcell_to_cell,cell)
  
      push!(subcell_nodes,new_nodes[lnodes_out])
      push!(subcell_polys,p_out)
      push!(subcell_to_inout,OUT)
      push!(subcell_to_cell,cell)
    end
  end
  
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

n = 52
model = simplexify(CartesianDiscreteModel((0,1,0,1),(n,n)))
#model = simplexify(CartesianDiscreteModel((0,1,0,1,0,1),(n,n,n)))

reffe = ReferenceFE(lagrangian,Float64,1)
V = TestFESpace(model,reffe)
# φh = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-0.35,V) # Circle
# φh = interpolate(x->cos(2π*x[1])*cos(2π*x[2])*cos(2π*x[3])-0.11,V)
φh = interpolate(x->cos(2π*x[1])*cos(2π*x[2])-0.11,V)

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
