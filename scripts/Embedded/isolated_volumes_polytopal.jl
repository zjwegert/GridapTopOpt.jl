
using Gridap
using Gridap.Geometry, Gridap.Arrays, Gridap.TensorValues, Gridap.Helpers
using Gridap.ReferenceFEs
using Gridap.ReferenceFEs: get_graph, isactive

using STLCutters
using STLCutters: complete_graph!, disconnect_graph!, add_open_vertices!, OPEN

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
  poly = Polyhedron(copy(vertices),graph)
  compact!(poly)
end

function split(p::GeneralPolytope{D},vertex_values) where D
  is_in(v) = v < 0
  is_out(v) = v > 0
  interpolate_values(v1,v2,w1,w2) = v1 + w1/(w1+w2)*(v2-v1)

  all(is_in, vertex_values) && return p, nothing
  all(is_out, vertex_values) && return nothing, p

  graph = get_graph(p)
  edge_nodes = get_faces(p,1,0)

  vertices = copy(get_vertex_coordinates(p))
  edges = Int[]
  in_graph = deepcopy(graph)
  out_graph = deepcopy(graph)
  
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
  p_in, p_out, new_vertices, edges
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
  subcell_to_isin = Bool[]
  
  n_nodes = num_faces(topo,0)
  edge_to_new_node = zeros(Int,num_faces(topo,1))
  for (cell,(nodes,values)) in enumerate(zip(cell_nodes,cell_values))
    iscut = any(v->v<0,values) && any(v->v>0,values)
  
    vertices = vertex_coordinates[nodes]
    p = Polyhedron(p_ref,vertices)
    new_nodes = nodes[node_reindex]
    if !iscut
      push!(subcell_nodes,nodes)
      push!(subcell_polys,p)
      push!(subcell_to_isin,all(v->v<0,values))
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
      push!(subcell_to_isin,false)
  
      push!(subcell_nodes,new_nodes[lnodes_out])
      push!(subcell_polys,p_out)
      push!(subcell_to_isin,true)
    end
  end
  
  subcell_nodes = Table(subcell_nodes)
  ptopo = Geometry.PolytopalGridTopology(vertex_coordinates,subcell_nodes,subcell_polys)
  return ptopo, subcell_to_isin
end

function cut_conforming(model::UnstructuredDiscreteModel{D}, cell_values) where D
  ptopo, subcell_to_isin = cut_conforming(get_grid_topology(model),cell_values)
  pgrid = Geometry.PolytopalGrid(
    Geometry.get_vertex_coordinates(ptopo),
    Geometry.get_faces(ptopo,D,0),
    get_polytopes(ptopo)
  )
  plabels = FaceLabeling(ptopo)
  pmodel = Geometry.PolytopalDiscreteModel(pgrid,ptopo,plabels)
  return pmodel, subcell_to_isin
end

n = 32
model = simplexify(CartesianDiscreteModel((0,1,0,1,0,1),(n,n,n)))

reffe = ReferenceFE(lagrangian,Float64,1)
V = TestFESpace(model,reffe)
#φh = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-0.35,V) # Circle

φh = interpolate(x->cos(2π*x[1])*cos(2π*x[2])*cos(2π*x[3])-0.11,V)
cell_values = get_cell_dof_values(φh)

pmodel, subcell_to_isin = cut_conforming(model,cell_values)

writevtk(pmodel,"results/polymodel";append=false)

in_model = Geometry.restrict(pmodel,findall(subcell_to_isin))
writevtk(in_model,"results/in_model";append=false)


