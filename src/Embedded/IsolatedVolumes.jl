

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

function find_unique_points(
  point_to_coords::AbstractVector
)
  f(pt::Number) = round(pt;sigdigits=12)
  f(id::Integer) = f(point_to_coords[id])

  ids = Base.OneTo(length(point_to_coords))
  upoints = unique(f,ids)
  return collect(Int32,indexin(f.(ids),f.(upoints))), point_to_coords[upoints]
end

function generate_subcell_topology(cutgeo)
  model = get_background_model(cutgeo)

  bgcell_to_inoutcut = GridapEmbedded.Interfaces.compute_bgcell_to_inoutcut(cutgeo,cutgeo.geo)
  subcell_to_inout = GridapEmbedded.Interfaces.compute_subcell_to_inout(cutgeo,cutgeo.geo)
  cell_to_bgcell = findall(!iszero,bgcell_to_inoutcut)

  n_bgcells = length(cell_to_bgcell)
  n_subcells = length(subcell_to_inout)
  n_cells = n_bgcells + n_subcells

  cell_to_inout = vcat(bgcell_to_inoutcut[cell_to_bgcell],subcell_to_inout)

  Dc = num_cell_dims(model)
  topo = get_grid_topology(model)
  n_bgnodes = num_faces(topo, 0)
  bgcell_to_bgnode = Geometry.get_faces(topo, Dc, 0)
  subcell_to_bgnode = cutgeo.subcells.cell_to_points

  point_to_upoint, ucoords = find_unique_points(
    lazy_append(Geometry.get_vertex_coordinates(topo), cutgeo.subcells.point_to_coords)
  )

  data = Vector{Int32}(undef,length(bgcell_to_bgnode.data)+length(subcell_to_bgnode.data))
  ptrs = Vector{Int32}(undef,n_cells+1)
  ptrs[1] = 1
  cell = 1
  for bgcell in cell_to_bgcell
    nodes = view(point_to_upoint,view(bgcell_to_bgnode,bgcell))
    ptrs[cell+1] = ptrs[cell] + length(nodes)
    data[ptrs[cell]:ptrs[cell+1]-1] .= nodes
    cell += 1
  end
  for subcell in 1:n_subcells
    nodes = view(point_to_upoint,view(subcell_to_bgnode,subcell) .+ n_bgnodes)
    ptrs[cell+1] = ptrs[cell] + length(nodes)
    data[ptrs[cell]:ptrs[cell+1]-1] .= nodes
    cell += 1
  end
  resize!(data,ptrs[end]-1)

  cell_to_nodes = Table(data,ptrs)

  polys = [get_polytopes(model)...,TRI]
  cell_types = collect(Int8,lazy_append(get_cell_type(topo)[cell_to_bgcell],Fill(Int8(length(polys)),n_subcells)))

  new_topo = UnstructuredGridTopology(ucoords,cell_to_nodes,cell_types,polys)
  cell_to_bgcell = collect(lazy_append(cell_to_bgcell,cutgeo.subcells.cell_to_bgcell))
  return new_topo, cell_to_inout, cell_to_bgcell
end

function generate_subcell_model(cutgeo)
  topo, cell_to_inout, cell_to_bgcell = GridapTopOpt.generate_subcell_topology(cutgeo)
  grid = UnstructuredGrid(
    Geometry.get_vertex_coordinates(topo), Geometry.get_faces(topo,2,0),
    map(p -> Gridap.ReferenceFEs.LagrangianRefFE(Float64,p,1), get_polytopes(topo)),
    get_cell_type(topo)
  )
  model = UnstructuredDiscreteModel(grid,topo,FaceLabeling(topo))
  return model, cell_to_inout, cell_to_bgcell
end

function project_colors(
  model,cell_to_bgcell,cell_to_color,color_to_group,target
)
  bgcell_to_color = zeros(Int16,num_cells(model))
  for (cell,bgcell) in enumerate(cell_to_bgcell)
    color = cell_to_color[cell]
    group = color_to_group[color]
    if (group == target) || iszero(bgcell_to_color[bgcell])
      bgcell_to_color[bgcell] = color
    end
  end
  return bgcell_to_color
end

function get_isolated_volumes_mask_v2(
  cutgeo::EmbeddedDiscretization,dirichlet_tags
)
  model = get_background_model(cutgeo)

  scmodel, cell_to_inout, cell_to_bgcell = generate_subcell_model(cutgeo)
  cell_to_color, color_to_group = tag_disconnected_volumes(scmodel,cell_to_inout;groups=(IN,OUT))

  bgcell_to_color_IN = project_colors(model,cell_to_bgcell,cell_to_color,color_to_group,1)
  color_to_isolated_IN = find_isolated_volumes(model,dirichlet_tags,bgcell_to_color_IN,color_to_group)
  data_IN = map(c -> color_to_isolated_IN[c] && isone(color_to_group[c]), bgcell_to_color_IN)
  cf_IN = CellField(collect(Float64,data_IN),Triangulation(model))

  bgcell_to_color_OUT = project_colors(model,cell_to_bgcell,cell_to_color,color_to_group,2)
  color_to_isolated_OUT = find_isolated_volumes(model,dirichlet_tags,bgcell_to_color_OUT,color_to_group)
  data_OUT = map(c -> color_to_isolated_OUT[c] && !isone(color_to_group[c]), bgcell_to_color_OUT)
  cf_OUT = CellField(collect(Float64,data_OUT),Triangulation(model))

  return cf_IN, cf_OUT
end

"""
    function tag_volume!(
        cell::Int,color::Int16,group::Union{Integer,NTuple{N,Integer}},
        cell_to_nbors::Vector{Vector{Int32}},
        cell_to_state::Vector{Int8},
        cell_to_color::Vector{Int16},
        touched::BitVector
    )

Starting from a cell `cell`, crawls the cell graph provided by `cell_to_nbors` and colors all cells
connected to `cell` that
  - belong to the group `group` (i.e., `cell_to_state[cell] ∈ group`), and
  - have not been seen yet (i.e., `!touched[cell]`).

This is done by using a breadth-first search algorithm.
"""
function tag_volume!(
  cell::Int,color::Int16,group::Union{Integer,NTuple{N,Integer}},
  cell_to_nbors::Vector{Vector{Int32}},
  cell_to_state::Vector{Int8},
  cell_to_color::Vector{Int16},
  touched::BitVector
) where N
  @assert cell_to_state[cell] ∈ group

  q = Queue{Int}()
  enqueue!(q,cell)
  touched[cell] = true

  while !isempty(q)
    cell = dequeue!(q)
    cell_to_color[cell] = color

    nbors = cell_to_nbors[cell]
    for nbor in nbors
      state = cell_to_state[nbor]
      if !touched[nbor] && state ∈ group
        enqueue!(q,nbor)
        touched[nbor] = true
      end
    end
  end
end

"""
    function tag_disconnected_volumes(
        model::DiscreteModel{Dc},
        cell_to_state::Vector{<:Integer};
        groups = Tuple(unique(cell_to_state))
    )

Given a DiscreteModel `model` and an initial coloring `cell_to_state`,
returns another coloring such that each color corresponds to a connected component of the
graph of cells that are connected by a face and have their state in the same group.
"""
function tag_disconnected_volumes(
  model::DiscreteModel{Dc},
  cell_to_state::Vector{<:Integer};
  groups = Tuple(unique(cell_to_state))
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
      state = cell_to_state[cell]
      group = findfirst(g -> state ∈ g, groups)
      push!(color_to_group, group)
      tag_volume!(cell, n_color, groups[group], cell_to_nbors, cell_to_state, cell_to_color, touched)
    end
    cell += 1
  end

  return cell_to_color, color_to_group
end

function find_isolated_volumes(
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

  is_isolated = map(!, is_tagged)
  return is_isolated
end

function find_onlycut_volumes(
  model::DiscreteModel,
  cell_to_color::Vector{Int16},
  cell_to_state::Vector{Int8},
  color_to_group::Vector{Int8};
)
  onlycut = trues(length(color_to_group))
  for cell in 1:num_cells(model)
    color = cell_to_color[cell]
    if cell_to_state[cell] != CUT
      onlycut[color] = false
    end
  end
  return onlycut
end

"""
    get_isolated_volumes_mask(cutgeo::EmbeddedDiscretization,dirichlet_tags)

Given an EmbeddedDiscretization `cutgeo` and a list of tags `dirichlet_tags`,
this function returns a CellField which is `1` on isolated volumes and `0` otherwise.

We define an isolated volume as a volume that is IN but is not constrained by any
of the tags in `dirichlet_tags`. Specify the In domain using the first entry in groups.

If `remove_cuts` is `true`, then volumes that only contain CUT cells are also considered isolated.
"""
function get_isolated_volumes_mask(
  cutgeo::EmbeddedDiscretization,dirichlet_tags;
  groups = ((CUT,IN),OUT),
  remove_cuts = true
)
  model = get_background_model(cutgeo)
  geo = get_geometry(cutgeo)

  bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo)
  cell_to_color, color_to_group = tag_disconnected_volumes(model,bgcell_to_inoutcut;groups)
  color_to_isolated = find_isolated_volumes(model,dirichlet_tags,cell_to_color,color_to_group)

  if remove_cuts
    color_to_onlycut = find_onlycut_volumes(model,cell_to_color,bgcell_to_inoutcut,color_to_group)
    color_to_isolated .= color_to_isolated .| color_to_onlycut
  end

  data = map(c -> color_to_isolated[c] && isone(color_to_group[c]), cell_to_color)
  return CellField(collect(Float64,data),Triangulation(model))
end

# Distributed

function tag_disconnected_volumes(
  model::GridapDistributed.DistributedDiscreteModel{Dc},
  cell_to_state::AbstractVector{<:Vector{<:Integer}};
  groups::Tuple
) where Dc

  _cell_to_lcolor, _lcolor_to_group = map(local_views(model),cell_to_state) do model, cell_to_state
    tag_disconnected_volumes(model,cell_to_state;groups)
  end |> tuple_of_arrays

  cell_ids = partition(get_cell_gids(model))
  cell_to_lcolor, lcolor_to_group, color_gids = generate_volume_gids(cell_ids, _cell_to_lcolor, _lcolor_to_group)

  return cell_to_lcolor, lcolor_to_group, color_gids
end

function get_isolated_volumes_mask(
  cutgeo::GridapEmbedded.Distributed.DistributedEmbeddedDiscretization,dirichlet_tags;
  groups = ((CUT,IN),OUT),
  remove_cuts = true
)
  function consistent_vols!(lcolor_to_mask, color_gids)
    aux = PVector(lcolor_to_mask,partition(color_gids))
    assemble!(&,aux) |> wait
    consistent!(aux) |> wait
  end

  model = get_background_model(cutgeo)
  geo = get_geometry(cutgeo)

  bgcell_to_inoutcut = map(compute_bgcell_to_inoutcut,local_views(model),local_views(geo))
  cell_to_lcolor, lcolor_to_group, color_gids = tag_disconnected_volumes(model,bgcell_to_inoutcut;groups)

  lcolor_to_isolated = map(local_views(model),cell_to_lcolor,lcolor_to_group) do model, cell_to_lcolor, lcolor_to_group
    find_isolated_volumes(model,dirichlet_tags,cell_to_lcolor,lcolor_to_group)
  end
  consistent_vols!(lcolor_to_isolated,color_gids)

  if remove_cuts
    lcolor_to_onlycut = map(local_views(model),cell_to_lcolor,bgcell_to_inoutcut,lcolor_to_group) do model, cell_to_lcolor, bgcell_to_inoutcut, lcolor_to_group
      find_onlycut_volumes(model,cell_to_lcolor,bgcell_to_inoutcut,lcolor_to_group)
    end
    consistent_vols!(lcolor_to_onlycut,color_gids)
    map(lcolor_to_isolated,lcolor_to_onlycut) do isolated, onlycut
      isolated .= isolated .| onlycut
    end
  end

  trian = Triangulation(GridapDistributed.WithGhost(),model)
  fields = map(local_views(trian),cell_to_lcolor,lcolor_to_group,lcolor_to_isolated) do trian, cell_to_lcolor, lcolor_to_group, lcolor_to_isolated
    data = map(c -> lcolor_to_isolated[c] && isone(lcolor_to_group[c]), cell_to_lcolor)
    CellField(collect(Float64,data),trian)
  end

  return GridapDistributed.DistributedCellField(fields,trian)
end

# Given a vector_partition and indices, returns an exchange cache that
# exchanges both the ghost layer and the ghost layer of the neighbors
function double_layer_exchange_cache(vector_partition,index_partition)
  function f(a::JaggedArray,b::JaggedArray)
    JaggedArray(map(vcat,a,b))
  end
  neighbors_snd, neighbors_rcv = assembly_neighbors(index_partition)
  sl_indices_snd, sl_indices_rcv = assembly_local_indices(index_partition,neighbors_snd,neighbors_rcv)
  dl_indices_snd = map(f, sl_indices_snd, sl_indices_rcv)
  dl_indices_rcv = map(f, sl_indices_rcv, sl_indices_snd)
  buffers_snd, buffers_rcv = map(PartitionedArrays.assembly_buffers,vector_partition,dl_indices_snd,dl_indices_rcv) |> tuple_of_arrays
  map(PartitionedArrays.VectorAssemblyCache,neighbors_snd,neighbors_rcv,dl_indices_snd,dl_indices_rcv,buffers_snd,buffers_rcv)
end

function generate_volume_gids(
  cell_ids, cell_to_lcolor, lcolor_to_group
)
  # NOTE: We have to communicate both the snd and rcv layers, i.e
  # each processor sends and receives both
  #   1) their layer of ghost cells
  #   2) the layer of ghost cells of their neighbors
  # Otherwise, the constructed graph is non-symmetric in cases where a volume ends just
  # at the boundary of a processor. If the graph is not symmetric, we can end up with
  # false volumes...
  exchange_caches = double_layer_exchange_cache(cell_to_lcolor,cell_ids)
  # exchange_caches = PartitionedArrays.p_vector_cache_impl(Vector,cell_to_lcolor,cell_ids)

  # Send and receive local information from neighbors
  neighbors_snd, neighbors_rcv, buffer_snd, buffer_rcv = map(exchange_caches,cell_to_lcolor) do cache, cell_to_lcolor
    for (k,lid) in enumerate(cache.local_indices_snd)
      cache.buffer_snd[k] = cell_to_lcolor[lid]
    end
    cache.neighbors_snd,cache.neighbors_rcv,cache.buffer_snd,cache.buffer_rcv
  end |> tuple_of_arrays;

  graph = ExchangeGraph(neighbors_snd,neighbors_rcv)
  t = exchange!(buffer_rcv,buffer_snd,graph)
  wait(t)

  # Prepare local information
  # For each local volume (color), collect:
  #   1. All the neighbors touching the volume
  #   2. The local color of the volume in the neighbor
  n_lcolor = map(length,lcolor_to_group)
  ptrs, lcolor_to_nbors, lcolor_to_nbor_lcolor = map(
    n_lcolor, exchange_caches, cell_to_lcolor
  ) do n_lcolor, cache, cell_to_lcolor

    ptrs = zeros(Int16,n_lcolor+1)
    lcolor_to_nbors = [Int[] for _ in 1:n_lcolor]
    lcolor_to_nbor_lcolors = [Int16[] for _ in 1:n_lcolor]

    for (lnbor,nbor) in enumerate(cache.neighbors_rcv)
      seen = Set{Tuple{Int16,Int16}}()
      for k in cache.buffer_rcv.ptrs[lnbor]:cache.buffer_rcv.ptrs[lnbor+1]-1
        cell = cache.local_indices_rcv.data[k]
        nbor_lcolor = cache.buffer_rcv.data[k]
        lcolor = cell_to_lcolor[cell]
        if (lcolor,nbor_lcolor) ∉ seen
          push!(lcolor_to_nbors[lcolor],nbor)
          push!(lcolor_to_nbor_lcolors[lcolor],nbor_lcolor)
          push!(seen,(lcolor,nbor_lcolor))
          ptrs[lcolor+1] += 1
        end
      end
    end

    PartitionedArrays.length_to_ptrs!(ptrs)
    return ptrs, vcat(lcolor_to_nbors...), vcat(lcolor_to_nbor_lcolors...)
  end |> tuple_of_arrays

  # Gather local information in MAIN
  ptrs = gather(ptrs)
  lcolor_to_nbors = gather(lcolor_to_nbors)
  lcolor_to_nbor_lcolor = gather(lcolor_to_nbor_lcolor)

  # Create global ordering in MAIN
  lcolor_to_color, lcolor_to_owner, n_color = map_main(
    ptrs, lcolor_to_nbors, lcolor_to_nbor_lcolor;
    otherwise = (args...) -> (JaggedArray([Int16[]]), JaggedArray([Int[]]), zero(Int16))
  ) do ptrs, lcolor_to_nbors, lcolor_to_nbor_lcolor

    n_procs = length(ptrs)
    n_lcolors = map(p -> length(p)-1, ptrs)

    lcolor_to_nbors = map(jagged_array,lcolor_to_nbors,ptrs)
    lcolor_to_nbor_lcolor = map(jagged_array,lcolor_to_nbor_lcolor,ptrs)
    lcolor_to_color = [zeros(Int16,n) for n in n_lcolors]
    color_to_owner = Int[]

    n_color = zero(Int16)
    for p in 1:n_procs
      for lcolor in 1:n_lcolors[p]
        if iszero(lcolor_to_color[p][lcolor]) # New volume found
          n_color += one(Int16)
          push!(color_to_owner,p)
          lcolor_to_color[p][lcolor] = n_color
          @debug ">> NEW VOLUME: $n_color"

          q = Queue{Tuple{Int,Int}}()
          enqueue!(q,(p,lcolor))
          while !isempty(q)
            proc, proc_lcolor = dequeue!(q)
            @debug "   >> PROC: $proc, PROC_LCOLOR: $proc_lcolor"
            for (nbor, nbor_lcolor) in zip(lcolor_to_nbors[proc][proc_lcolor],lcolor_to_nbor_lcolor[proc][proc_lcolor])
              nbor_color = lcolor_to_color[nbor][nbor_lcolor]
              if iszero(nbor_color)
                # nbor_lcolor has not been colored yet: color it
                lcolor_to_color[nbor][nbor_lcolor] = n_color
                enqueue!(q,(nbor,nbor_lcolor))
              else
                # nbor has been colored: check consistency
                @assert nbor_color == n_color
              end
            end
          end
        end
      end
    end

    lcolor_to_owner = map(c -> color_to_owner[c], lcolor_to_color)
    return JaggedArray(lcolor_to_color), JaggedArray(lcolor_to_owner), n_color
  end |> tuple_of_arrays

  # Scatter global color information
  lcolor_to_color = scatter(lcolor_to_color)
  lcolor_to_owner = scatter(lcolor_to_owner)
  n_color = emit(n_color)

  # Glue together local volumes that are split in two (i.e two lcolors with same color)
  # Otherwise, there might be two lcolors with the same global color, which causes issues 
  # for information exchange.
  lcolor_to_color, lcolor_to_owner, lcolor_to_group = map(
    lcolor_to_color,lcolor_to_owner,cell_to_lcolor,lcolor_to_group
  ) do lcolor_to_color, lcolor_to_owner, cell_to_lcolor, lcolor_to_group
    ulcolor_to_color = unique(lcolor_to_color)
    lcolor_to_ulcolor = collect(Int,indexin(lcolor_to_color,ulcolor_to_color))

    # Below could be done more concisely, but we want to check for consistency
    ulcolor_to_owner = zeros(eltype(lcolor_to_owner),length(ulcolor_to_color))
    ulcolor_to_group = zeros(eltype(lcolor_to_group),length(ulcolor_to_color))
    for (lc,ulc) in enumerate(lcolor_to_ulcolor)
      @assert iszero(ulcolor_to_owner[ulc]) || isequal(ulcolor_to_owner[ulc],lcolor_to_owner[lc])
      @assert iszero(ulcolor_to_group[ulc]) || isequal(ulcolor_to_group[ulc],lcolor_to_group[lc])
      ulcolor_to_owner[ulc] = lcolor_to_owner[lc]
      ulcolor_to_group[ulc] = lcolor_to_group[lc]
    end

    for cell in eachindex(cell_to_lcolor)
      cell_to_lcolor[cell] = lcolor_to_ulcolor[cell_to_lcolor[cell]]
    end

    return ulcolor_to_color, ulcolor_to_owner, ulcolor_to_group
  end |> tuple_of_arrays

  # Locally build color gids
  color_ids = map(cell_ids,lcolor_to_color,lcolor_to_owner,n_color) do ids, lcolor_to_color, lcolor_to_owner, n_color
    me = part_id(ids)
    LocalIndices(Int(n_color),me,collect(Int,lcolor_to_color),collect(Int32,lcolor_to_owner))
  end

  return cell_to_lcolor, lcolor_to_group, PRange(color_ids)
end

# I tried to be clever, but I think this fails when we have volumes that are locally split in two
# such that they neighbor different neighbors.
# I have given a lot of though, and I think there is just no way around communicating the whole
# graph...
"""
function generate_volume_gids(
  cell_ids, n_lcolor, cell_to_lcolor
)
  exchange_caches = PartitionedArrays.p_vector_cache_impl(Vector,cell_to_lcolor,cell_ids)

  # Send and receive local information from neighbors
  neighbors_snd, neighbors_rcv, buffer_snd, buffer_rcv = map(exchange_caches,cell_to_lcolor) do cache, cell_to_lcolor
    for (k,lid) in enumerate(cache.local_indices_snd)
      cache.buffer_snd[k] = cell_to_lcolor[lid]
    end
    cache.neighbors_snd,cache.neighbors_rcv,cache.buffer_snd,cache.buffer_rcv
  end |> tuple_of_arrays;

  graph = ExchangeGraph(neighbors_snd,neighbors_rcv)
  t = exchange!(buffer_rcv,buffer_snd,graph)
  wait(t)

  # Prepare local information
  # For each local volume (color), find two things:
  #   1. The smallest neighbor id sharing the volume
  #   2. The volume's lcolor in the selected neighbor
  lcolor_to_nbor, lcolor_to_nbor_lcolor = map(cell_ids,n_lcolor,exchange_caches,cell_to_lcolor) do ids, n_lcolor, cache, cell_to_lcolor
    lcolor_to_nbor = fill(part_id(ids),n_lcolor)
    lcolor_to_nbor_lcolor = collect(Int16,1:n_lcolor)

    for (p,nbor) in enumerate(cache.neighbors_rcv)
      for k in cache.buffer_rcv.ptrs[p]:cache.buffer_rcv.ptrs[p+1]-1
        lid = cache.local_indices_rcv.data[k]
        nbor_lcolor = cache.buffer_rcv.data[k]
        lcolor = cell_to_lcolor[lid]
        if nbor < lcolor_to_nbor[lcolor]
          lcolor_to_nbor[lcolor] = nbor
          lcolor_to_nbor_lcolor[lcolor] = nbor_lcolor
        end
      end
    end
    return lcolor_to_nbor, lcolor_to_nbor_lcolor
  end |> tuple_of_arrays;

  # Gather local information in MAIN
  lcolor_to_nbor = gather(lcolor_to_nbor)
  lcolor_to_nbor_lcolor = gather(lcolor_to_nbor_lcolor)

  # Create global ordering in MAIN
  lcolor_to_color, lcolor_to_owner, n_color = map_main(lcolor_to_nbor,lcolor_to_nbor_lcolor; otherwise = (args...) -> (nothing, nothing, zero(Int16))) do lcolor_to_nbor, lcolor_to_nbor_lcolor
    ptrs = lcolor_to_nbor.ptrs
    lcolor_to_color = JaggedArray(zeros(Int16,ptrs[end]-1),ptrs)
    color_to_owner = Int[]

    n_procs = length(lcolor_to_nbor)
    n_color = zero(Int16)
    for p in 1:n_procs
      seen = Dict{Tuple{Int,Int16},Int16}()
      for k in ptrs[p]:ptrs[p+1]-1
        nbor = lcolor_to_nbor.data[k]
        nbor_lcolor = lcolor_to_nbor_lcolor.data[k]
        if nbor < p
          # Volume has been seen before by a neighbor,
          # so we can just copy the color
          color = lcolor_to_color.data[ptrs[nbor]+nbor_lcolor-1]
        else
          # I own the volume
          @assert nbor == p
          # The following logic is to deal with volumes which are locally split
          # I.e the same volume has several local colors (since it is locally disconnected)
          if iszero(nbor_lcolor) || !haskey(seen,(nbor,nbor_lcolor))
            n_color += one(Int16)
            color = n_color
            push!(color_to_owner,p)
            seen[(nbor,nbor_lcolor)] = color
          else
            color = seen[(nbor,nbor_lcolor)]
          end
        end
        lcolor_to_color.data[k] = color
      end
    end
    lcolor_to_owner = JaggedArray(map(c -> color_to_owner[c], lcolor_to_color.data),ptrs)
    return lcolor_to_color, lcolor_to_owner, n_color
  end |> tuple_of_arrays;

  # Scatter global color information
  lcolor_to_color = scatter(lcolor_to_color)
  lcolor_to_owner = scatter(lcolor_to_owner)
  n_color = emit(n_color)

  # Locally build color gids
  color_ids = map(cell_ids,lcolor_to_color,lcolor_to_owner,n_color) do ids, lcolor_to_color, lcolor_to_owner, n_color
    me = part_id(ids)
    LocalIndices(Int(n_color),me,collect(Int,lcolor_to_color),collect(Int32,lcolor_to_owner))
  end

  return PRange(color_ids)
end
"""

