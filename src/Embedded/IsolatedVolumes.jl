

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
    function tag_isolated_volumes(
        model::DiscreteModel{Dc},
        cell_to_state::Vector{<:Integer};
        groups = Tuple(unique(cell_to_state))
    )

Given a DiscreteModel `model` and an initial coloring `cell_to_state`,
returns another coloring such that each color corresponds to a connected component of the
graph of cells that are connected by a face and have their state in the same group.
"""
function tag_isolated_volumes(
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

"""
    get_isolated_volumes_mask(cutgeo::EmbeddedDiscretization,dirichlet_tags)

Given an EmbeddedDiscretization `cutgeo` and a list of tags `dirichlet_tags`,
this function returns a CellField which is `1` on isolated volumes and `0` otherwise.

We define an isolated volume as a volume that is IN but is not constrained by any
of the tags in `dirichlet_tags`.

Specify the in domain using `IN_is`. Taking `IN_is = OUT` will find isolated
volumes for the `OUT` domain.
"""
function get_isolated_volumes_mask(
  cutgeo::EmbeddedDiscretization,dirichlet_tags;
  IN_is = IN
)
  model = get_background_model(cutgeo)
  geo = get_geometry(cutgeo)

  bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo)
  # Note: We switch IN and OUT based on the IN_is parameter. Namely,
  #   IN_is = IN  -> groups = ((CUT,IN),IN*IN) = ((CUT,IN),OUT)
  #   IN_is = OUT -> groups = ((CUT,OUT),IN*OUT) = ((CUT,IN),IN)
  cell_to_color, color_to_group = tag_isolated_volumes(model,bgcell_to_inoutcut;groups=((CUT,IN_is),IN*IN_is))
  color_to_tagged = find_tagged_volumes(model,dirichlet_tags,cell_to_color,color_to_group)

  data = map(c -> !color_to_tagged[c] && isone(color_to_group[c]), cell_to_color)
  return CellField(collect(Float64,data),Triangulation(model))
end

"""
    get_isolated_volumes_mask_without_cuts(cutgeo::EmbeddedDiscretization,dirichlet_tags)

The same as `get_isolated_volumes_mask` but we do not mark CUT cells.
"""
function get_isolated_volumes_mask_without_cuts(
  cutgeo::EmbeddedDiscretization,dirichlet_tags;
  IN_is = IN
)
  model = get_background_model(cutgeo)
  geo = get_geometry(cutgeo)

  bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo)
  groups=(IN_is,(CUT,IN*IN_is))
  cell_to_color, color_to_group = tag_isolated_volumes(model,bgcell_to_inoutcut;groups)
  color_to_tagged = find_tagged_volumes(model,dirichlet_tags,cell_to_color,color_to_group)

  data = map(c -> !color_to_tagged[c] && isone(color_to_group[c]), cell_to_color)
  return CellField(collect(Float64,data),Triangulation(model))
end

# Distributed

function tag_isolated_volumes(
  model::GridapDistributed.DistributedDiscreteModel{Dc},
  cell_to_state::AbstractVector{<:Vector{<:Integer}};
  groups::Tuple
) where Dc

  cell_to_lcolor, lcolor_to_group = map(local_views(model),cell_to_state) do model, cell_to_state
    tag_isolated_volumes(model,cell_to_state;groups)
  end |> tuple_of_arrays

  cell_ids = partition(get_cell_gids(model))
  n_lcolor = map(length,lcolor_to_group)
  color_gids = generate_volume_gids(cell_ids, n_lcolor, cell_to_lcolor)

  return cell_to_lcolor, lcolor_to_group, color_gids
end

function get_isolated_volumes_mask(
  cutgeo::GridapEmbedded.Distributed.DistributedEmbeddedDiscretization,dirichlet_tags;
  IN_is = IN
)
  model = get_background_model(cutgeo)
  geo = get_geometry(cutgeo)

  bgcell_to_inoutcut = map(compute_bgcell_to_inoutcut,local_views(model),local_views(geo))
  cell_to_lcolor, lcolor_to_group, color_gids = tag_isolated_volumes(model,bgcell_to_inoutcut;groups=((CUT,IN_is),IN*IN_is))

  lcolor_to_tagged = map(local_views(model),cell_to_lcolor,lcolor_to_group) do model, cell_to_lcolor, lcolor_to_group
    find_tagged_volumes(model,dirichlet_tags,cell_to_lcolor,lcolor_to_group)
  end
  aux = PVector(lcolor_to_tagged,partition(color_gids))
  assemble!(|,aux) |> wait
  consistent!(aux) |> wait

  trian = Triangulation(GridapDistributed.WithGhost(),model)
  fields = map(local_views(trian),cell_to_lcolor,lcolor_to_group,lcolor_to_tagged) do trian, cell_to_lcolor, lcolor_to_group, lcolor_to_tagged
    data = map(c -> !lcolor_to_tagged[c] && isone(lcolor_to_group[c]), cell_to_lcolor)
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
  cell_ids, n_lcolor, cell_to_lcolor
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

