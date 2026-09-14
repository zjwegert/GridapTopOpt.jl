# The idea of this structure is that we like having a binary tree to store data and cache data
# but we also don't want to recreate the tree or resize it. So instead, we preallocate the tree to
# a depth that is equal to the number of level sets with with data for vertices, graphs, and inoutdata
# preallocated using the worst-case upper bound for the background mesh polytope + number of level sets.
#
# We then introduce a touched boolean on each node that is set to true when the data is manipulated.
# Once we've done our operations, on the tree we collect the leaf data into a preallocated
# array.
#
# Other than preallocation, the above is allocation free.
mutable struct TouchedBinaryNode{T}
  data::T
  parent::Union{Nothing,TouchedBinaryNode{T}}
  left::Union{Nothing,TouchedBinaryNode{T}}
  right::Union{Nothing,TouchedBinaryNode{T}}
  touched::Bool

  function TouchedBinaryNode{T}(data, parent=nothing, l=nothing, r=nothing, t=false) where T
    new{T}(data, parent, l, r, t)
  end
end
TouchedBinaryNode(data) = TouchedBinaryNode{typeof(data)}(data)

function leftchild!(parent::TouchedBinaryNode, data)
  @check isnothing(parent.left) "left child is already assigned"
  Node = typeof(parent)(data, parent)
  parent.left = Node
end
function rightchild!(parent::TouchedBinaryNode, data)
  @check isnothing(parent.right) "right child is already assigned"
  Node = typeof(parent)(data, parent)
  parent.right = Node
end

## Things we need to define
function AbstractTrees.children(Node::TouchedBinaryNode)
  if isnothing(Node.left) && isnothing(Node.right)
    ()
  elseif isnothing(Node.left) && !isnothing(Node.right)
    (Node.right,)
  elseif !isnothing(Node.left) && isnothing(Node.right)
    (Node.left,)
  else
    (Node.left, Node.right)
  end
end

AbstractTrees.nodevalue(n::TouchedBinaryNode) = n.data

AbstractTrees.ParentLinks(::Type{<:TouchedBinaryNode}) = StoredParents()

AbstractTrees.parent(n::TouchedBinaryNode) = n.parent

AbstractTrees.NodeType(::Type{<:TouchedBinaryNode{T}}) where {T} = HasNodeType()
AbstractTrees.nodetype(::Type{<:TouchedBinaryNode{T}}) where {T} = TouchedBinaryNode{T}

# Update touched when setting data, this is a convinence function to allow for
# setdata!(n) do
#   Manipulate n.data
# end
function setdata!(f!, n::TouchedBinaryNode)
  n.touched = true
  f!()
  nothing
end

# Create a complete binary tree of given depth with each node initialised using a deepcopy
# of the input data. The data represents the true data plus the cache for each node.
# Note, a depth of zero is simply a root node.
function preallocate_tree(data, depth::Int)
  depth < 0 && error("depth must be non-negative")
  root = TouchedBinaryNode(deepcopy(data))
  if depth == 0
    return root
  end
  function _preallocate!(node::TouchedBinaryNode, current_depth::Int)
    current_depth == depth && return
    leftchild!(node, deepcopy(data))
    rightchild!(node, deepcopy(data))
    _preallocate!(node.left, current_depth + 1)
    _preallocate!(node.right, current_depth + 1)
  end
  _preallocate!(root, 0)
  return root
end

# An allocation-free get leaves function. As this is recrusive, we cache
# a vector that keeps count of how many times we've set the leaf_data cache.
function get_leaves!(cache, node::TouchedBinaryNode)
  leaf_data, i_vec = cache
  if node.touched && (isnothing(node.left) || !node.left.touched) &&
      (isnothing(node.right) || !node.right.touched)
    i_vec[1] += 1
    leaf_data[i_vec[1]] = node.data
    return
  end
  !(isnothing(node.left) || !node.left.touched) && get_leaves!(cache, node.left)
  !(isnothing(node.right) || !node.right.touched) && get_leaves!(cache, node.right)
  nothing
end

# Instead of getting leaves, we apply a map f! over all the leaves
# Output should be stored in data.
function map_leaves!(f!::Function,data,node::TouchedBinaryNode)
  if node.touched && (isnothing(node.left) || !node.left.touched) &&
      (isnothing(node.right) || !node.right.touched)
    f!(data,node)
    return
  end
  !(isnothing(node.left) || !node.left.touched) && map_leaves!(f!,data,node.left)
  !(isnothing(node.right) || !node.right.touched) && map_leaves!(f!,data,node.right)
  nothing
end

# Reset touched flags
function reset_touched!(node::TouchedBinaryNode)
  node.touched = false
  !isnothing(node.left) && reset_touched!(node.left)
  !isnothing(node.right) && reset_touched!(node.right)
  nothing
end