"""
    save(filename::AbstractString, x)

Save an object `x` to `filename` as a JLD2 file.

Note: To `save` in MPI mode, use [`psave`](@ref).
"""
function save(filename::AbstractString, x)
  save_object(filename,x)
end

"""
    load(filename::AbstractString)

Load an object stored in a JLD2 file at `filename`.

Note: To `load` in MPI mode, use [`pload`](@ref).
"""
function load(filename::AbstractString)
  load_object(filename)
end

"""
    load!(filename::AbstractString, x)

Load an object stored in a JLD2 file at `filename` and copy its
contents to `x`.

Note: To `load!` in MPI mode, use [`pload!`](@ref).
"""
function load!(filename::AbstractString, x)
  y = load(filename)
  copyto!(x,y)
  return x
end

"""
    psave(filename::AbstractString, x)

Save a partitioned object `x` to a directory `dir` as 
a set of JLD2 files corresponding to each part.
"""
function psave(dir::AbstractString, x)
  ranks = get_parts(x)
  i_am_main(ranks) && mkpath(dir)
  arr = to_local_storage(x)
  map(ranks,arr) do id, arr
    filename = joinpath(dir,basename(dir)*"_$id.jdl2")
    save_object(filename,arr)
  end
end

"""
    pload(dir::AbstractString, ranks::AbstractArray{<:Integer})

Load a partitioned object stored in a set of JLD2 files in directory `dir`
indexed by MPI ranks `ranks`.
"""
function pload(dir::AbstractString, ranks::AbstractArray{<:Integer})
  arr = map(ranks) do id
    filename = joinpath(dir,basename(dir)*"_$id.jdl2")
    load_object(filename)
  end
  return from_local_storage(arr)
end

"""
    pload!(dir::AbstractString, x)

Load a partitioned object stored in a set of JLD2 files in directory `dir`
and copy contents to the equivilent object `x`.
"""
function pload!(dir::AbstractString, x)
  ranks = get_parts(x)
  y = pload(dir,ranks)
  copyto!(x,y)
  return x
end

## Handle storage of values and indices for `PVector` and PSparseMatrix

function to_local_storage(x)
  x
end

function from_local_storage(x)
  x
end

struct PVectorLocalStorage{A,B}
  values::A
  indices::B
end

function from_local_storage(x::AbstractArray{<:PVectorLocalStorage})
  values, indices = map(x) do x
    x.values, x.indices
  end |> tuple_of_arrays
  return PVector(values,indices)
end

function to_local_storage(x::PVector)
  map(PVectorLocalStorage,partition(x),partition(axes(x,1)))
end

struct PSparseMatrixLocalStorage{A,B,C}
  values::A
  rows::B
  cols::C
end

function from_local_storage(x::AbstractArray{<:PSparseMatrixLocalStorage})
  values, rows, cols = map(x) do x
    x.values, x.rows, x.cols
  end |> tuple_of_arrays
  return PSparseMatrix(values,rows,cols)
end

function to_local_storage(x::PSparseMatrix)
  map(PSparseMatrixLocalStorage,partition(x),partition(axes(x,1)),partition(axes(x,2)))
end
