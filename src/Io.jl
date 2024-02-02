
function psave(dir::AbstractString, x)
  ranks = get_parts(x)
  i_am_main(ranks) && mkpath(dir)
  arr = to_local_storage(x)
  map(ranks,arr) do id, arr
    filename = joinpath(dir,basename(dir)*"_$id.jdl2")
    JLD2.save_object(filename,arr)
  end
end

function pload(dir::AbstractString, ranks::AbstractArray{<:Integer})
  arr = map(ranks) do id
    filename = joinpath(dir,basename(dir)*"_$id.jdl2")
    JLD2.load_object(filename)
  end
  return from_local_storage(arr)
end

function pload!(dir::AbstractString, x)
  ranks = get_parts(x)
  y = pload(dir,ranks)
  copyto!(x,y)
  return x
end

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
