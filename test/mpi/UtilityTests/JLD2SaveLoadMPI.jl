module JLD2SaveLoadMPI
using Test

using Gridap, Gridap.MultiField, GridapDistributed, GridapPETSc, GridapSolvers,
  PartitionedArrays, GridapTopOpt, SparseMatricesCSR

function main(distribute,mesh_partition)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  path = joinpath("tmp","test")
  i_am_main(ranks) && mkpath(path)
  # setup
  model = CartesianDiscreteModel(ranks,mesh_partition,(0,1,0,1),(20,20))
  V_φ = TestFESpace(model,ReferenceFE(lagrangian,Float64,1))
  φh = interpolate(initial_lsf(4,0.2),V_φ)

  psave(path,get_free_dof_values(φh))
  x = pload(path,ranks)
  @assert x == get_free_dof_values(φh)
  pload!(path,x)
  @assert x == get_free_dof_values(φh)
  true
end

with_mpi() do distribute
  @test main(distribute,(2,2))
end

end # module