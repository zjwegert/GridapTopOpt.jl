module VelocityExtensionTestsMPI

using Test
using MPI
using GridapTopOpt

using Gridap, GridapDistributed, GridapPETSc, GridapSolvers,
  PartitionedArrays, SparseMatricesCSR

testdir = @__DIR__
istest(f) = endswith(f, ".jl") && !(f=="runtests.jl")
testfiles = sort(filter(istest, readdir(testdir)))

MPI.mpiexec() do cmd
  for file in testfiles
    path = joinpath(testdir,file)
    _cmd = `$(cmd) -np 4 --allow-run-as-root --oversubscribe $(Base.julia_cmd()) --project=. $path`
    @show _cmd
    run(_cmd)
    @test true
  end
end

end