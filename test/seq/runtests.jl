module GridapTopOptSequentialTests

using Test

@time @testset "StateMapTests" begin include("StateMapTests/runtests.jl") end
@time @testset "UtilityTests" begin include("UtilityTests/runtests.jl") end
@time @testset "EmbeddedTests" begin include("EmbeddedTests/runtests.jl") end
@time @testset "ReinitialiserTests" begin include("ReinitialiserTests/runtests.jl") end
@time @testset "EvolverTests" begin include("EvolverTests/runtests.jl") end
@time @testset "GridapTopOptTests" begin include("GridapTopOptTests/runtests.jl") end

end # module