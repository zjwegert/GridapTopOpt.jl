module GridapTopOptSequentialTests

using Test

@time @testset "StateMapTests" begin include("StateMapTests/runtests.jl") end
@time @testset "UtilityTests" begin include("UtilityTests/runtests.jl") end
@time @testset "EmbeddedTests" begin include("EmbeddedTests/runtests.jl") end
@time @testset "UnfittedEvolution" begin include("UnfittedEvolutionTests/runtests.jl") end
@time @testset "GridapTopOptTests" begin include("GridapTopOptTests/runtests.jl") end

end # module