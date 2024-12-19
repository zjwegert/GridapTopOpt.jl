module EmbeddedTests

using Test

@time @testset "EmbeddedDifferentiationTests" begin include("EmbeddedDifferentiationTests.jl") end
@time @testset "EmbeddedCollectionsTests" begin include("EmbeddedCollectionsTests.jl") end
@time @testset "IsolatedVolumeTests" begin include("IsolatedVolumeTests.jl") end

end # module