module EmbeddedTests

using Test

@time @testset "EmbeddedCollectionsTests" begin include("EmbeddedCollectionsTests.jl") end
@time @testset "IsolatedVolumeTests" begin include("IsolatedVolumeTests.jl") end

end # module