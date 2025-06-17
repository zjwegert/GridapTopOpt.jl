module UtilityTests

using Test

@time @testset "JLD2SaveLoad" begin include("JLD2SaveLoad.jl") end
@time @testset "ElementDiameterTests" begin include("ElementDiameterTests.jl") end

end # module