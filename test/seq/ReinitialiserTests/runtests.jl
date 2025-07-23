module ReinitialiserTests

using Test

@testset "Reinitialisation (artificial viscosity)" begin include("ArtificialViscosityStabilisedReinitTest.jl") end
@testset "Reinitialisation (interior penalty)" begin include("InteriorPenaltyStabilisedReinitTest.jl") end
@testset "Reinitialisation (multi-stage)" begin include("MultiStageStabilisedReinitTest.jl") end
@testset "Reinitialisation (heat reinit)" begin include("HeatReinitialiserTest.jl") end
@testset "Reinitialisation (heat reinit gmsh)" begin include("HeatReinitialiserGmshTest.jl") end
@testset "Reinitialisation (heat reinit AD)" begin include("HeatReinitialiserDifferentiability.jl") end

end # module