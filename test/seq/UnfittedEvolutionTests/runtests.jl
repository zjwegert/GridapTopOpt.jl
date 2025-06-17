module UnfittedEvolutionTests

using Test

@testset "Reinitialisation (artificial viscosity)" begin include("ArtificialViscosityStabilisedReinitTest.jl") end
@testset "Reinitialisation (interior penalty)" begin include("InteriorPenaltyStabilisedReinitTest.jl") end
@testset "Reinitialisation (multi-stage)" begin include("MultiStageStabilisedReinitTest.jl") end
@testset "CutFEM evolution" begin include("CutFEMEvolveTest.jl") end

end # module