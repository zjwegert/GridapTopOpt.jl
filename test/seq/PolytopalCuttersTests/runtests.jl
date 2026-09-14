module PolytopalCuttersTests

using Test

@time @testset "PolytopalCutterTests" begin include("PolytopalCutterTests.jl") end

@time @testset "PoissonCutFEM" begin include("PoissonCutFEMTests.jl") end

@time @testset "PoissonAgFEM" begin include("PoissonAgFEMTests.jl") end

@time @testset "PeriodicPoissonAgFEM" begin include("PeriodicPoissonAgFEMTests.jl") end

@time @testset "BimaterialPoissonCutFEM" begin include("BimaterialPoissonCutFEMTests.jl") end

@time @testset "EmbeddedBimaterialPoissonCutFEM" begin include("EmbeddedBimaterialPoissonCutFEMTests.jl") end

@time @testset "StokesCutFEM" begin include("StokesCutFEMTests.jl") end

@time @testset "StokesAgFEM" begin include("StokesAgFEMTests.jl") end

@time @testset "TraceFEM" begin include("TraceFEMTests.jl") end

@time @testset "DifferentiableCutPolyTriangulationsTests" begin include("DifferentiableCutPolyTriangulationsTests.jl") end

# # To remove
# @time @testset "DifferentiableCutPolyTriangulationsTests_Analytic1" begin include("DifferentiableCutPolyTriangulationsTests_Analytic1.jl") end
# @time @testset "DifferentiableCutPolyTriangulationsTests_Analytic2" begin include("DifferentiableCutPolyTriangulationsTests_Analytic2.jl") end
# @time @testset "DifferentiableCutPolyTriangulationsTests_Analytic3" begin include("DifferentiableCutPolyTriangulationsTests_Analytic3.jl") end

end
