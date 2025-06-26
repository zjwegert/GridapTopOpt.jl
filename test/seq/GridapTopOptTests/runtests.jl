module GridapTopOptTests

using Test

@time @testset "Thermal Compliance - ALM" begin include("ThermalComplianceALMTests.jl") end
@time @testset "Thermal Compliance - HPM" begin include("ThermalComplianceHPMTests.jl") end
@time @testset "Nonlinear Thermal Compliance - ALM" begin include("NonlinearThermalComplianceALMTests.jl") end
@time @testset "Nonlinear Neohook with Jacobian - ALM" begin include("NeohookAnalyticJacALMTests.jl") end
@time @testset "Inverse Homogenisation - ALM" begin include("InverseHomogenisationALMTests.jl") end
@time @testset "Inverter - HPM" begin include("InverterHPMTests.jl") end
@time @testset "PZMultiFieldRepeatingState - HPM" begin include("PZMultiFieldRepeatingStateTests.jl") end
@time @testset "Thermal Cut FEM" begin include("ThermalCutFEMTest.jl") end
@time @testset "FSI CutFEM" begin include("FSICutFEMTest.jl") end

end # module