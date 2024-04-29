module LSTOSequentialTests

using Test

@time @testset "Thermal Compliance - ALM" begin include("ThermalComplianceALMTests.jl") end
@time @testset "Nonlinear Thermal Compliance - ALM" begin include("NonLinearThermalComplianceALMTests.jl") end
@time @testset "Inverse Homogenisation - ALM" begin include("InverseHomogenisationALMTests.jl") end
@time @testset "Inverter - HPM" begin include("InverterHPMTests.jl") end

end # module