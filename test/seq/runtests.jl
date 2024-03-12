module LSTOSequentialTests

using Test

@time @testset "Thermal Compliance - ALM" begin include("ThermalComplianceALMTests.jl") end

end # module