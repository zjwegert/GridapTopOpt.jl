module StateMapTests

using Test

@time @testset "StateMapTests" begin
  include("AffineFEStateMapTest.jl")
  include("MultiFieldAffineFEStateMapTest.jl")
  include("MultiFieldNonlinearFEStateMapTest.jl")
  include("NonlinearFEStateMapTest.jl")
  include("RepeatingAffineFEStateMapTest.jl")
  include("TwoStaggeredAffineFEStateMapTest.jl")
  include("ThreeStaggeredAffineFEStateMapTests.jl")
  include("ThreeStaggeredNonlinearFEStateMapTest.jl")
end

end # module