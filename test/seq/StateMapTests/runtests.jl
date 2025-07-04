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
  include("AffineFEStateMapTest_CutFEM_Elast.jl")
  include("Zygote_TwoStaggeredAffineFEStateMapTest.jl")

  include("TwoStaggeredAffineFEStateMapTest_ADTypeUnstableBug.jl")
  include("NonSymmetricThreeStaggeredNonlinearFEStateMapTest.jl")
end

end # module