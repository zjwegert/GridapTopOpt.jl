module JLD2SaveLoad
using Test

using Gridap, GridapTopOpt

function main()
  path = joinpath(mktempdir(), "test.jld2")
  # setup
  model = CartesianDiscreteModel((0,1,0,1),(20,20))
  V_φ = TestFESpace(model,ReferenceFE(lagrangian,Float64,1))
  φh = interpolate(initial_lsf(4,0.2),V_φ)

  save(path,get_free_dof_values(φh))
  x = load(path)
  @test x == get_free_dof_values(φh)
  load!(path,x)
  @test x == get_free_dof_values(φh)
end

main()

end # module