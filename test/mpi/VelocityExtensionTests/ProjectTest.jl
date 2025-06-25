module ProjectTestMPI

using Gridap, Gridap.FESpaces, Gridap.Geometry, Gridap.CellData, Gridap.Fields, Gridap.Helpers
using GridapDistributed, PartitionedArrays
using GridapTopOpt
using GridapDistributed: allocate_in_domain, allocate_in_range
using Test

function setup(model)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2)

  V_φ = TestFESpace(model,ReferenceFE(lagrangian,Float64,1))
  φh = interpolate(x->cos(x[1]),V_φ)
  V_reg = TestFESpace(model,ReferenceFE(lagrangian,Float64,2);dirichlet_tags=["boundary"])
  U_reg = TrialFESpace(V_reg,1)

  vel_ext = VelocityExtension((u,v)->∫(∇(u)⋅∇(v)+u*v)dΩ,U_reg,V_reg)
  dF = assemble_vector(∇(φ->∫(sin ∘ φ)dΩ,φh),V_φ)

  return V_φ, vel_ext, dF
end

function main_serial(model)
  V_φ, vel_ext, dF = setup(model)

  uhd = zero(V_φ)
  project!(vel_ext,dF,V_φ,uhd)

  return FEFunction(V_φ,dF), V_φ
end

function main_dist(model)
  V_φ, vel_ext, dF = setup(model)

  uhd = zero(V_φ)
  project!(vel_ext,dF,V_φ,uhd)

  ### This last map is a sanity check to ensure that dF contains the right data,
  #   doesn't need `consistent!` call as ghosts are correct due to interpolate! above.
  V_φ_gids  = get_free_dof_ids(V_φ)
  dF_gids = dF.index_partition
  dF_dofed = get_free_dof_values(uhd)
  map(GridapTopOpt._map_rhs_to_dofs!,local_views(dF_dofed),local_views(V_φ_gids),local_views(dF),dF_gids)

  return uhd, V_φ
end

function run(ranks,n)
  serial_model = CartesianDiscreteModel((0,1,0,1),(n,n))
  dFh_serial, V_φ_serial = main_serial(serial_model)
  model = GridapTopOpt.ordered_distributed_model_from_serial_model(ranks,serial_model)
  dFh_dist, V_φ_dist = main_dist(model)

  _test = GridapTopOpt.test_serial_and_distributed_fields(dFh_dist,V_φ_dist,dFh_serial,V_φ_serial)
  map_main(_test) do x
    @test x
    nothing
  end
end

with_mpi() do distribute
  ranks = distribute(LinearIndices((4,)))
  run(ranks,6)
end

end