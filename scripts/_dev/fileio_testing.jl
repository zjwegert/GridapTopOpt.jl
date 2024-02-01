using Gridap, GridapDistributed, GridapPETSc, PartitionedArrays, LSTO_Distributed

with_mpi() do distribute
  mesh_partition = (2,2);
  el_size = (10,10)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  model = CartesianDiscreteModel(ranks,mesh_partition,(0,1,0,1),el_size);
  Ω = Triangulation(model);
  reffe_scalar = ReferenceFE(lagrangian,Float64,1);
  V_φ = TestFESpace(model,reffe_scalar);

  φh = interpolate(x->x[1]^2*x[2]^2,V_φ);
  φ = get_free_dof_values(φh)
  φ0 = copy(get_free_dof_values(φh))

  write_vector("./results/afolder/LSF_Data_1",φ,',')

  fill!(φ,0.0)

  write_file_to_vector!("./results/afolder/LSF_Data_1",φ,',')

  @assert φ == φ0
end