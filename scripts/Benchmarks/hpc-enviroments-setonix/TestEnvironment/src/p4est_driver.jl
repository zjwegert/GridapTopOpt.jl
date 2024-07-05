
function main_p4est(;nprocs)
  with_mpi() do distribute
    main_p4est(distribute,nprocs)
  end
end

function main_p4est(distribute,nprocs)
  ranks = distribute(LinearIndices((prod(nprocs),)))

  GridapP4est.with(ranks) do
    domain = (0,1,0,1)
    coarse_model = CartesianDiscreteModel(domain,(2,2))
    model = OctreeDistributedDiscreteModel(ranks,coarse_model,2)
    rmodel, _ = refine(model)
    dmodel, _ = redistribute(model)
  end
end
