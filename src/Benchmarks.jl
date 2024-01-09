
function process_timer(t::PTimer)
  data = t.data
  tmax, tmin, tmean = map_main(data) do data
    times = map(x -> x.max,values(data))
    maximum(times), minimum(times), sum(times)/length(times)
  end |> PartitionedArrays.tuple_of_arrays
  return tmax, tmin, tmean
end

function benchmark(f, args, ranks; nreps = 10)
  t = PTimer(ranks)
  f(args...)
  for i in 1:nreps
    tic!(t;barrier=true)
    f(args...)
    toc!(t,"t_$(i)")
  end
  return process_timer(t)
end

function benchmark_optimizer(m::Optimizer, niter, ranks; nreps = 10)
  function f(m)
    _, state = iterate(m)
    for _ in 1:niter
      _, state = iterate(m, state)
    end
  end
  return benchmark(f, m, ranks; nreps)
end

function benchmark_forward_problem(m::AbstractFEStateMap, φh, ranks; nreps = 10)
  function f(m, φh)
    forward_solve(m,φh)
  end
  return benchmark(f, (m,φh), ranks; nreps)
end

