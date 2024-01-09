
function process_timer(t::PTimer)
  data = t.data
  tmax, tmin, tmean = map_main(data) do data
    times = map(x -> x.max,values(data))
    maximum(times), minimum(times), sum(times)/length(times)
  end |> PartitionedArrays.tuple_of_arrays
  return tmax, tmin, tmean
end

function benchmark(f, args, ranks; nreps = 10; reset! = x -> nothing)
  t = PTimer(ranks)
  f(args...)
  for i in 1:nreps
    reset!(args...)
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

  φ0 = copy(get_free_dof_values(m.φ0))
  function reset!(m)
    copy!(get_free_dof_values(m.φ0), φ0)
    reset!(get_history(m))
  end
  return benchmark(f, m, ranks; nreps, reset!)
end

function benchmark_forward_problem(m::AbstractFEStateMap, φh, ranks; nreps = 10)
  function f(m, φh)
    forward_solve(m,φh)
  end
  return benchmark(f, (m,φh), ranks; nreps)
end

function benchmark_advection(stencil::AdvectionStencil,φ0,v0,γ,ranks; nreps = 10)
  function f(stencil,φ,v,γ)
    advect!(stencil,φ,v,γ)
  end
  function reset!(stencil,φ,v,γ)
    copy!(φ,φ0)
    copy!(v,v0)
  end
  φ = copy(φ0)
  v = copy(v0)
  return benchmark(f, (stencil,φ,v,γ), ranks; nreps, reset!)
end

function benchmark_reinitialisation(stencil::AdvectionStencil,φ0,v0,γ,ranks; nreps = 10)
  function f(stencil,φ,v,γ)
    reinit!(stencil,φ,v,γ)
  end
  function reset!(stencil,φ,v,γ)
    copy!(φ,φ0)
    copy!(v,v0)
  end
  φ = copy(φ0)
  v = copy(v0)
  return benchmark(f, (stencil,φ,v,γ), ranks; nreps, reset!)
end

function benchmark_velocity_extension(ext::VelocityExtension,v0,ranks; nreps = 10)
  function f(ext,v)
    project!(ext,v)
  end
  function reset!(ext,v)
    copy!(v,v0)
  end
  v = copy(v0)
  return benchmark(f, (ext,v), ranks; nreps, reset!)
end
