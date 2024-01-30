function process_timer(t::PTimer)
  data = t.data
  map_main(data) do data
    times = map(x -> x.max,values(data))
    process_timer(times)
  end |> PartitionedArrays.getany
end

function process_timer(t::Vector)
  return length(t), maximum(t), minimum(t), sum(t)/length(t)
end

function benchmark(f, args, ranks::Nothing; nreps = 10, reset! = (x...) -> nothing)
  t = zeros(Float64,nreps)
  println("<------------- Compilation ------------->")
  f(args...)
  println("<------------- Benchmarking ------------->")
  for i in 1:nreps
    println("      Benchmark - Iteration $i of $nreps")
    reset!(args...)
    t[i] = @elapsed f(args...)
  end
  return process_timer(t)
end

function benchmark(f, args, ranks; nreps = 10, reset! = (x...) -> nothing)
  t = PTimer(ranks)
  i_am_main(ranks) && println("<------------- Compilation ------------->")
  f(args...)
  i_am_main(ranks) && println("<------------- Benchmarking ------------->")
  for i in 1:nreps
    i_am_main(ranks) && println("      Benchmark - Iteration $i of $nreps")
    reset!(args...)
    tic!(t;barrier=true)
    f(args...)
    toc!(t,"t_$(i)")
  end
  return process_timer(t)
end

function benchmark_optimizer(m::Optimiser, niter, ranks; nreps = 10)
  function f(m)
    _, state = iterate(m)
    for _ in 1:niter
      _, state = iterate(m, state)
    end
  end

  φ0 = copy(get_free_dof_values(m.φ0))
  function opt_reset!(m::Optimiser)
    copy!(get_free_dof_values(m.φ0), φ0)
    reset!(get_history(m))
  end
  return benchmark(f, (m,), ranks; nreps, reset! = opt_reset!)
end

function benchmark_forward_problem(m::AbstractFEStateMap, φh, ranks; nreps = 10)
  function f(m, φh)
    forward_solve(m,φh)
  end
  function reset!(m,φh)
    u = get_free_dof_values(get_state(m));
    fill!(u,zero(eltype(u)))
  end
  return benchmark(f, (m,φh), ranks; nreps)
end

function benchmark_advection(stencil::AdvectionStencil, φ0, v0, γ, ranks; nreps = 10)
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

function benchmark_reinitialisation(stencil::AdvectionStencil, φ0, γ_reinit, ranks; nreps = 10)
  function f(stencil,φ,γ_reinit)
    reinit!(stencil,φ,γ_reinit)
  end
  function reset!(stencil,φ,γ_reinit)
    copy!(φ,φ0)
  end
  φ = copy(φ0)
  return benchmark(f, (stencil,φ,γ_reinit), ranks; nreps, reset!)
end

function benchmark_velocity_extension(ext::VelocityExtension, v0, ranks; nreps = 10)
  function f(ext,v)
    project!(ext,v)
  end
  function reset!(ext,v)
    copy!(v,v0)
  end
  v = copy(v0)
  return benchmark(f, (ext,v), ranks; nreps, reset!)
end

function benchmark_hilbertian_projection_map(m::HilbertianProjectionMap, dV, C, dC, K, ranks; nreps = 10)
  function f(m,dV,C,dC,K)
    update_descent_direction!(m,dV,C,dC,K)
  end
  return benchmark(f, (m,dV,C,dC,K), ranks; nreps)
end
