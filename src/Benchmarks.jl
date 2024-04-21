"""
    benchmark(f, args, ranks; nreps, reset!)

Benchmark a function `f` that takes arguments `args`. 
  
In MPI mode, benchmark will always return the maximum CPU time across all ranks.
This behaviour can be changed by overwritting `process_timer`.

# Important
The input `ranks` allows the user to provide the MPI ranks, `benchmark`
will not function correctly in MPI mode if these are not supplied. In serial,
set `ranks = nothing`.

# Optional

- `nreps = 10`: Number of benchmark repetitions
- `reset!= (x...) -> nothing`: Function for resetting inital data (e.g., level-set function ``\\varphi``). 
"""
function benchmark(f, args, ranks::Nothing; nreps = 10, reset! = (x...) -> nothing)
  t = zeros(Float64,nreps)
  println("<------------- Compilation  ------------->")
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
  i_am_main(ranks) && println("<------------- Compilation  ------------->")
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

function process_timer(t::Vector)
  # return length(t), maximum(t), minimum(t), sum(t)/length(t)
  return t
end

function process_timer(t::PTimer)
  data = t.data
  map_main(data) do data
    times = map(x -> x.max,values(data))
    process_timer(times)
  end |> PartitionedArrays.getany
end

## Standard benchmarks
"""
    benchmark_optimizer(m::Optimiser, niter, ranks; nreps)

Given an optimiser `m`, benchmark `niter` iterations.
"""
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

"""
    benchmark_single_iteration(m::Optimiser, ranks; nreps)

Given an optimiser `m`, benchmark a single iteration after 0th iteration.
"""
function benchmark_single_iteration(m::Optimiser, ranks; nreps = 10)
  _, state = iterate(m)
  _, L, J, C, dL, dJ, dC, uh, φh, vel, λ, Λ, γ, os_it = state;
  function f(m)
    _state = (;it=1,L,J,C,dL,dJ,dC,uh,φh,vel,λ,Λ,γ,os_it)
    iterate(m, _state)
  end

  φ0 = copy(get_free_dof_values(m.φ0))
  function opt_reset!(m::Optimiser)
    copy!(get_free_dof_values(m.φ0), φ0)
    reset!(get_history(m))
  end
  return benchmark(f, (m,), ranks; nreps, reset! = opt_reset!)
end

"""
    benchmark_forward_problem(m::AbstractFEStateMap, φh, ranks; nreps)

Benchmark the forward FE solve given `m::AbstractFEStateMap` and a level-set 
function `φh`. See [`forward_solve!`](@ref) for input types.
"""
function benchmark_forward_problem(m::AbstractFEStateMap, φh, ranks; nreps = 10)
  function f(m, φh)
    forward_solve!(m,φh)
  end
  function reset!(m,φh)
    u = get_free_dof_values(get_state(m));
    fill!(u,zero(eltype(u)))
  end
  return benchmark(f, (m,φh), ranks; nreps)
end

"""
    benchmark_advection(stencil::LevelSetEvolution, φ0, v0, γ, ranks; nreps)

Benchmark solving the Hamilton-Jacobi evolution equation given a `stencil`,
level-set function `φ0`, velocity function `v0`, and time step coefficient `γ`. 
See [`evolve!`](@ref) for input types.
"""
function benchmark_advection(stencil::LevelSetEvolution, φ0, v0, γ, ranks; nreps = 10)
  function f(stencil,φ,v,γ)
    evolve!(stencil,φ,v,γ)
  end
  function reset!(stencil,φ,v,γ)
    copy!(φ,φ0)
    copy!(v,v0)
  end
  φ = copy(φ0)
  v = copy(v0)
  return benchmark(f, (stencil,φ,v,γ), ranks; nreps, reset!)
end

"""
    benchmark_reinitialisation(stencil::LevelSetEvolution, φ0, γ_reinit, ranks; nreps)

Benchmark solving the reinitialisation equation given a `stencil`, level-set function
`φ0`, and time step coefficient `γ`. See [`reinit!`](@ref) for input types.
"""
function benchmark_reinitialisation(stencil::LevelSetEvolution, φ0, γ_reinit, ranks; nreps = 10)
  function f(stencil,φ,γ_reinit)
    reinit!(stencil,φ,γ_reinit)
  end
  function reset!(stencil,φ,γ_reinit)
    copy!(φ,φ0)
  end
  φ = copy(φ0)
  return benchmark(f, (stencil,φ,γ_reinit), ranks; nreps, reset!)
end

"""
    benchmark_velocity_extension(ext::VelocityExtension, v0, ranks; nreps)

Benchmark the Hilbertian velocity-extension method `ext` given a RHS `v0`.
See [`project!`](@ref) for input types.
"""
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

"""
    benchmark_hilbertian_projection_map(m::HilbertianProjectionMap, dV, C, dC, K, ranks; nreps)

Benchmark `update_descent_direction!` for `HilbertianProjectionMap` given a objective 
sensitivity `dV`, constraint values C, constraint sensitivities `dC`, and stiffness
matrix `K` for the velocity-extension.
"""
function benchmark_hilbertian_projection_map(m::HilbertianProjectionMap, dV, C, dC, K, ranks; nreps = 10)
  function f(m,dV,C,dC,K)
    update_descent_direction!(m,dV,C,dC,K)
  end
  return benchmark(f, (m,dV,C,dC,K), ranks; nreps)
end
