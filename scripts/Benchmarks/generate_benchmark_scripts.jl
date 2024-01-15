using Mustache

function generate(name,type;cputype="7702",wallhr=2,n_mesh_partition,
    mem=8*n_mesh_partition^3,n_el_size,fe_order=1,verbose=1)

  ncpus = n_mesh_partition^3

  _div,_rem=divrem(_N,cpus_per_node)
  sel_text(nnode,ncpus,mem,type) = "select=$nnode:ncpus=$ncpus:mpiprocs=$ncpus:mem=$(mem)GB:cputype=$type"
  
  if _div < 1
    select = sel_text(1,_rem,ceil.(Int,_N*gb_per_cpu),cputype)
  elseif _div > 0 && iszero(_rem)
    select = sel_text(_div,cpus_per_node,mem_per_node,cputype)
  elseif _div > 0 && _rem > 0
    select = sel_text(_div,cpus_per_node,mem_per_node,cputype)*"+"
    select *= sel_text(1,_rem,ceil.(Int,_rem*gb_per_cpu),cputype)
  else
    error()
  end

  job_data = """
  #!/bin/bash -l

  #PBS -P LSTO
  #PBS -N "{{:name}}"
  #PBS -l $select
  #PBS -l walltime={{:wallhr}}:00:00
  #PBS -j oe

  cd \$PBS_O_WORKDIR

  source \$HOME/hpc-environments-main/lyra/load-ompi.sh
  export mpiexecjl=\$HOME/.julia/bin/mpiexecjl

  julia --project=. -e "using Pkg; Pkg.precompile()"
  
  NAME="{{:name}}"
  PROB_TYPE={{:type}}
  N={{:n_mesh_partition}}
  N_EL={{:n_el_size}}
  ORDER={{:fe_order}}
  VERBOSE={{:verbose}}

  \$mpiexecjl --project=. -n {{:ncpus}} julia \\
    --check-bounds no \\
    ./scripts/Benchmarks/benchmark.jl \\
    \$NAME \\
    \$PROB_TYPE \\
    \$N \\
    \$N_EL \\
    \$ORDER \\
    \$VERBOSE
  """;

  settings = (;name,type,cputype,wallhr,ncpus,n_mesh_partition,n_el_size,mem,fe_order,verbose)
  Mustache.render(job_data, settings)
end

mem_per_node = 1003;
cpus_per_node = 128;
gb_per_cpu = mem_per_node/cpus_per_node*(1-0.001);
dofs_per_proc = 32000;

function generate_jobs(phys_type,ndof_per_node)
  N = vcat(1,2:2:16);
  strong = (N).^3;
  strong_el_x=128;
  mem = ceil.(Int,strong*gb_per_cpu);
  weak_el_x = @. floor(Int,(dofs_per_proc*strong/ndof_per_node)^(1/3)-1);

  node_cpu_usage = strong/cpus_per_node
  node_mem_usage = mem/mem_per_node
  dof_sanity_check = maximum(abs,@. ndof_per_node*(weak_el_x+1)^3/strong/dofs_per_proc - 1)

  sname(phys_type,ndof_per_node,n,elx) = "STRONG_$(phys_type)_dof$(ndof_per_node)_N$(n)_elx$(elx)"
  wname(phys_type,ndof_per_node,n,elx) = "WEAK_$(phys_type)_dof$(ndof_per_node)_N$(n)_elx$(elx)"

  strong_jobs = map(n->(sname(phys_type,ndof_per_node,n,strong_el_x),
    generate((sname(phys_type,ndof_per_node,n,strong_el_x)),phys_type;
    n_mesh_partition=n,n_el_size=strong_el_x)),N)
  weak_jobs = map((n,elx)->(wname(phys_type,ndof_per_node,n,elx),
    generate((wname(phys_type,ndof_per_node,n,elx)),phys_type;
    n_mesh_partition=n,n_el_size=elx)),N,weak_el_x)

  strong_jobs,weak_jobs,node_cpu_usage,node_mem_usage,dof_sanity_check
end

nlelas_strong_jobs,nlelas_weak_jobs,nlelas_node_cpu_usage,
  nlelas_node_mem_usage,nlelas_dof_sanity_check = generate_jobs("NLELAST",3)

elas_strong_jobs,elas_weak_jobs,elas_node_cpu_usage,
  elas_node_mem_usage,elas_dof_sanity_check = generate_jobs("ELAST",3)

ther_strong_jobs,ther_weak_jobs,ther_node_cpu_usage,
  ther_node_mem_usage,ther_dof_sanity_check = generate_jobs("THERM",1)

for job in vcat(nlelas_strong_jobs,nlelas_weak_jobs,
    elas_strong_jobs,elas_weak_jobs,
    ther_strong_jobs,ther_weak_jobs)
  name = job[1]
  content = job[2]
  open("./scripts/Benchmarks/jobs/$name.pbs","w") do f
    write(f,content)
  end
end