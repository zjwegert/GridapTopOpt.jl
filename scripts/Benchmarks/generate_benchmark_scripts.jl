using Mustache

function generate(
    name,
    type,
    cputype,
    wallhr,
    n_mesh_partition,
    n_el_size,
    fe_order,
    verbose,
    dir_name
  )

  ncpus = n_mesh_partition^3

  _div,_rem=divrem(ncpus,cpus_per_node)
  sel_text(nnode,ncpus,mem,type) = "$nnode:ncpus=$ncpus:mpiprocs=$ncpus:mem=$(mem)GB:cputype=$type"
  
  if _div < 1
    select = sel_text(1,_rem,ceil.(Int,ncpus*gb_per_cpu),cputype)
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
  #PBS -l select=$select
  #PBS -l walltime={{:wallhr}}:00:00
  #PBS -j oe

  source \$HOME/hpc-environments-main/lyra/load-ompi.sh
  PROJECT_DIR=\$HOME/{{:dir_name}}/

  julia --project=\$PROJECT_DIR -e "using Pkg; Pkg.precompile()"

  mpiexec --hostfile \$PBS_NODEFILE 
    julia --project=\$PROJECT_DIR --check-bounds no -O3 \$PROJECT_DIR/scripts/Benchmarks/benchmark.jl \\
    {{:name}} \\
    {{:type}} \\
    {{:n_mesh_partition}} \\
    {{:n_el_size}} \\
    {{:fe_order}} \\
    {{:verbose}}
  """;

  settings = (;name,type,cputype,wallhr,ncpus,
    n_mesh_partition,n_el_size,fe_order,verbose,dir_name)
  Mustache.render(job_data, settings)
end

function generate_jobs(phys_type,ndof_per_node)
  strong = (N).^3;
  weak_el_x = @. floor(Int,(dofs_per_proc*strong/ndof_per_node)^(1/3)-1);
  dof_sanity_check = @.(floor(Int,ndof_per_node*(weak_el_x+1)^3/strong)),
    maximum(abs,@. ndof_per_node*(weak_el_x+1)^3/strong/dofs_per_proc - 1)

  sname(phys_type,ndof_per_node,n,elx) = "STRONG_$(phys_type)_dof$(ndof_per_node)_N$(n)_elx$(elx)"
  wname(phys_type,ndof_per_node,n,elx) = "WEAK_$(phys_type)_dof$(ndof_per_node)_N$(n)_elx$(elx)"

  strong_jobs = map(n->(sname(phys_type,ndof_per_node,n,strong_el_x),
    generate(sname(phys_type,ndof_per_node,n,strong_el_x),phys_type,
    cputype,wallhr,n,strong_el_x,fe_order,verbose,dir_name)),N)
  weak_jobs = map((n,elx)->(wname(phys_type,ndof_per_node,n,elx),
    generate(wname(phys_type,ndof_per_node,n,elx),phys_type,
    cputype,wallhr,n,elx,fe_order,verbose,dir_name)),N,weak_el_x)

  strong_jobs,weak_jobs,dof_sanity_check
end

# SETUP PARAMETERS
cputype="7702";
mem_per_node = 1003; # GB
cpus_per_node = 128;
gb_per_cpu = mem_per_node/cpus_per_node*(1-0.001); # Only used for partial node use
wallhr=2; # Hours

dofs_per_proc = 32000;
fe_order=1;
verbose=1;
dir_name=splitpath(Base.active_project())[end-1];

mem_per_node = 1003;
cpus_per_node = 128;
gb_per_cpu = mem_per_node/cpus_per_node*(1-0.001);

N = vcat(1,2:2:16); # Number of partitions in x-axis
strong_el_x=128; # Number of elements in x-axis (strong scaling)

# Phys type and number of dofs per node, this corresponds to driver
phys_types = [
  ("NLELAST",3),
  ("ELAST",3),
  ("THERM",1)
];

## Generate Jobs
jobs_by_phys = map(x->(x[1],generate_jobs(x[1],x[2])),phys_types);

for jobs in jobs_by_phys
  strong_jobs,weak_jobs,dof_sanity_check = jobs[2]
  
  println("$(jobs[1]): Weak dofs = $(dof_sanity_check[1])\n     Error = $(dof_sanity_check[2])\n")
  for job in vcat(strong_jobs,weak_jobs)
    name = job[1]
    content = job[2]
    open("./scripts/Benchmarks/jobs/$name.pbs","w") do f
      write(f,content)
    end
  end
end