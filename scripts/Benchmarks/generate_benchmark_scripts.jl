using Mustache

function generate(
    template,
    name,
    type,
    cputype,
    wallhr,
    n_mesh_partition,
    n_el_size,
    fe_order,
    verbose,
    dir_name,
    write_dir
  )

  ncpus = n_mesh_partition^3

  _div,_rem=divrem(ncpus,cpus_per_node)
  function sel_text(nnode,ncpus,mem,type)
    if occursin("STRONG",name)
      _mem = nnode <= 2 && ncpus <= 32 ? 256 : mem;
    else
      _mem = mem
    end
    return "$nnode:ncpus=$ncpus:mpiprocs=$ncpus:mem=$(_mem)GB:cputype=$type"
  end

  if occursin("STRONG",name)
    n_mesh_partition <= 2 ? wallhr = 200 : nothing
  end
  
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

  settings = (;name,select,type,cputype,wallhr,ncpus,
    n_mesh_partition,n_el_size,fe_order,verbose,dir_name,write_dir)
  Mustache.render(template, settings)
end

function generate_jobs(template,phys_type,ndof_per_node)
  strong = (N).^3;
  weak_el_x = @. floor(Int,(dofs_per_proc*strong/ndof_per_node)^(1/3)-1);
  dof_sanity_check = @.(floor(Int,ndof_per_node*(weak_el_x+1)^3/strong)),
    maximum(abs,@. ndof_per_node*(weak_el_x+1)^3/strong/dofs_per_proc - 1)

  sname(phys_type,ndof_per_node,n,elx) = "STRONG_$(phys_type)_dof$(ndof_per_node)_N$(n)_elx$(elx)_DoFsPerProc$dofs_per_proc"
  wname(phys_type,ndof_per_node,n,elx) = "WEAK_$(phys_type)_dof$(ndof_per_node)_N$(n)_elx$(elx)_DoFsPerProc$dofs_per_proc"

  strong_jobs = map(n->(sname(phys_type,ndof_per_node,n,strong_el_x),
    generate(template,sname(phys_type,ndof_per_node,n,strong_el_x),phys_type,
    cputype,wallhr,n,strong_el_x,fe_order,verbose,dir_name,write_dir)),N)
  weak_jobs = map((n,elx)->(wname(phys_type,ndof_per_node,n,elx),
    generate(template,wname(phys_type,ndof_per_node,n,elx),phys_type,
    cputype,wallhr,n,elx,fe_order,verbose,dir_name,write_dir)),N,weak_el_x)

  strong_jobs,weak_jobs,dof_sanity_check
end

job_output_path = "./scripts/Benchmarks/jobs/DoFsPerProc_40000/";
mkpath(job_output_path);

# SETUP PARAMETERS
cputype="7702";
mem_per_node = 1003; # GB
cpus_per_node = 128;
gb_per_cpu = mem_per_node/cpus_per_node*(1-0.001); # Only used for partial node use
wallhr=24; # Hours

dofs_per_proc = 40000;
fe_order=1;
verbose=1;
dir_name=splitpath(Base.active_project())[end-1];
write_dir = "\$HOME/$dir_name/results/benchmarks/"

# N = vcat(1,2:2:16); # Number of partitions in x-axis
# strong_el_x=128; # Number of elements in x-axis (strong scaling)
N = 1:5; # Number of partitions in x-axis
strong_el_x=100; # Number of elements in x-axis (strong scaling)

# Phys type and number of dofs per node, this corresponds to driver
phys_types = [
  # ("NLELAST",3),
  # ("ELAST",3),
  # ("INVERTER_HPM",3),
  ("THERM",1)
];

# Template
template = read("./scripts/Benchmarks/jobtemplate.sh",String)

## Generate Jobs
jobs_by_phys = map(x->(x[1],generate_jobs(template,x[1],x[2])),phys_types);

for jobs in jobs_by_phys
  strong_jobs,weak_jobs,dof_sanity_check = jobs[2]
  
  println("$(jobs[1]): Weak dofs = $(dof_sanity_check[1])\n     Error = $(dof_sanity_check[2])\n")
  for job in vcat(strong_jobs,weak_jobs)
    name = job[1]
    content = job[2]
    open(job_output_path*"$name.pbs","w") do f
      write(f,content)
    end
  end
end