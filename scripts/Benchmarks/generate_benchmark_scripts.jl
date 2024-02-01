using Pkg
Pkg.activate()
using Mustache

function generate(
    template,
    name,
    type,
    bmark_type,
    cputype,
    wallhr,
    n_mesh_partition,
    n_el_size,
    fe_order,
    verbose,
    dir_name,
    write_dir,
    nreps
  )

  ncpus = n_mesh_partition^3
  wallhr = occursin("STRONG",name) && n_mesh_partition <= 3 ? 50 : wallhr
  mem = occursin("STRONG",name) && n_mesh_partition == 1 ? 256 : 
        occursin("STRONG",name) && n_mesh_partition == 2 ? 32 : gb_per_cpu;

  settings = (;name,type,bmark_type,cputype,wallhr,ncpus,mem,
    n_mesh_partition,n_el_size,fe_order,verbose,dir_name,write_dir,nreps)
  Mustache.render(template, settings)
end

function generate_jobs(template,phys_type,ndof_per_node,bmark_types)
  strong = (N).^3;
  weak_el_x = @. floor(Int,(dofs_per_proc*strong/ndof_per_node)^(1/3)-1);
  dof_sanity_check = @.(floor(Int,ndof_per_node*(weak_el_x+1)^3/strong)),
    maximum(abs,@. ndof_per_node*(weak_el_x+1)^3/strong/dofs_per_proc - 1)

  sname(phys_type,ndof_per_node,n,elx) = "STRONG_$(phys_type)_dof$(ndof_per_node)_N$(n)_elx$(elx)"
  wname(phys_type,ndof_per_node,n,elx) = "WEAK_$(phys_type)_dof$(ndof_per_node)_N$(n)_elx$(elx)"

  strong_jobs = map(n->(sname(phys_type,ndof_per_node,n,strong_el_x),
    generate(template,sname(phys_type,ndof_per_node,n,strong_el_x),phys_type,bmark_types,
    cputype,wallhr,n,strong_el_x,fe_order,verbose,dir_name,write_dir,nreps)),N)
  weak_jobs = map((n,elx)->(wname(phys_type,ndof_per_node,n,elx),
    generate(template,wname(phys_type,ndof_per_node,n,elx),phys_type,bmark_types,
    cputype,wallhr,n,elx,fe_order,verbose,dir_name,write_dir,nreps)),N,weak_el_x)

  strong_jobs,weak_jobs,dof_sanity_check
end

job_output_path = "./scripts/Benchmarks/jobs/";
mkpath(job_output_path);

# SETUP PARAMETERS
cputype="7702";
gb_per_cpu = 8 # GB
wallhr = 3 ; # Hours (Note may want to manually change some afterwards)

nreps = 10; # Number of benchmark repetitions
dofs_per_proc = 32000;
fe_order= 1;
verbose= 1;
dir_name= "LSTO_Distributed";
write_dir = "\$HOME/$dir_name/results/benchmarks/"

N = 1:10; # Number of partitions in x-axis
strong_el_x=100; # Number of elements in x-axis (strong scaling)

# Phys type and number of dofs per node, and what to benchmark
phys_types = [
  ("THERM",1,"bopt0,bopt1,bfwd,badv,breinit,bvelext"),
  ("ELAST",3,"bopt0,bopt1,bfwd"),
  ("NLELAST",3,"bopt0,bopt1,bfwd"),
  ("INVERTER_HPM",3,"bhpm"),
];

# Template
template = read("./scripts/Benchmarks/jobtemplate.sh",String)

## Generate Jobs
jobs_by_phys = map(x->(x[1],generate_jobs(template,x[1],x[2],x[3])),phys_types);

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