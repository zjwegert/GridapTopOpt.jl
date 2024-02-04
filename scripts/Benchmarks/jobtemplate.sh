#!/bin/bash -l

#PBS -P LSTO
#PBS -N "{{:name}}"
#PBS -l select={{:ncpus}}:ncpus=1:mpiprocs=1:ompthreads=1:mem={{:mem}}GB:cputype={{:cputype}}
#PBS -l walltime={{:wallhr}}:00:00
#PBS -j oe

source $HOME/hpc-environments-main/lyra/load-ompi.sh
PROJECT_DIR=$HOME/{{:dir_name}}/

julia --project=$PROJECT_DIR -e "using Pkg; Pkg.precompile()"

mpirun --hostfile $PBS_NODEFILE julia --project=$PROJECT_DIR --check-bounds no -O3 --compiled-modules=no \
    $PROJECT_DIR/scripts/Benchmarks/benchmark.jl \
    {{:name}} \
    {{{:write_dir}}} \
    {{:type}} \
    {{:bmark_type}} \
    {{:n_mesh_partition}} \
    {{:n_el_size}} \
    {{:fe_order}} \
    {{:verbose}} \
    {{:nreps}}

read _ _ PBS_WALLTIME  <<< `qstat -f $PBS_JOBID | grep "resources_used.walltime"`
PBS_WALLTIME_SECS=$(echo $PBS_WALLTIME | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }')
echo $PBS_WALLTIME_SECS >> {{{:write_dir}}}/{{:name}}_walltime.txt