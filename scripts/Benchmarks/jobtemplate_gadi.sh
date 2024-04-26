#!/bin/bash

#PBS -P np01
#PBS -q normal 
#PBS -N "{{:name}}"
#PBS -l ncpus={{:ncpus}}
#PBS -l mem={{:mem}}GB
#PBS -l walltime={{:wallhr}}:00:00
#PBS -j oe

source $HOME/hpc-environments/gadi/load-intel.sh
PROJECT_DIR=$SCRATCH/{{:dir_name}}/

julia --project=$PROJECT_DIR -e "using Pkg; Pkg.precompile()"

mpirun -n {{:ncpus}} julia --project=$PROJECT_DIR --check-bounds no -O3 \
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