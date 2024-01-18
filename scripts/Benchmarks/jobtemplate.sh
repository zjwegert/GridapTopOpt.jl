#!/bin/bash -l

#PBS -P LSTO
#PBS -N "{{:name}}"
#PBS -l select={{:select}}
#PBS -l walltime={{:wallhr}}:00:00
#PBS -j oe

source $HOME/hpc-environments-main/lyra/load-ompi.sh
PROJECT_DIR=$HOME/{{:dir_name}}/

julia --project=$PROJECT_DIR -e "using Pkg; Pkg.precompile()"

mpiexec --hostfile $PBS_NODEFILE \
  julia --project=$PROJECT_DIR --check-bounds no -O3 \
  $PROJECT_DIR/scripts/Benchmarks/benchmark.jl \
  {{:name}} \
  {{{:write_dir}}} \
  {{:type}} \
  {{:n_mesh_partition}} \
  {{:n_el_size}} \
  {{:fe_order}} \
  {{:verbose}}