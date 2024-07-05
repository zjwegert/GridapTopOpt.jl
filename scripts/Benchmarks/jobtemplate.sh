#!/bin/bash --login
#SBATCH --account=pawsey1076
#SBATCH --partition=work
#SBATCH --ntasks={{:ncpus}}
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH --time={{:wallhr}}:00:00

# Load Julia/MPI/PETSc enviroment variables
source $MYSCRATCH/GridapTopOpt.jl/scripts/Benchmarks/hpc-enviroments-setonix/load-configs.sh
source $MYSCRATCH/GridapTopOpt.jl/scripts/Benchmarks/hpc-enviroments-setonix/load-cray-mpich.sh
export PROJECT_DIR=$MYSCRATCH/GridapTopOpt.jl/

# Set MPI related environment variables. (Not all need to be set)
# Main variables for multi-node jobs (activate for multinode jobs)
export MPICH_OFI_STARTUP_CONNECT=1
export MPICH_OFI_VERBOSE=1
#Ask MPI to provide useful runtime information (activate if debugging)
#export MPICH_ENV_DISPLAY=1
#export MPICH_MEMORY_REPORT=1

srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS julia --project=$PROJECT_DIR --check-bounds no -O3 \
    $PROJECT_DIR/scripts/Benchmarks/benchmark.jl \
    {{:name}} \
    {{{:write_dir}}} \
    {{:type}} \
    {{:bmark_type}} \
    {{:Nx_partition}} \
    {{:Ny_partition}} \
    {{:Nz_partition}} \
    {{:n_el_size}} \
    {{:fe_order}} \
    {{:verbose}} \
    {{:nreps}}