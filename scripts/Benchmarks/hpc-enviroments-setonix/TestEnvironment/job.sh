#!/bin/bash --login
#SBATCH --account=pawsey1076
#SBATCH --partition=work
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00

# Load Julia/MPI/PETSc enviroment variables
source $SLURM_SUBMIT_DIR/../load-configs.sh
source $SLURM_SUBMIT_DIR/../load-cray-mpich.sh

# Set MPI related environment variables. (Not all need to be set)
# Main variables for multi-node jobs (activate for multinode jobs)
# export MPICH_OFI_STARTUP_CONNECT=1
# export MPICH_OFI_VERBOSE=1
#Ask MPI to provide useful runtime information (activate if debugging)
#export MPICH_ENV_DISPLAY=1
#export MPICH_MEMORY_REPORT=1

srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS julia --project=$SLURM_SUBMIT_DIR -J$SLURM_SUBMIT_DIR.so -e'
  using TestEnvironment;
  main_poisson(;nprocs=(2,2),ncells=(100,100))
'