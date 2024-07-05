#!/bin/bash --login
#SBATCH --account=pawsey1076
#SBATCH --partition=work
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

source $SLURM_SUBMIT_DIR/../load-configs.sh
source $SLURM_SUBMIT_DIR/../load-cray-mpich.sh

srun -N 1 -n 1 -c 1 julia --project=$SLURM_SUBMIT_DIR $SLURM_SUBMIT_DIR/compile/compile.jl
