#!/bin/bash --login
#SBATCH --account=a_challis
#SBATCH --job-name=VIS_Sym_FSI_P1P0dc
#SBATCH --partition=general
#SBATCH --ntasks=96
#SBATCH --ntasks-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=10:00:00
#SBATCH --constraint=epyc3
#SBATCH --batch=epyc3

source $HOME/UnfittedJobs/modules.sh
cd $HOME/scratch/GridapTopOpt/

ncpus=$SLURM_NTASKS
result_path=/scratch/user/zacharywegert/GridapTopOpt/results/CutFEM_Wheel_MinCompliance_Neumann_gammag_0.01_vf_0.3_superlu_cylinder
mesh_name=wheel_cylinder.msh
I0=0
IF=485
Imod=1

julia --project -O3 -e\
  '
  @time "Preloading libraries (serial)" begin
    using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
    using GridapEmbedded, GridapEmbedded.LevelSetCutters
    using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
    using GridapGmsh
    using GridapTopOpt
    using GridapDistributed,PartitionedArrays,GridapPETSc
  end
  '

srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS --export=ALL julia --project -O3 \
  scripts/Embedded/Examples/FluidStructure/Burman-Stokes/Symmetric_FSI_3D_P1P0dc_MPI_superlu.jl \
  $ncpus \
  $result_path \
  $mesh_name \
  $I0 \
  $IF \
  $Imod
