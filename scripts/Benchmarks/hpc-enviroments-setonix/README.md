# Setonix environment

It is recommended that the [Setonix documentation](https://pawsey.atlassian.net/wiki/spaces/US/pages/51925226/Setonix+Guides) is first read before usage. The setup is quite different to Gadi and uses Slurm.

## Setup

Three environment variables are required for everything to work:

  - `PROJECT`: Project name. For instance, `bt62` for Santi's project.
  - `P4EST_VERSION`: Version used for P4est. For instance, `2.8.5`.
  - `PETSC_VERSION`: Version used for PETSc. For instance, `3.19.5`.

These variables are setup like in `load-configs.sh`, and you should probably have them in your `.bashrc`. Although not compulsory, we also recommend setting up your Julia depot like in `load-configs.sh` or you might run with memory problems in your home folder when compiling.

After these variables have been setup, you can load the necessary modules for MPI with the scripts `load-intel.sh` (Intel MPI) or `load-ompi.sh` (OpenMPI).

We also provide scripts `install-p4est.sh` and `install-petsc.sh` that using the given variables installs the selected configuration for teh libraries in your home directory `$HOME/bin/library/version-mpiversion`.

To load julia, you might add `module load julia/X.Y.Z` to your `.bashrc` or setup your own binaries in `$HOME/bin/julia/X.Y.Z` and then add it to your path.

## MPICH Usage
Setonix uses Cray-MPICH. To tell MPI.jl that this is the case, use
```julia
MPIPreferences.use_system_binary(mpiexec="srun", vendor="cray")
```

When running multi-node jobs the following enviroment variables must be set:
```bash
export MPICH_OFI_STARTUP_CONNECT=1
export MPICH_OFI_VERBOSE=1
```

Executing an MPI application is then done via
```bash
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS julia ...
```
where `SLURM_JOB_NUM_NODES` and `SLURM_NTASKS` are set via
```bash
#SBATCH --ntasks={{:ncpus}}
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
```

Here we exclusively use full nodes. There is another example of shared nodes in `TestEnvironment/job.sh`. See this [link](https://pawsey.atlassian.net/wiki/spaces/US/pages/51927426/Example+Slurm+Batch+Scripts+for+Setonix+on+CPU+Compute+Nodes#ExampleSlurmBatchScriptsforSetonixonCPUComputeNodes-Exclusiveaccesstothenode.1) for further info.

## Interactive jobs
These can be launched with the `salloc` command. E.g.,
```
salloc -p debug --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=16G
```

## Misc (PBS -> Slurm)
- Submit a batch job: `qsub` -> `sbatch`
- Submit an interactive job: `qsub -I` -> `salloc`
- Delete a job: `qdel <job id>` -> `scancel <job id>`
- Job status: `qstat` -> `squeue`
- Hold a job: `qhold <job id>` -> `scontrol hold <job id>`
- Release a job: `qrls <job id>` -> `scontrol release <job id>`
- Cluster status: `qstat -B` -> `sinfo`

## Issues
- P4est 2.8.6 or greater must be installed. Previous version do not find the `srun` command and fail to configure.