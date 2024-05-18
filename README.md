# GridapTopOpt - `Wegert_et_al_2024` branch
This branch contains the scripts and results for the referenced paper below and is locked to ensure that scripts, source code, and results matches that of the paper. Scripts and results are available as follows:
- `scripts/` contains `Paper_scripts/` and `Benchmarks/` that are used to generate the results from Section 4 and 5, respectively.
- `results/` contains the benchmark results and job logs for Section 5 in `Benchmarks/`. The results for Section 4 are held in a [separate repository](https://github.com/zjwegert/Wegert_et_al_2024_Results) as the visualisation files (`.vtu`) are large.

## Installation/Usage
We recommend using this branch of the package in develop mode, this ensures that files are readily available in `/.julia/dev/GridapTopOpt/`. Note that the `.julia` folder is located in the users home directory. For users who are familiarising themselves with Julia please follow these installation instructions:
1. Add the package in developer mode by pressing `]` and running `pkg> dev GridapTopOpt`.
2. Switch to this branch by running `git checkout Wegert_et_al_2024` in `/.julia/dev/GridapTopOpt/`.
3. Copy scripts from `/scripts/Paper_scripts/` to a convenient directory. **Note**: from this point, usage/installation is as in the paper. We outline this below for completeness of this guide.
4. Install the full set of package dependencies:
    - Add packages by running `pkg> add MPI, GridapDistributed, GridapPETSc, GridapSolvers, PartitionedArrays, SparseMatricesCSR`.
    - Install `mpiexecjl` via `using MPI; MPI.install_mpiexecjl()` and add `/.julia/bin/` to the system path.

Scripts can then be run from your convenient directory as in the paper. E.g.,
- `julia therm_comp_serial.jl` for a serial topology optimisation problem; or
- `mpiexecjl -n 4 julia therm_comp_MPI.jl results/` for an MPI problem.

**Note for Windows users**: You will need a bash shell to run `mpiexecjl`. A convenient shell is provided by [Git Bash](https://gitforwindows.org/).

## Software versions
Results in the paper were generated in Linux with the following software versions on Gadi@NCI:
- `Julia v1.9.4`
- `PETSc v3.19.5`
- `Intel MPI v2021.10.0`
