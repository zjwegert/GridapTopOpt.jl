# GridapTopOpt - `Wegert_et_al_2024` branch
This branch contains the scripts and results for the referenced paper below and is locked to ensure that scripts, source code, and results matches that of the paper. Scripts and results are available as follows:
- `scripts/` contains `Paper_scripts/` and `Benchmarks/` that are used to generate the results from Section 4 and 5, respectively.
- `results/` contains the benchmark results and job logs for Section 5 in `Benchmarks/`. The results for Section 4 are held in a [separate repository](https://github.com/zjwegert/Wegert_et_al_2024_Results) as the visualisation files (`.vtu`) are large.

## Installation/Usage
Currently we expect this branch to be used as the Julia environment (see *). For first time setup:
1. Clone/download this branch to a directory (e.g., `.../GridapTopOpt.jl/`).
2. Launch Julia from inside `.../GridapTopOpt.jl/`.
3. Run `pkg> activate .` followed by `pkg> instantiate`.
4. Install `mpiexecjl` as in the manuscript.

Scripts can then be run from `.../GridapTopOpt.jl/` with the `--project` flag. E.g.,
- `julia --project ./scripts/Paper_scripts/therm_comp_serial.jl` for a serial TO problem; or
- `mpiexecjl --project -n 4 julia ./scripts/Paper_scripts/therm_comp_serial.jl results/` for an MPI problem.

*: We include the `Manifest.toml` file to ensure the correct branch of `GridapSolvers` is included when instantiating the package. This will be removed in future.

## Software versions
Results in the paper were generated with the following software versions on Gadi:
- `Julia v1.9.4`
- `PETSc v3.19.5`
- `Intel MPI v2021.10.0`

Package versions are as in the Manifest/Project files.