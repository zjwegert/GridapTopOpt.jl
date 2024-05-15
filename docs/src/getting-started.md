# Getting started

## Installation

`GridapTopOpt.jl` and additional dependencies can be installed in an existing Julia environment using the package manager. This can be accessed in the Julia REPL (read-eval–print loop) by pressing `]`. We then add the required packages via:
```
pkg> add GridapTopOpt, Gridap, GridapDistributed, GridapPETSc, GridapSolvers, PartitionedArrays, SparseMatricesCSR
```
Once installed, serial driver scripts can be run immediately, whereas parallel problems also require an MPI installation.

### MPI
For basic users, [`MPI.jl`](https://github.com/JuliaParallel/MPI.jl) provides such an implementation and a Julia wrapper for `mpiexec` - the MPI executor. This is installed via:
```
pkg> add MPI
julia> using MPI
julia> MPI.install_mpiexecjl()
```
Once the `mpiexecjl` wrapper has been added to the system `PATH`, MPI scripts can be executed in a terminal via
```
mpiexecjl -n P julia  main.jl
```
where `main` is a driver script, `P` denotes the number of processors.

### PETSc
In `GridapTopOpt.jl` we rely on the [`GridapPETSc.jl`](https://github.com/gridap/GridapPETSc.jl) satellite package to interface with the linear and nonlinear solvers provided by the PETSc (Portable, Extensible Toolkit for Scientific Computation) library. For basic users these solvers are provided by `GridapPETSc.jl` with no additional work.

### Advanced installation
For more advanced installations, such as use of a custom MPI/PETSc installation on a HPC cluster, we refer the reader to the [discussion](https://github.com/gridap/GridapPETSc.jl) for `GridapPETSc.jl` and the [configuration page](https://juliaparallel.org/MPI.jl/stable/configuration/) for `MPI.jl`.

## Usage and tutorials
In order to get familiar with the library we recommend following the numerical examples described in:

> Zachary J. Wegert, Jordi Manyer, Connor Mallon, Santiago Badia, and Vivien J. Challis (2024). "GridapTopOpt.jl: A scalable computational toolbox for level set-based topology optimisation". In preparation.

In addition, there are several driver scripts available in `/scripts/..`

More general tutorials for familiarising ones self with Gridap are available via the [Gridap Tutorials](https://gridap.github.io/Tutorials/dev/).

## Known issues
- PETSc's GAMG preconditioner breaks for split Dirichlet DoFs (e.g., x constrained while y free for a single node). There is no simple fix for this. We recommend instead using MUMPS or another preconditioner for this case.
- Currently, our implementation of automatic differentiation does not support multiplication and division of optimisation functionals. We plan to add this in a future release of `GridapTopOpt.jl` -- Issue [#38](https://github.com/zjwegert/GridapTopOpt.jl/issues/38).
- Analytic gradient breaks in parallel for integrals of certain measures -- Issue [#46](https://github.com/zjwegert/GridapTopOpt.jl/issues/46)