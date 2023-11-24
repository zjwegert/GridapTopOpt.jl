# LSTO_Distributed

## To-do for paper:

- [ ] Benchmark and scaling tests
- [ ] Implement optimisation problems and test:

  |          **Problem type**         | **2D**  | **3D**  | **Serial** | **MPI** |
  |:---------------------------------:|---------|---------|:----------:|---------|
  | Minimum thermal compliance        | &#9745; | &#9744; | &#9745;    | &#9745; |
  | Minimum elastic compliance        | &#9745; | &#9744; | &#9745;    | &#9744; |
  | Inverter mechanism                | &#9745; | &#9744; | &#9745;    | &#9744; |
  | Elastic inverse homogenisation    | &#9745; | &#9744; | &#9745;    | &#9744; |
  | Minimum NL thermal compliance (?) | &#9744; | &#9744; | &#9744;    | &#9744; |

- [ ] Testing automatic differentation
  * &#9744; Test inverter mechanism problem with analytic adjoint. 
  * &#9745; Compare to ray integrals for multi-material ([Paper](https://doi.org/10.1051/cocv/2013076)) and additive manufacturing ([Paper](https://doi.org/10.1051/m2an/2019056)) problems. Edit: We cannot use AD for these. Related to discussion below.
- [ ] Implement HPM ([Paper](https://doi.org/10.1007/s00158-023-03663-0))
- [ ] Implement NonlinearFEStateSpace

## Future work (not this paper)
- [ ] Implement CutFEM in serial using GridapEmbedded.
- [ ] Extend to parallel (I'm sure this will be a massive undertaking)

## Known bugs
- [ ] Higher order FEs breaks `Advection.jl` in parallel.
- [ ] `create_dof_permutation` doesn't work for periodic models.
- [ ] There appears to be a memory leak in `write_vtk`.

## Some notes on auto differentiation

The shape derivative of $J^\prime(\Omega)$ can be defined as the Frechet derivative under a mapping $\tilde\Omega = (\boldsymbol{I}+\boldsymbol{\theta})(\Omega)$ at $\boldsymbol{0}$ with

$$J(\tilde{\Omega})=J(\Omega)+J^\prime(\Omega)(\boldsymbol{\theta})+o(\lVert\boldsymbol{\theta}\rVert).$$

Consider some illistrative examples. First suppose that $J_1(\Omega)=\int_\Omega f(\boldsymbol{x})\~\mathrm{d}\boldsymbol{x}$, then

$$J_1(\tilde{\Omega})=\int_{\tilde{\Omega}}f(\tilde{\boldsymbol{x}})\~\mathrm{d}\boldsymbol{x}=\int_{\Omega}f(\boldsymbol{x}+\boldsymbol{\theta})\left\lvert\frac{\partial\boldsymbol{\tilde{x}}}{\partial\boldsymbol{x}}\right\rvert\~\mathrm{d}\boldsymbol{x}=...=J(\Omega)+\int_{\partial\Omega}f(\boldsymbol{x})\~\boldsymbol{\theta}\cdot\boldsymbol{n}\~\mathrm{d}s+...$$

So, for this simple case $$J_1^\prime(\Omega)(\boldsymbol{\theta})=\int_{\partial\Omega}f(\boldsymbol{x})\~\boldsymbol{\theta}\cdot\boldsymbol{n}\~\mathrm{d}s\~\~(\star).$$ For a surface integral $J_2(\Omega)=\int_{\partial\Omega} f(\boldsymbol{x})\~\mathrm{d}s$, one finds in a similar way $$J_2^\prime(\Omega)(\boldsymbol{\theta})=\int_{\partial\Omega}(f\nabla\cdot\boldsymbol{n}+\nabla f\cdot\boldsymbol{n})\~\boldsymbol{\theta}\cdot\boldsymbol{n}\~\mathrm{d}s.$$

Suppose that we attribute a signed distance function $\varphi:D\rightarrow\mathbb{R}$ to our domain $\Omega\subset D$ with $\bar{\Omega}=\lbrace \boldsymbol{x}:\varphi(\boldsymbol{x})\leq0\rbrace$ and $\Omega^\complement=\lbrace \boldsymbol{x}:\varphi(\boldsymbol{x})>0\rbrace$. We can then define a smooth characteristic function $\chi_\epsilon:D\rightarrow[\epsilon,1]$ as $\chi_\epsilon(\boldsymbol{x})=1-H(\varphi(\boldsymbol{x}))+\epsilon H(\varphi(\boldsymbol{x}))$ where $H$ is a smoothed Heaviside function and $\epsilon\ll1$ allows for an ersatz material approximation but of course can be taken as zero depending on the computational regime. We can now rewrite $J_1$ over the computational domain $D$ as $J_{1\Omega}(\varphi)=\int_D \chi_\epsilon(\varphi)f(\boldsymbol{x})\~\mathrm{d}\boldsymbol{x}$. Considering the directional derivative of $J_{1\Omega}$ under a variation $\tilde{\varphi}=\varphi+sv$ gives

$$J_{1\Omega}^\prime(\varphi)(v)=\frac{\mathrm{d}}{\mathrm{d}s} J_{1\Omega}(\varphi+sv)\rvert_{s=0}=\int_D v\chi_\epsilon^\prime(\varphi)f(\boldsymbol{x})\~\mathrm{d}\boldsymbol{x}$$

or

$$J_{1\Omega}^\prime(\varphi)(v)=\int_D (\epsilon-1)vH^\prime(\varphi)f(\boldsymbol{x})\~\mathrm{d}\boldsymbol{x}$$

The final result of course does not yet match $(\star)$. Over a fixed computational domain we may relax integrals to be over all of $D$ via $\mathrm{d}s = H'(\varphi)\lvert\nabla\varphi\rvert\~\mathrm{d}\boldsymbol{x}$. In addition suppose we take $\boldsymbol{\theta}=-v\boldsymbol{n}$. Then $(\star)$ can be rewritten as
$$J_1^\prime(\Omega)(v\boldsymbol{n})=-\int_{D}vf(\boldsymbol{x})H'(\varphi)\lvert\nabla\varphi\rvert\~\mathrm{d}s$$

As $\varphi$ is a signed distance function we have $\lvert\nabla\varphi\rvert=1$ for $D\setminus\Sigma$ where $\Sigma$ is the skeleton of $\Omega$ and $\Omega^\complement$. Furthermore, $H'(\varphi)$ provides support only within a band of $\partial\Omega$. Therefore, we have that almost everywhere

$${\huge{|}} J_1^\prime(\Omega)(v\boldsymbol{n}) - J_{1\Omega}^\prime(\varphi)(v){\huge{|}}=O(\epsilon),$$

with equaility as $\epsilon\rightarrow0$.

This is not the case when we consider applying the same process to a surface integral such as $J_2(\Omega)$. In this case, we can never recover $\nabla f$ under a variation of $\varphi$. For argument sake we can take $J_{2\Omega}(\varphi)=\int_D f(\boldsymbol{x})H'(\varphi)\lvert\nabla\varphi\rvert\~\mathrm{d}\boldsymbol{x}$. Then a variation in $\varphi$ gives
$$J_{2\Omega}^\prime(\varphi)(v)=\int_D f(\boldsymbol{x})\left(H^{\prime\prime}(\varphi)\lvert\nabla\varphi\rvert v + H^\prime(\varphi)\frac{\nabla v\cdot\nabla \varphi}{\lvert\nabla\varphi\rvert}\right)\~\mathrm{d}\boldsymbol{x}.$$
On the other hand, relaxing the shape derivative of $J_2$ in the same way as above gives
$$J_2^\prime(\Omega)(-v\boldsymbol{n})=\int_{D}-\left(f\nabla\cdot\frac{\nabla\varphi}{\lvert\nabla\varphi\rvert}+\nabla f\cdot\frac{\nabla\varphi}{\lvert\nabla\varphi\rvert}\right)vH'(\varphi)\lvert\nabla\varphi\rvert\~\mathrm{d}\boldsymbol{x}.$$

### Adding PDE constraints
Consider $\Omega\subset D$ with $J(\Omega,u)=\int_\Omega j(u)\~\mathrm{d}\boldsymbol{x}+\int_\Gamma l(u)\~\mathrm{d}\Gamma$ where $u$ satisfies

$$
\begin{aligned}
-\mathrm{div}(C\varepsilon(\boldsymbol{u})) &= f\text{ on }\Omega, \\
u &= 0\text{ on }\Gamma_D,\\
C\varepsilon(\boldsymbol{u})\boldsymbol{n} &= 0\text{ on }\Gamma_0,\\
C\varepsilon(\boldsymbol{u})\boldsymbol{n} &= g\text{ on }\Gamma_N,\\
C\varepsilon(\boldsymbol{u})\boldsymbol{n} &= w(u)\text{ on }\Gamma_R.\\
\end{aligned}
$$

In the above $\varepsilon$ is the strain tensor, $C$ is the stiffness tensor, $\Gamma_0 = \partial\Omega\setminus(\Gamma\cup\Gamma_D\cup\Gamma_N\cup\Gamma_R)$, and that $\Gamma$ is disjoint from the other partitions of $\partial\Omega$ for simplicity.

### A rule of thumb

### What is not captured by a Gateaux derivative of $J$ at $\varphi$ in the direction $v$?

<br /><br /><br /><br /><br /><br /><br /><br /><br />
---------------------------------------------------------------------------------------------------
# Deprecated:
## Setup:

1. Download and activate repo in Julia using `]` followed by `activate .`.
2. Download dependencies using `]` followed by `instantiate`.
3. Configure    
    * (Desktop) install `mpiexecjl` by running `using MPI; MPI.install_mpiexecjl()` in Julia. Add `.../.julia/bin/` to system path.
    * (HPC) It is best to utilise pre-exisiting MPI and PETSc installations on the HPC. The PBS Pro script provided below builds required packges with system settings.
4. Run 
    * (Desktop) `mpiexecjl` from terminal (GIT Bash if on Windows). E.g.,
        `mpiexecjl --project=. -n 4 julia MainMPI.jl pipe_setup_2d 2 2`
    * (HPC) Queue a run script - see examples provided.

## Options:
We typically execute `mpiexecjl` from the terminal via the following
    `mpiexecjl --project=. -n NP julia MainMPI.jl SETUP_NAME NX NY NZ`
where
* `NP` -> Number of processors to launch,
* `SETUP_NAME` -> Name of setup function,
* `NX` -> Number of partitions in X,
* `NY` -> Number of partitions in Y,
* `NZ` -> Number of partitions in Z (optional).


`MainMPI` can also be replaced by `MainSerial` to run the code in serial mode (currently this is quite slow?).

## Output & Visualisation
Output files consist of a `.csv` file containing the iteration history, `.pvtu` files and `.vtu` files in corresponding subdirectories. For the augmented Lagrangian method the `history.csv` file consists of three columns for the objective, volume, and Lagrangian respectively.

Results can be visualised in Paraview via the following:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Open (any `.pvtu`) -> Apply -> `CTRL + Space` -> `Iso Volume`

Use the following settings in the `Iso Volume` filter: `Input Scalars` = `phi`, `Minimum` = -1, `Maximum` = 0.

Bug: Currently `.pvtu` files contain the whole path to the corresponding `.vtu` instead of the relative path. This is caused by passing the whole path to `write_vtk`. Passing the relative path to `write_vtk` includes the location folder in the path to each piece (`.vtu` file). If moving several visualisation files use the command (replace `/` by `\/` in `PATH` below)
    `sed -i 's/PATH/./g' *.pvtu`

## Algorithm Overview:

The flowchart below gives a rough overview of the augmented Lagrangian-based optimisation algorithm for the problem `pipe_setup_2d` with objective `thermal_compliance`.

![Screenshot](Images/Algorithm.png)

## HPC PBS Scripts:

The following PBS script is provided for setting up `MPI.jl` in conjuction with `GridapDistributed.jl`, `GridapPETSc.jl`, and `MUMPS` on a PBS Pro-based HPC cluster:
```
#!/bin/bash -l

#PBS -P LSTO_Distributed_Setup
#PBS -l ncpus=1
#PBS -l mem=32GB
#PBS -l walltime=00:30:00
#PBS -j oe

module load julia/1.8.3-linux-x86_64
module load mumps/5.5.1-foss-2022a-metis
module load petsc/3.18.6-foss-2022a
module load openmpi/4.1.4-gcc-11.3.0

# Path to PETSc Library - replace with appropriate system path
export JULIA_PETSC_LIBRARY=$EBROOTPETSC/lib/libpetsc.so 

cd $PBS_O_WORKDIR

julia --project=. -e 'using Pkg; Pkg.instantiate()'
echo '------------------------------------------------------------'
julia --project -e 'using Pkg; Pkg.add("MPIPreferences"); 
    using MPIPreferences; MPIPreferences.use_system_binary()'
echo '------------------------------------------------------------'
julia --project=. -e 'using MPI; MPI.install_mpiexecjl()'
echo '------------------------------------------------------------'
julia --project=. -e 'using Pkg; Pkg.build("MPI"; verbose=true);
    Pkg.build("GridapPETSc")'
echo '------------------------------------------------------------'
julia --project=. -e 'using MPI, Gridap, GridapDistributed, GridapPETSc'
echo '------------------------------------------------------------'
echo 'Done'
```

The following PBS script can be used to run the code on a PBS Pro-based HPC:
```
#!/bin/bash -l

#PBS -P LSTO_Dist_Pipe
#PBS -l cputype=7713
#PBS -l ncpus=27
#PBS -l mpiprocs=27
#PBS -l mem=128GB
#PBS -l walltime=48:00:00
#PBS -j oe

module load julia/1.8.3-linux-x86_64
module load mumps/5.5.1-foss-2022a-metis
module load petsc/3.18.6-foss-2022a
module load openmpi/4.1.4-gcc-11.3.0

# Path to PETSc Library - replace with appropriate system path
export JULIA_PETSC_LIBRARY=$EBROOTPETSC/lib/libpetsc.so
export mpiexecjl=~/.julia/bin/mpiexecjl

cd $PBS_O_WORKDIR

$mpiexecjl --project=. -n 27 julia ./MainMPI.jl pipe_setup_3d 3 3 3
```
