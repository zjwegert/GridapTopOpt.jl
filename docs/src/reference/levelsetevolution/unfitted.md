# Unfitted schemes

!!! warning
    This page is still under construction! Please refer to the tutorial
    for a discussion of functionality.

## Level-set evolution
In the following, we use the approach for evolution proposed by [Burman et al. (2018)](https://doi.org/10.1016/j.cma.2017.09.005) to update the level-set function. Namely, we evolve the level-set function by solving a transport equation
```math
\begin{equation}
    \begin{cases}
        \displaystyle\frac{\partial\phi(t,\boldsymbol{x})}{\partial t}+\boldsymbol{\beta}\cdot\boldsymbol{\nabla}\phi(t,\boldsymbol{x})=0,\\
        \phi(0,\boldsymbol{x})=\phi_0(\boldsymbol{x}),\\
        \boldsymbol{x}\in D,~t\in(0,T),
    \end{cases}
\end{equation}
```
where $\boldsymbol{\beta}$ is a velocity field. In this work $\boldsymbol{\beta}$ is computed as $\boldsymbol{\beta}=\boldsymbol{n}g_\Omega$,
where $g_\Omega$ is the regularised sensitivity resulting from the [Hilbertian extension-regularisation approach](../velext.md). The transport equation above is solved using an interior penalty approach and Crank-Nicolson for the discretisation in time (see [Burman et al. (2018)](https://doi.org/10.1016/j.cma.2017.09.005) and references there in for further discussion). The weak formulation of this problem is: for all $t\in(0,T)$, find $\phi\in W$ such that
```math
\int_D\left[v\frac{\partial\phi}{\partial t}+v\boldsymbol{\beta}\cdot\boldsymbol{\nabla}\phi\right]~\mathrm{d}\boldsymbol{x}+\sum_{F\in\mathscr{S}_h}\int_Fc_eh_F^2\lvert\boldsymbol{n}_F\cdot\boldsymbol{\beta}\rvert\llbracket \boldsymbol{n}_F\cdot\boldsymbol{\nabla}\phi\rrbracket\llbracket\boldsymbol{n}_F\cdot\boldsymbol{\nabla}v\rrbracket~\mathrm{d}s=0,~\forall v\in W^h,
```
where $\mathscr{S}_h$ is the set of interior mesh facets, $h_F$ is the average element diameters of the elements sharing a facet, $\boldsymbol{n}_F$ is the normal to the facet $F$, $\llbracket v\rrbracket = v^+-v^-$ is the jump in a function $v$ over the facet $F$, and $c_e$ is a stabilisation coefficient. Further discussion regarding solving this can be found at [Wegert et al. (2025)](https://doi.org/10.1016/j.cma.2025.118203).

```@autodocs
Modules = [GridapTopOpt]
Pages = ["LevelSetEvolution/UnfittedEvolution/CutFEMEvolver.jl"]
```

## Level-set reinitialisation
We solve the reinitialisation equation
```math
\left\lbrace\begin{aligned}
    \frac{\partial\phi(t,\boldsymbol{x})}{\partial t} + \operatorname{sign}(\phi_0(\boldsymbol{x}))\left(\lvert\boldsymbol{\nabla}\phi(t,\boldsymbol{x})\rvert-1\right) = 0,\\
    \phi(0,\boldsymbol{x})=\phi_0(\boldsymbol{x}),\\
    \boldsymbol{x}\in D, t>0.
\end{aligned}\right.
```
To steady state using an approach based on the one proposed by [Mallon et al. (2025)](https://doi.org/10.1002/nme.70004). The weak formulation for this problem is given as: find $\phi\in W$ such that
```math
    \int_Dv\boldsymbol{w}\cdot\boldsymbol{\nabla}\phi-v\operatorname{sign}(\phi_0)~\mathrm{d}\boldsymbol{x}+\int_\Gamma \frac{\gamma_d}{h}\phi v~\mathrm{d}s+j(\phi,v)=0,~\forall v\in W^h,
```
where $\boldsymbol{w}=\operatorname{sign}(\phi_0)\frac{\boldsymbol{\nabla}\phi}{\lVert\boldsymbol{\nabla}\phi\rVert}$. Further discussion regarding solving this can be found at [Wegert et al. (2025)](https://doi.org/10.1016/j.cma.2025.118203).

```@autodocs
Modules = [GridapTopOpt]
Pages = ["LevelSetEvolution/UnfittedEvolution/StabilisedReinitialiser.jl"]
```

### Custom `UnfittedEvolution`
```@docs
GridapTopOpt.Evolver
GridapTopOpt.solve!(::GridapTopOpt.Evolver,φ,args...)
GridapTopOpt.Reinitialiser
GridapTopOpt.solve!(::GridapTopOpt.Reinitialiser,φ,args...)
```