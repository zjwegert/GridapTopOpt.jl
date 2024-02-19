# Minimum thermal compliance

The goal of this tutorial is to learn
- How to formulate a topology optimisation problem
- How to describe the problem over a fixed computational domain ``D`` via the level-set method.
- How to setup and solve the problem in LevelSetTopOpt

We consider the following extensions at the end of the tutorial:
- How to extend the problem to 3D and utilise PETSc solvers
- How to solve problems with nonlinear state equations with automatic differentiation
- How to run in MPI mode

We will first consider formulation of the state equations and a topology optimisation problem in a continuous setting. We will then discretise via a level set function in a fixed computational regime. Note that this approach corresponds to an "optimise-then-discretise" approach [4] where shape derivatives are calculated analytically in the continuous space then relaxed via a level set function ``\varphi``. Automatic differentiation can be used to calculate these quantities and is discussed [here](../usage/ad.md).

## State equations

The homogeneous steady-state heat equation (equivalently Laplace's equation) is perhaps one of the most well-understood partial differential equations and usually the first introduced to an undergraduate student in applied mathematics. For this reason, we will use it to describe the heat transfer through a solid and how one comes to the notion of optimising the shape of that solid.

Consider the geometric conditions outlined in the Figure 1 and suppose that we prescribe the following conditions:
- *Heat source*: unitary normal heat flow across ``\Gamma_{N}``.
- *Insulating*: zero normal heat flow across $\partial\Omega\setminus\Gamma_N$,
- *Heat sink*: zero heat on ``\Gamma_D``.

| ![](2d_min_thermal_comp_setup.png) |
|:--:|
|Figure 1: The setup for the two-dimensional minimum thermal compliance problem|

Physically we can imagine this as describing the transfer of heat through a domain ``\Omega`` from the sources to the sinks. From a mathematical perspective, we can write down the partial differential equations describing this as

```math
\begin{aligned}
-\nabla(\kappa\nabla u) &= 0~\text{in }\Omega,\\
\kappa\nabla u\cdot\boldsymbol{n} &= g~\text{on }\Gamma_N,\\
\kappa\nabla u\cdot\boldsymbol{n} &= 0~\text{on }\partial\Omega\setminus\Gamma_N,\\
u &= 0~\text{on }\Gamma_D.
\end{aligned}
```

where ``\kappa`` is the diffusivity through ``\Omega`` and ``\boldsymbol{n}`` is the unit normal on the boundary. The weak formulation of the above strong formulation can be found by multiplying by a test function $v$ and applying integration by parts. This gives

```math
\begin{aligned}
&\textit{Find }u\in H^1_{\Gamma_D}(\Omega)\textit{ such that}\\
&\int_{\Omega}\kappa\boldsymbol{\nabla}u\cdot\boldsymbol{\nabla}v~\mathrm{d}\boldsymbol{x} = \int_{\Gamma_N}gv~\mathrm{d}s,~\forall v\in H^1_{\Gamma_D}(\Omega)
\end{aligned}
```

where ``H^1_{\Gamma_D}(\Omega)=\{v\in H^1(\Omega):~v=0\text{ on }\Gamma_D\}``.

## Optimisation problem

For this tutorial, we consider minimising the thermal compliance (or dissipated energy) as discussed in [1,2]. The corresponding optimisation problem is

```math
\begin{aligned}
\min_{\Omega\in\mathcal{U}}&~J(\Omega)=\int_{\Omega}\kappa\lvert\boldsymbol{\nabla}u\rvert^2~\mathrm{d}\boldsymbol{x}\\
\text{s.t. }&~\operatorname{Vol}(\Omega)=V_f,\\
&\left\{
\begin{aligned}
&\textit{Find }u\in H^1_{\Gamma_D}(\Omega)\textit{ such that}\\
&\int_{\Omega}\kappa\boldsymbol{\nabla}u\cdot\boldsymbol{\nabla}v~\mathrm{d}\boldsymbol{x} = \int_{\Gamma_N}gv~\mathrm{d}s,~\forall v\in H^1_{\Gamma_D}(\Omega)
\end{aligned}
\right.
\end{aligned}
```
where ``\operatorname{Vol}(\Omega)=\int_\Omega1~\mathrm{d}\boldsymbol{x}``. This objective is equivalent to equivalent to maximising the heat transfer efficiency through ``\Omega``. 

## Shape differentiation

We consider the change in quantities under the variation of the domain using *shape derivatives*. For the purpose of this tutorial we will give the mathematical description of a shape derivative along with the shape derivatives of the functionals ``J`` and ``\operatorname{Vol}``. Further discussion can be found in [3,4].

Suppose that we consider smooth variations of the domain ``\Omega`` of the form ``\Omega_{\boldsymbol{\theta}} =(\boldsymbol{I}+\boldsymbol{\theta})(\Omega)``, where ``\boldsymbol{\theta} \in W^{1,\infty}(\mathbb{R}^d,\mathbb{R}^d)``. Then the following definition and lemma follow:

!!! note "Definition [3]"
    The shape derivative of ``J(\Omega)`` at ``\Omega`` is defined as the Fréchet derivative in ``W^{1, \infty}(\mathbb{R}^d, \mathbb{R}^d)`` at ``\boldsymbol{\theta}`` of the application ``\boldsymbol{\theta} \rightarrow J(\Omega_{\boldsymbol{\theta}})``, i.e.,
    ```math
    J(\Omega_{\boldsymbol{\theta}})(\Omega)=J(\Omega)+J^{\prime}(\Omega)(\boldsymbol{\theta})+\mathrm{o}(\boldsymbol{\theta})  
    ```
    with ``\lim _{\boldsymbol{\theta} \rightarrow 0} \frac{\lvert\mathrm{o}(\boldsymbol{\theta})\rvert}{\|\boldsymbol{\theta}\|}=0,`` where the shape derivative ``J^{\prime}(\Omega)`` is a continuous linear form on ``W^{1, \infty}(\mathbb{R}^d, \mathbb{R}^d)``


The shape derivatives of ``J`` and ``\operatorname{Vol}`` are then

```math
J'(\Omega)(\boldsymbol{\theta}) = -\int_{\Gamma}\kappa\boldsymbol{\nabla}(u)\cdot\boldsymbol{\nabla}(u)~\boldsymbol{\theta}\cdot\boldsymbol{n}~\mathrm{d}s
```
and
```math
\operatorname{Vol}'(\Omega)(\boldsymbol{\theta}) = \int_{\Gamma}\boldsymbol{\theta}\cdot\boldsymbol{n}~\mathrm{d}s
```
where ``\Gamma = \partial\Omega\setminus(\Gamma_D\cup\Gamma_N)``. The first of these follows from Céa's formal method (see discussion in [3,4]), while the latter result follows from application of Lemma 4 of [3]. Finally, taking a deformation field according to ``\boldsymbol{\theta}=-q\boldsymbol{n}`` amounts to a descent direction according to the definition above. This gives
```math
J'(\Omega)(-q\boldsymbol{n}) = \int_{\Gamma}q\kappa\boldsymbol{\nabla}(u)\cdot\boldsymbol{\nabla}(u)~\mathrm{d}s
```
and
```math
\operatorname{Vol}'(\Omega)(-q\boldsymbol{n}) = -\int_{\Gamma}q~\mathrm{d}s.
```

## Discretisation via a level set 

Suppose that we attribute a level set function ``\varphi:D\rightarrow\mathbb{R}`` to our domain ``\Omega\subset D`` with ``\bar{\Omega}=\lbrace \boldsymbol{x}:\varphi(\boldsymbol{x})\leq0\rbrace`` and ``\Omega^\complement=\lbrace \boldsymbol{x}:\varphi(\boldsymbol{x})>0\rbrace``. We can then define a smooth characteristic function ``I:\mathbb{R}\rightarrow[\epsilon,1]`` as ``I(\varphi)=(1-H(\varphi))+\epsilon H(\varphi)`` where ``H`` is a smoothed Heaviside function with smoothing radius ``\eta``, and ``\epsilon\ll1`` allows for an ersatz material approximation. Of course, ``\epsilon`` can be taken as zero depending on the computational regime. Over the fixed computational domain we may relax integrals to be over all of ``D`` via ``\mathrm{d}\boldsymbol{x}= H(\varphi)~\mathrm{d}\boldsymbol{x}`` and ``\mathrm{d}s = H'(\varphi)\lvert\nabla\varphi\rvert~\mathrm{d}\boldsymbol{x}``. The above optimisation problem then rewrites in terms of ``\varphi`` as

```math
\begin{aligned}
\min_{\varphi}&~J(\varphi)=\int_{D}I(\varphi)\kappa\lvert\boldsymbol{\nabla}u\rvert^2~\mathrm{d}\boldsymbol{x}\\
\text{s.t. }&~C(\varphi)=0,\\
&\left\{
\begin{aligned}
&\textit{Find }u\in H^1_{\Gamma_D}(D)\\
&\int_{D}I(\varphi)\kappa\boldsymbol{\nabla}(u)\cdot\boldsymbol{\nabla}(v)~\mathrm{d}\boldsymbol{x} = \int_{\Gamma_N}v~\mathrm{d}s,~\forall v\in H^1_{\Gamma_D}(D)
\end{aligned}
\right.
\end{aligned}
```

where we retain an exact triangulation and measure of  ``\Gamma_N`` as this is a fixed boundary. In addition, we have rewritten the volume constraint as

```math
\begin{aligned}
C(\varphi)&=\int_D (\rho(\varphi) - V_f)/\operatorname{Vol}(D)~\mathrm{d}\boldsymbol{x}\\
&=\int_D \rho(\varphi)~\mathrm{d}\boldsymbol{x}/\operatorname{Vol}(D) - V_f\\
&=\int_\Omega~\mathrm{d}\boldsymbol{x}-V_f = \operatorname{Vol}(\Omega)-V_f
\end{aligned}
``` 

where ``\rho(\varphi)=1-H(\varphi)`` is the smoothed volume density function.

!!! note
    In LevelSetTopOpt we assume constraints are of the integral form above.

The shape derivatives from the previous section can be relaxed over the computational domain as 

```math
J'(\varphi)(-q\boldsymbol{n}) = \int_{D}q\kappa\boldsymbol{\nabla}(u)\cdot\boldsymbol{\nabla}(u)H'(\varphi)\lvert\nabla\varphi\rvert~\mathrm{d}\boldsymbol{x}
```
and
```math
C'(\varphi)(-q\boldsymbol{n}) = -\int_{D}wH'(\varphi)\lvert\nabla\varphi\rvert~\mathrm{d}\boldsymbol{x}/\operatorname{Vol}(D).
```

## Computational method

In the following, we discuss the implementation of the above optimisation problem in LevelSetTopOpt. For the purpose of this tutorial we break the computational formulation into chunks.

The first step in creating our script is to load any packages required:
```julia
using LevelSetTopOpt, Gridap
```

### Parameters
The following are user defined parameters for the problem. 
These parameters will be discussed over the course of this tutorial. 

```julia
# FE parameters
order = 1                                       # Finite element order
xmax=ymax=1.0                                   # Domain size
el_size = (200,200)                             # Mesh partition size
prop_Γ_N = 0.4                                  # Γ_N size parameter
prop_Γ_D = 0.2                                  # Γ_D size parameter
f_Γ_N(x) = (x[1] ≈ xmax &&                      # Γ_N indicator function
  ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= ymax/2+ymax*prop_Γ_N/4 + eps())
f_Γ_D(x) = (x[1] ≈ 0.0 &&                       # Γ_D indicator function
  (x[2] <= ymax*prop_Γ_D + eps() || x[2] >= ymax-ymax*prop_Γ_D - eps()))
# FD parameters
γ = 0.1                                         # HJ equation time step coefficient
γ_reinit = 0.5                                  # Reinit. equation time step coefficient
max_steps = floor(Int,minimum(el_size)/10)      # Max steps for advection
tol = 1/(10order^2)*prod(inv,minimum(el_size))  # Advection tolerance
# Problem parameters
κ = 1                                           # Diffusivity
g = 1                                           # Heat flow in
vf = 0.4                                        # Volume fraction constraint
lsf_func = initial_lsf(4,0.2)                   # Initial level set function
iter_mod = 10                                   # Output VTK files every 10th iteration
path = "./results/min_thermal_compliance_tut"   # Output path
mkpath(path)                                    # Create path
```

### Finite element setup
We first create a Cartesian mesh over ``[0,x_{\max}]\times[0,y_{\max}]`` with partition size `el_size` by creating an object `CartesianDiscreteModel`. In addition, we label the boundaries ``\Gamma_D`` and ``\Gamma_N`` using the [`update_labels!`](@ref) function.
```julia
# Model
model = CartesianDiscreteModel((0,xmax,0,ymax),el_size);
update_labels!(1,model,f_Γ_D,"Gamma_D")
update_labels!(2,model,f_Γ_N,"Gamma_N")
```
The first argument of [`update_labels!`](@ref) indicates the label number associated to the region as indicated by the functions `f_Γ_D` and `f_Γ_N`. These functions should take a vector `x` and return `true` or `false` depending on whether a point is present in this region.

Once the model is defined we create an integration mesh and measure for both ``\Omega`` and ``\Gamma_N``. These are built using
```julia
# Triangulation and measures
Ω = Triangulation(model)
Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
dΩ = Measure(Ω,2*order)
dΓ_N = Measure(Γ_N,2*order)
```
where `2*order` indicates the quadrature degree for numerical integration.

The final stage of the finite element setup is the approximation of the finite element spaces. This is given as follows: 
```julia
# Spaces
reffe = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"])
U = TrialFESpace(V,0.0)
V_φ = TestFESpace(model,reffe)
V_reg = TestFESpace(model,reffe;dirichlet_tags=["Gamma_N"])
U_reg = TrialFESpace(V_reg,0)
```

In the above, we first define a scalar-valued Lagrangian reference element. This is then used to define the test space `V` and trial space `U` corresponding to ``H^1_{\Gamma_{D}}(\Omega)``. We then construct an FE space `V_φ` over which the level set function is defined, along with an FE test space `V_reg` and trial space `U_reg` over which derivatives are defined. We require that `V_reg` and `U_reg` have zero Dirichlet boundary conditions over regions where the extended shape sensitivity is zero. In general, we allow Dirichlet boundaries to have non-zero shape sensitivity.

### Initial level set function and interpolant

We interpolate an initial level set function onto `V_φ` given a function `lsf_func` using the `interpolate` provided by Gridap.
```julia
# Level set and interpolator
φh = interpolate(lsf_func,V_φ)
```
For this problem we set `lsf_func` using the function [`initial_lsf`](@ref) in the problem parameters. This generates an initial level set according to
```math
\varphi_{\xi,a}(\boldsymbol{x})=-\frac{1}{4} \prod_i^D(\cos(\xi\pi x_i)) - a/4
```
with ``\xi,a=(4,0.2)`` and ``D=2`` in two dimensions. 

We also generate a smooth characteristic function of radius ``\eta`` using:

```julia
interp = SmoothErsatzMaterialInterpolation(η = 2*maximum(get_el_Δ(model)))
I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ
```

This the [`SmoothErsatzMaterialInterpolation`](@ref) structure defines the characteristic or interpolator `I`, the smoothed Heaviside function `H` and it's derivative `DH`, and the smoothed density function `ρ`. Below we visualise `φh` and the smoothed density function `ρ` at `φh`:

| ![](2d_min_thermal_comp_initial_lsf_combined.png) |
|:--:|
|Figure 2: A visualisation of the initial level set function and the interpolated density function ``\rho`` for ``\Omega``.|

Optional: we can generate a VTK file for visualisation in Paraview via 
```julia
writevtk(Ω,"initial_lsf",cellfields=["phi"=>φh,
  "ρ(phi)"=>(ρ ∘ φh),"|nabla(phi)|"=>(norm ∘ ∇(φh))])
```
Note that the operator `∘` is used to compose other Julia `Functions` with Gridap `FEFunctions`. This will be used extensively as we progress through the tutorial.

### Weak formulation and the state map
The weak formulation for the problem above can be written as
```julia
# Weak formulation
a(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*κ*∇(u)⋅∇(v))dΩ
l(v,φ,dΩ,dΓ_N) = ∫(g*v)dΓ_N
```
where the `∘` operator composes the interpolator `I` with the level set function `φ`.

!!! warning
    The measures must be included as arguments at the end of all functionals.
    This ensures compatibility with Gridap's automatic differentiation.

At this point we can build an [`AffineFEStateMap`](@ref). This structure is designed to
1) Enable the forward problem that solves a Gridap `AffineFEOperator`; and
2) Encode the implicit dependence of the solution `u` on the level set function `φ` to enable the differentiation of `u`.

```
state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh,dΩ,dΓ_N)
```

### Optimisation functionals

The objective functional ``J`` and it's shape derivative is given by

```julia
# Objective and constraints
J(u,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*κ*∇(u)⋅∇(u))dΩ
dJ(q,u,φ,dΩ,dΓ_N) = ∫(-κ*∇(u)⋅∇(u)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
```

while the constraint on the volume and its derivative is

```julia
vol_D = sum(∫(1)dΩ)
C(u,φ,dΩ,dΓ_N) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
dC(q,u,φ,dΩ,dΓ_N) = ∫(1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
```

We can now create an object [`PDEConstrainedFunctionals`](@ref) that handles the objective and constraints, and their analytic or automatic differentiation.
```julia
pcfs = PDEConstrainedFunctionals(J,[C],state_map,analytic_dJ=dJ,analytic_dC=[dC])
```
In this case, the analytic shape derivatives are passed as optional arguments. When these are not given, automatic differentiation in ``φ`` is used.

### Velocity extension-regularisation method

The Hilbertian extension-regularisation [4] method involves solving an 
identification problem over a Hilbert space ``H`` on ``D`` with 
inner product ``\langle\cdot,\cdot\rangle_H``: 
*Find* ``g_\Omega\in H`` *such that* ``\langle g_\Omega,q\rangle_H
=-J^{\prime}(\Omega)(q\boldsymbol{n})~
\forall q\in H.``

This provides two benefits: 
 1) It naturally extends the shape sensitivity from ``\partial\Omega`` 
    onto the bounding domain ``D``; and
 2) ensures a descent direction for ``J(\Omega)`` with additional regularity 
    (i.e., ``H`` as opposed to ``L^2(\partial\Omega)``).

For our problem above we take the inner product

```math
\langle p,q\rangle_H=\int_{D}\alpha^2\nabla(p)\nabla(q)+pq~\mathrm{d}\boldsymbol{x},
```
where ``\alpha`` is the smoothing length scale. Equivalently in our script we have
```julia
# Velocity extension
α = 4*maximum(get_el_Δ(model))
a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ
```

We then build an object [`VelocityExtension`](@ref). This object provides a method [`project!`](@ref) that applies the Hilbertian velocity-extension method to a given shape derivative. 

```julia
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)
```

### Advection and reinitialisation

To advect the level set function, we solve the Hamilton-Jacobi evolution equation [3,4,5]. This is given by

```math
\frac{\partial\phi}{\partial t} + V(\boldsymbol{x})\lVert\boldsymbol{\nabla}\phi\rVert = 0,
```

with ``\phi(0,\boldsymbol{x})=\phi_0(\boldsymbol{x})`` and ``\boldsymbol{x}\in D,~t\in(0,T)``.

After advection of the interface, we solve the reinitialisation equation to find an equivalent signed distance function for the given level set function. This is given by

```math
\frac{\partial\phi}{\partial t} + \mathrm{sign}(\phi_0)(\lVert\boldsymbol{\nabla}\phi\rVert-1) = 0,
```

with ``\phi(0,\boldsymbol{x})=\phi_0(\boldsymbol{x})`` and ``\boldsymbol{x}\in D,~t\in(0,T)``.

Both of these equations can be solved numerically on a Cartesian mesh using a first order Godunov upwind difference scheme based on [5]. This functionality is provided by the following objects:

```julia
# Finite difference scheme
scheme = FirstOrderStencil(2,Float64)
stencil = AdvectionStencil(scheme,model,V_φ,tol,max_steps)
```

In the above we first build an object [`FirstOrderStencil`](@ref) that represents a finite difference stencil for a single step of the Hamilton-Jacobi evolution equation and reinitialisation equation. We then create an [`AdvectionStencil`](@ref) which enables finite differencing on order `O` finite elements in serial or parallel. The [`AdvectionStencil`](@ref) object provides two important methods [`advect!`](@ref) and [`reinit!`](@ref) that correspond to solving the Hamilton-Jacobi evolution equation and reinitialisation equation, respectively.

### Optimiser, visualisation and IO

We may now create the optimiser object. This structure holds all information regarding the optimisation problem that we wish to solve and implements an optimisation algorithm as a Julia [iterator](https://docs.julialang.org/en/v1/manual/interfaces/). For the purpose of this tutorial we use a standard augmented Lagrangian method based on [6]. In our script, we create an instance of the [`AugmentedLagrangian`](@ref) via

```julia
# Optimiser
optimiser = AugmentedLagrangian(pcfs,stencil,vel_ext,φh;γ,γ_reinit,verbose=true,constraint_names=[:Vol])
```

As optimisers inheriting from [`LevelSetTopOpt.Optimiser`](@ref) implement Julia's iterator functionality, we can solve the optimisation problem to convergence by iterating over the optimiser:

```julia
# Solve
for (it,uh,φh) in optimiser end
```

This allows the user to inject code between iterations. For example, we can write VTK files for visualisation and save the history using the following:

```julia
# Solve
for (it,uh,φh) in optimiser
  data = ["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi)|"=>(norm ∘ ∇(φh)),"uh"=>uh]
  iszero((it-1) % iter_mod) && writevtk(Ω,path*"_$it",cellfields=data)
  write_history(path*"/history.txt",get_history(optimiser))
end
```

Depending on the use of `iszero((it-1) % iter_mod)`, the VTK file for the final structure
may need to be saved using

```julia
# Final structure
it = get_history(optimiser).niter; uh = get_state(pcfs)
writevtk(Ω,path*"_$it",cellfields=["phi"=>φh,
  "H(phi)"=>(H ∘ φh),"|nabla(phi)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
```

### The full script

```@raw html
<details><summary>Combining the above gives (click me!)</summary>
```

```julia
using Gridap, LevelSetTopOpt

# FE parameters
order = 1                                       # Finite element order
xmax=ymax=1.0                                   # Domain size
el_size = (200,200)                             # Mesh partition size
prop_Γ_N = 0.4                                  # Γ_N size parameter
prop_Γ_D = 0.2                                  # Γ_D size parameter
f_Γ_N(x) = (x[1] ≈ xmax &&                      # Γ_N indicator function
  ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= ymax/2+ymax*prop_Γ_N/4 + eps())
f_Γ_D(x) = (x[1] ≈ 0.0 &&                       # Γ_D indicator function
  (x[2] <= ymax*prop_Γ_D + eps() || x[2] >= ymax-ymax*prop_Γ_D - eps()))
# FD parameters
γ = 0.1                                         # HJ equation time step coefficient
γ_reinit = 0.5                                  # Reinit. equation time step coefficient
max_steps = floor(Int,minimum(el_size)/10)      # Max steps for advection
tol = 1/(10order^2)*prod(inv,minimum(el_size))  # Advection tolerance
# Problem parameters
κ = 1                                           # Diffusivity
g = 1                                           # Heat flow in
vf = 0.4                                        # Volume fraction constraint
lsf_func = initial_lsf(4,0.2)                   # Initial level set function
iter_mod = 10                                   # Output VTK files every 10th iteration
path = "./results/min_thermal_compliance_tut/"  # Output path
mkpath(path)                                    # Create path
# Model
model = CartesianDiscreteModel((0,xmax,0,ymax),el_size);
update_labels!(1,model,f_Γ_D,"Gamma_D")
update_labels!(2,model,f_Γ_N,"Gamma_N")
# Triangulation and measures
Ω = Triangulation(model)
Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
dΩ = Measure(Ω,2*order)
dΓ_N = Measure(Γ_N,2*order)
# Spaces
reffe = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"])
U = TrialFESpace(V,0.0)
V_φ = TestFESpace(model,reffe)
V_reg = TestFESpace(model,reffe;dirichlet_tags=["Gamma_N"])
U_reg = TrialFESpace(V_reg,0)
# Level set and interpolator
φh = interpolate(lsf_func,V_φ)
interp = SmoothErsatzMaterialInterpolation(η = 2*maximum(get_el_Δ(model)))
I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ
# Weak formulation
a(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*κ*∇(u)⋅∇(v))dΩ
l(v,φ,dΩ,dΓ_N) = ∫(g*v)dΓ_N
state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh,dΩ,dΓ_N)
# Objective and constraints
J(u,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*κ*∇(u)⋅∇(u))dΩ
dJ(q,u,φ,dΩ,dΓ_N) = ∫(-κ*∇(u)⋅∇(u)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
vol_D = sum(∫(1)dΩ)
C(u,φ,dΩ,dΓ_N) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
dC(q,u,φ,dΩ,dΓ_N) = ∫(1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
pcfs = PDEConstrainedFunctionals(J,[C],state_map,analytic_dJ=dJ,analytic_dC=[dC])
# Velocity extension
α = 4*maximum(get_el_Δ(model))
a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)
# Finite difference scheme
scheme = FirstOrderStencil(2,Float64)
stencil = AdvectionStencil(scheme,model,V_φ,tol,max_steps)
# Optimiser
optimiser = AugmentedLagrangian(pcfs,stencil,vel_ext,φh;γ,γ_reinit,verbose=true,constraint_names=[:Vol])
# Solve
for (it,uh,φh) in optimiser
  data = ["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi)|"=>(norm ∘ ∇(φh)),"uh"=>uh]
  iszero((it-1) % iter_mod) && writevtk(Ω,path*"struc_$it",cellfields=data)
  write_history(path*"/history.txt",get_history(optimiser))
end
# Final structure
it = get_history(optimiser).niter; uh = get_state(pcfs)
writevtk(Ω,path*"_$it",cellfields=["phi"=>φh,
  "H(phi)"=>(H ∘ φh),"|nabla(phi)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
```

```@raw html
</details>
```

Running this problem until convergence gives 
```
Iteration: ... | L=...
```
with ``\Omega`` given by

...



## Extensions

In the following we outline several extensions to the avoid optimisation problem. These can be considered as destict changes or implemented together. A script containing all extensions can be found under `/scripts/MPI/3d_nonlinear_thermal_compliance_ALM.jl`.

### 3D
The first, and most straightforward in terms of programatic changes is extending the problem to 3D.

### PETSc
To utilise PETSc, we rely on the GridapPETSc satalite package. This provides the neccessary structures to efficently interface with the linear and nonlinear solvers provided by the PETSc library.

### Nonlinear diffusion

Our final extension considers a nonlinear diffusion problem:

```math
\begin{aligned}
-\nabla(\kappa(u)\nabla u) &= 0~\text{in }\Omega,\\
\kappa(u)\nabla u\cdot\boldsymbol{n} &= g~\text{on }\Gamma_N,\\
\kappa(u)\nabla u\cdot\boldsymbol{n} &= 0~\text{on }\partial\Omega\setminus\Gamma_N,\\
u &= 0~\text{on }\Gamma_D.
\end{aligned}
```

where ``\kappa(u)=\kappa_0\exp{\xi u}``. The weak formulation for this problem with relaxation over the computational domain is: *Find* ``u\in H^1_{\Gamma_D}(D)`` *such that* ``R(u,v;\varphi)=0`` for all ``v\in H^1_{\Gamma_D}(D)`` where

```math
R(u,v;\varphi) = \int_{D}I(\varphi)\kappa(u)\boldsymbol{\nabla}u\cdot\boldsymbol{\nabla}v~\mathrm{d}\boldsymbol{x} - \int_{\Gamma_N}gv~\mathrm{d}\boldsymbol{x}.
```

To handle a nonlinear finite element problem we first replace `a` and `l` by

```julia
κ(u) = κ0*(exp(ξ*u))
R(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*(κ ∘ u)*∇(u)⋅∇(v))dΩ - ∫(g*v)dΓ_N
```

In addition we replace the `AffineFEStateMap` with a [`NonlinearFEStateMap`](@ref). This enables automatic differentiation when the forward problem is nonlinear.

```julia
lin_solver = PETScLinearSolver()
nl_solver = NewtonSolver(lin_solver;maxiter=50,rtol=10^-8,verbose=i_am_main(ranks))
state_map = NonlinearFEStateMap(
    res,U,V,V_φ,U_reg,φh,dΩ,dΓ_N;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
    nls = nl_solver, adjoint_ls = lin_solver
)
```
The `state_map` above implements a standard `NewtonSolver` from GridapSolvers while utilising the `PETScLinearSolver` for intermediate linear solves involving the Jacobian.

In addition, the objective functional for this problem is rewritten as ...

### Serial to MPI

## References
> 1. *Z. Guo, X. Cheng, and Z. Xia. Least dissipation principle of heat transport potential capacity and its application in heat conduction  optimization. Chinese Science Bulletin, 48(4):406–410, Feb 2003. ISSN 1861-9541. doi: 10.1007/BF03183239.*
> 
> 2. *C. Zhuang, Z. Xiong, and H. Ding. A level set method for topology optimization of heat conduction problem under multiple load cases. Computer Methods in Applied Mechanics and Engineering, 196(4–6):1074–1084, Jan 2007. ISSN 00457825. doi: 10.1016/j.cma.2006.08.005.*
>
> 3. *Allaire G, Jouve F, Toader AM (2004) Structural optimization using sensitivity analysis and a level-set method. Journal of Computational Physics 194(1):363–393. doi: 10.1016/j.jcp.2003.09.032*
> 
> 4. *Allaire G, Dapogny C, Jouve F (2021) Shape and topology optimization, vol 22, Elsevier, p 1–132. doi: 10.1016/bs.hna.2020.10.004*
>
> 5. *Osher S, Fedkiw R (2006) Level Set Methods and Dynamic Implicit Surfaces, 1st edn. Applied Mathematical Sciences, Springer Science & Business Media. doi: 10.1007/b98879*
>
> 6. *Nocedal J, Wright SJ (2006) Numerical optimization, 2nd edn. Springer series in operations research, Springer, New York. doi: 10.1007/978-0-387-40065-5*