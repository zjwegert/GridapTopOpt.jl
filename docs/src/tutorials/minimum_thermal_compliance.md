# Minimum thermal compliance

The goal of this tutorial is to learn
- How to formulate a topology optimisation problem
- How to describe the problem over a fixed computational domain ``D`` via the level-set method.
- How to setup and solve the problem in LevelSetTopOpt

We consider the following extensions at the end of the tutorial:
- How to extend the problem to 3D and utilise PETSc solvers
- How to solver problems with nonlinear state equations
- How to run in MPI mode

## Thermal conductivity

The homogeneous steady-state heat equation (equivalently Laplace's equation) is perhaps one of the most well-understood partial differential equations and usually the first introduced to an undergraduate student. For this reason, we will use it to describe the heat transfer through a solid and how one comes to the notion of optimising the shape of that solid.

Consider the geometric conditions outlined in the Figure 1 and suppose that we prescribe the following conditions:
- *Heat source*: unitary normal heat flow across ``\Gamma_{N}``.
- *Insulating*: zero normal heat flow across $\partial\Omega\setminus\Gamma_N$,
- *Heat sink*: zero heat on ``\Gamma_D``.

| ![](2d_min_thermal_comp_setup.png) |
|:--:|
|Figure 1: ...|

Physically we can imagine this as describing the transfer of heat through the domain ``\Omega`` from the sources to the sinks. From a mathematical perspective, we can write down the state equations describing this as

```math
\begin{aligned}
-\nabla(\kappa\nabla u) &= 0~\text{in }\Omega,\\
\kappa\nabla u\cdot\boldsymbol{n} &= g~\text{on }\Gamma_N,\\
\kappa\nabla u\cdot\boldsymbol{n} &= 0~\text{on }\partial\Omega\setminus\Gamma_N,\\
u &= 0~\text{on }\Gamma_D.
\end{aligned}
```

where ``\kappa`` is the diffusivity through ``\Omega`` and ``\boldsymbol{n}`` is the unit normal on the boundary.

## Optimisation problem

```math
\begin{aligned}
\min_{\Omega\in\mathcal{U}}&~J(\Omega)=\int_{\Omega}\kappa\lvert\boldsymbol{\nabla}u\rvert^2~\mathrm{d}\boldsymbol{x}\\
\text{s.t. }&~\operatorname{Vol}(\Omega)=V_f,\\
&\left\{
\begin{aligned}
&\textit{Find }u\in H^1_{\Gamma_D}(\Omega)\\
&\int_{\Omega}\kappa\boldsymbol{\nabla}(u)\cdot\boldsymbol{\nabla}(v)~\mathrm{d}\boldsymbol{x} = \int_{\Gamma_N}v~\mathrm{d}s,~\forall v\in H^1_{\Gamma_D}(\Omega)
\end{aligned}
\right.
\end{aligned}
```
where ``\operatorname{Vol}(\Omega)=\int_\Omega1~\mathrm{d}\boldsymbol{x}``.

Figure, discussion.

As per ... we define a shape derivative as ...

The shape derivatives of ``J`` and ``\operatorname{Vol}`` are then

```math
J'(\Omega)(\boldsymbol{\theta}) = -\int_{\Gamma}\kappa\boldsymbol{\nabla}(u)\cdot\boldsymbol{\nabla}(u)~\boldsymbol{\theta}\cdot\boldsymbol{n}~\mathrm{d}s
```
and
```math
\operatorname{Vol}'(\Omega)(\boldsymbol{\theta}) = \int_{\Gamma}\boldsymbol{\theta}\cdot\boldsymbol{n}~\mathrm{d}s
```
where ``\Gamma = \partial\Omega\setminus(\Gamma_D\cup\Gamma_N)``. These follow from ...

## The level-set method 

Suppose that we attribute a level set function ``\varphi:D\rightarrow\mathbb{R}`` to our domain ``\Omega\subset D`` with ``\bar{\Omega}=\lbrace \boldsymbol{x}:\varphi(\boldsymbol{x})\leq0\rbrace`` and ``\Omega^\complement=\lbrace \boldsymbol{x}:\varphi(\boldsymbol{x})>0\rbrace``. We can then define a smooth characteristic function ``I:\mathbb{R}\rightarrow[\epsilon,1]`` as ``I(\varphi)=(1-H(\varphi))+\epsilon H(\varphi)`` where ``H`` is a smoothed Heaviside function with smoothing radius ``\eta``, and ``\epsilon\ll1`` allows for an ersatz material approximation. Of course, ``\epsilon`` can be taken as zero depending on the computational regime. 

Over the fixed computational domain we may relax integrals to be over all of ``D`` via ``\mathrm{d}\boldsymbol{x}\approx H(\varphi)~\mathrm{d}\boldsymbol{x}`` and ``\mathrm{d}s \approx H'(\varphi)\lvert\nabla\varphi\rvert~\mathrm{d}\boldsymbol{x}``. The above optimisation problem then rewrites as

```math
\begin{aligned}
\min_{\Omega\in\mathcal{U}}&~J(\Omega)=\int_{D}I(\varphi)\kappa\lvert\boldsymbol{\nabla}u\rvert^2~\mathrm{d}\boldsymbol{x}\\
\text{s.t. }&~C(\Omega)=0,\\
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
C(\Omega)=\int_D (\rho(\varphi) - V_f)/\operatorname{Vol}(D)~\mathrm{d}\boldsymbol{x}
``` 

where ``\rho(\varphi)=(1-H(\varphi))`` is the smoothed volume density function.

!!! note
    In LevelSetTopOpt we assume constraints are of the integral form above.

Relaxing the the shape derivatives from the previous section ...

## Computational method

This problem can be solved using the methodologies available in LevelSetTopOpt. 
For the purpose of this tutorial we break the computational formulation down
into chunks. 

### Parameters
The following are user defined parameters for the problem. 
These parameters will be discussed over the course of this tutorial. 

```
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
γ = 0.1                                         # HJ equation time step coefficent
γ_reinit = 0.5                                  # Reinit. equation time step coefficent
max_steps = floor(Int,minimum(el_size)/10)      # Max steps for advection
tol = 1/(10order^2)*prod(inv,minimum(el_size))  # Advection tolerance

# Interpolation & extension-regularisation parameters
η = 2*maximum(el_size)                          # Interpolation band radius
α = 4*maximum(el_size)                          # Smoothing length scale

# Problem parameters
κ = 1                                           # Diffusivity
g = 1                                           # Heat flow in
vf = 0.4                                        # Volume fraction constraint
lsf_func = initial_lsf(4,0.2)                   # Initial level set function
```

### Finite element setup
We first create a Cartesian mesh over ``[0,x_{\max}]\times[0,y_{\max}]`` with partition size `el_size` by creating an object `CartesianDiscreteModel`. In addition, we label the boundaries ``\Gamma_D`` and ``\Gamma_N`` using the `update_labels!` function.
```
model = CartesianDiscreteModel((0,xmax,0,ymax),el_size);
update_labels!(1,model,f_Γ_D,"Gamma_D")
update_labels!(2,model,f_Γ_N,"Gamma_N")
```
The first argument of `update_labels` indicates the label number associated to the region as indicated by the functions `f_Γ_D` and `f_Γ_N`. These functions should take a vector `x` and return `true` or `false` depending on whether a point is present in this region.

Once the model is defined we create an integration mesh and measure for both ``\Omega`` and ``\Gamma_N``. These are built using
```
Ω = Triangulation(model)
Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
dΩ = Measure(Ω,2*order)
dΓ_N = Measure(Γ_N,2*order)
```
where `2*order` indicates the quadrature degree for numerical integration.

The final stage of the finite element setup is the approximation of the finite element spaces. This is given as follows: 
```
reffe = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"])
U = TrialFESpace(V,0.0)
V_φ = TestFESpace(model,reffe)
V_reg = TestFESpace(model,reffe;dirichlet_tags=["Gamma_N","Gamma_D"])
U_reg = TrialFESpace(V_reg,0)
```

In the above, we first define a scalar-valued Lagrangian reference element. This is then used to define the test space `V` and trial space `U` corresponding to ``H^1_{\Gamma_{D}}(\Omega)``. We then construct an FE space `V_φ` over which the level set function is defined, along with an FE test space `V_reg` and trial space `U_reg` over which derivatives are defined. We require that `V_reg` and `U_reg` have zero Dirichlet boundary conditions over regions where the extended shape sensitivity is zero.  

### Initial level set function and interpolant

```
φh = interpolate(lsf_func,V_φ)
```

```
interp = SmoothErsatzMaterialInterpolation(η)
I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ
```

### Weak formulation and the state map
The weak formulation for the problem above can be written as
```
a(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*κ*∇(u)⋅∇(v))dΩ
l(v,φ,dΩ,dΓ_N) = ∫(g*v)dΓ_N
```
where the `∘` operator composes the interpolator `I` with the level set function `φ`.

!!! warning
    The measures must be included as arguments at the end of all functionals.
    This ensures compatibility with Gridap's automatic differentiation.

At this point we can build an `AffineFEStateMap`. This structure is designed to
1) Enable the forward problem that solves an `AffineFEOperator`; and
2) Encode the implicit dependence of the solution `u` on the level set function `φ` to enable the differentiation of `u`.

```
state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh,dΩ,dΓ_N)
```

### Optimisation functionals

The objective functional ``J`` and it's shape derivative is given by

```
J(u,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*κ*∇(u)⋅∇(u))dΩ
dJ(q,u,φ,dΩ,dΓ_N) = ∫(-κ*∇(u)⋅∇(u)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
```

while the constraint on the volume and its derivative is

```
vol_D = sum(∫(1)dΩ)
C(u,φ,dΩ,dΓ_N) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
dC(q,u,φ,dΩ,dΓ_N) = ∫(1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
```

We can now create an object `PDEConstrainedFunctionals` that handles the objective and constrations, and their analytic or automatic differentiation.
```
pcfs = PDEConstrainedFunctionals(J,[C],state_map,analytic_dJ=dJ,analytic_dC=[dC])
```
In this case, the analytic shape derivatives are passed as optional arguments. When these are not given, automatic differentiation is used.

### Velocity extension-regularisation method

The Hilbertian extension-regularisation CITATION method involves solving an 
identification problem over a Hilbert space ``H`` on ``D`` with 
inner product ``\langle\cdot,\cdot\rangle_H``: 
*Find* ``g_\Omega\in H`` *such that* ``\langle g_\Omega,w\rangle_H
=-J^{\prime}(\Omega)(w\boldsymbol{n})~
\forall w\in H.``

This provides two benefits: 
 1) It naturally extends the shape sensitivity from ``\partial\Omega`` 
    onto the bounding domain ``D``; and
 2) ensures a descent direction for ``J(\Omega)`` with additional regularity 
    (i.e., ``H`` as opposed to ``L^2(\partial\Omega)``).

For our problem above we take the inner product

```math
\langle p,q\rangle_H=\int_{D}\alpha^2\nabla(p)\nabla(q)+pq~\mathrm{d}\boldsymbol{x},
```

or equivilantly in our script:
```
a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ
```

We then build an object `VelocityExtension`. This object provides a method `project!` that applyies the Hilbertian velocity-extension method to a given shape derivative. 

```
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)
```

### Advection and reinitialisation

To advect the level set function, we solve the Hamilton-Jacobi evolution equation CITATION. This is given by

```math
\frac{\partial\phi}{\partial t} + V(\boldsymbol{x})\lVert\boldsymbol{\nabla}\phi\rVert = 0,
```

with ``\phi(0,\boldsymbol{x})=\phi_0(\boldsymbol{x})`` and ``\boldsymbol{x}\in D,~t\in(0,T)``.

After advection of the interface, we solve the reinitialisation equation to find an equivilant signed distance function for the given level set function. This is given by

```math
\frac{\partial\phi}{\partial t} + \mathrm{sign}(\phi_0)(\lVert\boldsymbol{\nabla}\phi\rVert-1) = 0,
```

with ``\phi(0,\boldsymbol{x})=\phi_0(\boldsymbol{x})`` and ``\boldsymbol{x}\in D,~t\in(0,T)``.

Both of these equations can be solved numerically on a Cartesian mesh using a first order Godunov upwind difference scheme based on Osher and Fedkiw 
([link](https://doi.org/10.1007/b98879)). We first build an object `FirstOrderStencil` that represents a finite difference stencil for a single step of the Hamilton-Jacobi evolution equation and reinitialisation equation.

```
scheme = FirstOrderStencil(2,Float64)
stencil = AdvectionStencil(scheme,model,V_φ,tol,max_steps)
```

### Optimiser, visualisation and IO

```
optimiser = AugmentedLagrangian(pcfs,stencil,vel_ext,φh;
    γ,γ_reinit,verbose=true,constraint_names=[:Vol])
for (it,uh,φh) in optimiser 
    write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh"=>uh])
    write_history(path*"/history.txt",get_history(optimiser))
end
```

We output a VTK file for the final structure via 

```
it = get_history(optimiser).niter; uh = get_state(pcfs)
write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh"=>uh];iter_mod=1)
### ARE WE MISSING A write_history CALL HERE??
```

## Extensions

### 3D with PETSc

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
R(u,v;\varphi) = \int_{D}I(\varphi)\kappa(u)\boldsymbol{\nabla}(u)\cdot\boldsymbol{\nabla}(v)~\mathrm{d}\boldsymbol{x} - \int_{\Gamma_N}v~\mathrm{d}\boldsymbol{x}.
```

To handle a nonlinear problem we replace `a`, `l`, and `state_map` in our script with

```
κ(u) = κ0*(exp(ξ*u))
R(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*(κ ∘ u)*∇(u)⋅∇(v))dΩ - ∫(v)dΓ_N

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

In addition, the objective functional for this problem rewrites as ...

### Serial to MPI