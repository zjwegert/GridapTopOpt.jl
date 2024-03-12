# Shape derivative versus Gâteaux derivative in ``\varphi``

The shape derivative of ``J^\prime(\Omega)`` can be defined as the Fréchet derivative under a mapping ``\tilde\Omega = (\boldsymbol{I}+\boldsymbol{\theta})(\Omega)`` at ``\boldsymbol{0}`` with

```math
J(\tilde{\Omega})=J(\Omega)+J^\prime(\Omega)(\boldsymbol{\theta})+o(\lVert\boldsymbol{\theta}\rVert).
```

Consider some illustrative examples. First suppose that ``J_1(\Omega)=\int_\Omega f(\boldsymbol{x})~\mathrm{d}\boldsymbol{x}``, then

```math
J_1(\tilde{\Omega})=\int_{\tilde{\Omega}}f(\tilde{\boldsymbol{x}})~\mathrm{d}\boldsymbol{x}=\int_{\Omega}f(\boldsymbol{x}+\boldsymbol{\theta})\left\lvert\frac{\partial\boldsymbol{\tilde{x}}}{\partial\boldsymbol{x}}\right\rvert~\mathrm{d}\boldsymbol{x}=...=J(\Omega)+\int_{\partial\Omega}f(\boldsymbol{x})~\boldsymbol{\theta}\cdot\boldsymbol{n}~\mathrm{d}s+...
```

So, for this simple case ``J_1^\prime(\Omega)(\boldsymbol{\theta})=\int_{\partial\Omega}f(\boldsymbol{x})~\boldsymbol{\theta}\cdot\boldsymbol{n}~\mathrm{d}s~~(\star).`` For a surface integral ``J_2(\Omega)=\int_{\partial\Omega} f(\boldsymbol{x})~\mathrm{d}s``, one finds in a similar way ``J_2^\prime(\Omega)(\boldsymbol{\theta})=\int_{\partial\Omega}(f\nabla\cdot\boldsymbol{n}+\nabla f\cdot\boldsymbol{n})~\boldsymbol{\theta}\cdot\boldsymbol{n}~\mathrm{d}s.``

Suppose that we attribute a signed distance function ``\varphi:D\rightarrow\mathbb{R}`` to our domain ``\Omega\subset D`` with ``\bar{\Omega}=\lbrace \boldsymbol{x}:\varphi(\boldsymbol{x})\leq0\rbrace`` and ``\Omega^\complement=\lbrace \boldsymbol{x}:\varphi(\boldsymbol{x})>0\rbrace``. We can then define a smooth characteristic function ``\chi_\epsilon:\mathbb{R}\rightarrow[\epsilon,1]`` as ``\chi_\epsilon(\varphi)=(1-H(\varphi))+\epsilon H(\varphi)`` where ``H`` is a smoothed Heaviside function with smoothing radius ``\eta``, and ``\epsilon\ll1`` allows for an ersatz material approximation. Of course, ``\epsilon`` can be taken as zero depending on the considered integral and/or computational regime. We can now rewrite ``J_1`` over the computational domain ``D`` as ``J_{1\Omega}(\varphi)=\int_D \chi_\epsilon(\varphi)f(\boldsymbol{x})~\mathrm{d}\boldsymbol{x}``. Considering the directional derivative of ``J_{1\Omega}`` under a variation ``\tilde{\varphi}=\varphi+sv`` gives

```math
J_{1\Omega}^\prime(\varphi)(v)=\frac{\mathrm{d}}{\mathrm{d}s} J_{1\Omega}(\varphi+sv)\rvert_{s=0}=\int_D v\chi_\epsilon^\prime(\varphi)f(\boldsymbol{x})~\mathrm{d}\boldsymbol{x}
```

or

```math
J_{1\Omega}^\prime(\varphi)(v)=\int_D (\epsilon-1)vH^\prime(\varphi)f(\boldsymbol{x})~\mathrm{d}\boldsymbol{x}.
```

The final result of course does not yet match ``(\star)``. Over a fixed computational domain we may relax integrals to be over all of ``D`` via ``\mathrm{d}s = H'(\varphi)\lvert\nabla\varphi\rvert~\mathrm{d}\boldsymbol{x}``. In addition suppose we take ``\boldsymbol{\theta}=-v\boldsymbol{n}``. Then ``(\star)`` can be rewritten as

```math
J_1^\prime(\Omega)(-v\boldsymbol{n})=-\int_{D}vf(\boldsymbol{x})H'(\varphi)\lvert\nabla\varphi\rvert~\mathrm{d}s.
```

As ``\varphi`` is a signed distance function we have ``\lvert\nabla\varphi\rvert=1`` for ``D\setminus\Sigma`` where ``\Sigma`` is the skeleton of ``\Omega`` and ``\Omega^\complement``. Furthermore, ``H'(\varphi)`` provides support only within a band of ``\partial\Omega``. 

!!! tip "Result I"
    Therefore, we have that almost everywhere

    ```math
    {\huge{|}} J_1^\prime(\Omega)(-v\boldsymbol{n}) - J_{1\Omega}^\prime(\varphi)(v){\huge{|}}=O(\epsilon),
    ```

    with equaility as ``\epsilon\rightarrow0``.

This is not the case when we consider applying the same process to a surface integral such as ``J_2(\Omega)``. In this case, we can never recover ``\nabla f`` under a variation of ``\varphi``. For argument sake we can take ``J_{2\Omega}(\varphi)=\int_D f(\boldsymbol{x})H'(\varphi)\lvert\nabla\varphi\rvert~\mathrm{d}\boldsymbol{x}``. Then a variation in ``\varphi`` gives

```math
J_{2\Omega}^\prime(\varphi)(v)=\int_D f(\boldsymbol{x})\left(H^{\prime\prime}(\varphi)\lvert\nabla\varphi\rvert v + H^\prime(\varphi)\frac{\nabla v\cdot\nabla \varphi}{\lvert\nabla\varphi\rvert}\right)~\mathrm{d}\boldsymbol{x}.
```

On the other hand, relaxing the shape derivative of ``J_2`` in the same way as above gives

```math
J_2^\prime(\Omega)(-v\boldsymbol{n})=\int_{D}-\left(f\nabla\cdot\frac{\nabla\varphi}{\lvert\nabla\varphi\rvert}+\nabla f\cdot\frac{\nabla\varphi}{\lvert\nabla\varphi\rvert}\right)vH'(\varphi)\lvert\nabla\varphi\rvert~\mathrm{d}\boldsymbol{x}.
```

## Adding PDE constraints
Consider ``\Omega\subset D`` with ``J(\Omega)=\int_\Omega j(\boldsymbol{u})~\mathrm{d}\boldsymbol{x}+\int_{\Gamma_N} l_1(\boldsymbol{u})~\mathrm{d}s+\int_{\Gamma_R} l_2(\boldsymbol{u})~\mathrm{d}s`` where ``\boldsymbol{u}`` satisfies

```math
\begin{aligned}
-\mathrm{div}(\boldsymbol{C\varepsilon}(\boldsymbol{u})) &= \boldsymbol{f}\text{ on }\Omega, \\
\boldsymbol{u} &= \boldsymbol{0}\text{ on }\Gamma_D,\\
\boldsymbol{C\varepsilon}(\boldsymbol{u})\boldsymbol{n} &= \boldsymbol{0}\text{ on }\Gamma_0,\\
\boldsymbol{C\varepsilon}(\boldsymbol{u})\boldsymbol{n} &= \boldsymbol{g}\text{ on }\Gamma_N,\\
\boldsymbol{C\varepsilon}(\boldsymbol{u})\boldsymbol{n} &= -\boldsymbol{w}(\boldsymbol{u})\text{ on }\Gamma_R.\\
\end{aligned}
```

In the above ``\boldsymbol{\varepsilon}`` is the strain tensor, ``\boldsymbol{C}`` is the stiffness tensor, ``\Gamma_0 = \partial\Omega\setminus(\Gamma_D\cup\Gamma_N\cup\Gamma_R)``, and ``\Gamma_D``, ``\Gamma_N``, ``\Gamma_R`` are required to be fixed.

### Shape derivative
Let us first consider the shape derivative of ``J``. Disregarding embedding inside the computational domain ``D``, the above strong form can be written in weak form as: 

``\quad`` *Find* ``\boldsymbol{u}\in H^1_{\Gamma_D}(\Omega)^d`` *such that* 

```math
\int_{\Omega} \boldsymbol{C\varepsilon}(\boldsymbol{u})\boldsymbol{\varepsilon}(\boldsymbol{v})~\mathrm{d}\boldsymbol{x}+\int_{\Gamma_R}\boldsymbol{w}(\boldsymbol{u})\cdot\boldsymbol{v}~\mathrm{d}s=\int_\Omega \boldsymbol{f}\cdot\boldsymbol{v}~\mathrm{d}\boldsymbol{x}+\int_{\Gamma_N}\boldsymbol{g}\cdot\boldsymbol{v}~\mathrm{d}s,~\forall \boldsymbol{v}\in H^1_{\Gamma_D}(\Omega)^d.
```

Following Céa's formal adjoint method we introduce the following Lagrangian

```math
\begin{aligned}
\mathcal{L}(\Omega,\boldsymbol{v},\boldsymbol{q})=&\int_\Omega j(\boldsymbol{v})~\mathrm{d}\boldsymbol{x}+\int_{\Gamma_N} l_1(\boldsymbol{v})~\mathrm{d}s+\int_{\Gamma_R} l_2(\boldsymbol{v})~\mathrm{d}s+\int_{\Omega} \boldsymbol{C\varepsilon}(\boldsymbol{v})\boldsymbol{\varepsilon}(\boldsymbol{q})~\mathrm{d}\boldsymbol{x}\\
&+\int_{\Gamma_R}\boldsymbol{w}(\boldsymbol{v})\cdot\boldsymbol{q}~\mathrm{d}s-\int_\Omega \boldsymbol{f}\cdot\boldsymbol{q}~\mathrm{d}\boldsymbol{x}-\int_{\Gamma_N}\boldsymbol{g}\cdot\boldsymbol{q}~\mathrm{d}s\\
&-\int_{\Gamma_D}\boldsymbol{q}\cdot\boldsymbol{C\varepsilon}(\boldsymbol{v})\boldsymbol{n}+\boldsymbol{v}\cdot\boldsymbol{C\varepsilon}(\boldsymbol{q})\boldsymbol{n}~\mathrm{d}s\\
\end{aligned}
```

where ``\boldsymbol{v},\boldsymbol{q}\in H^1(\mathbb{R}^d)^d``. Requiring stationarity of the Lagrangian and taking a partial derivative of ``\mathcal{L}`` with respect to ``\boldsymbol{q}`` in the direction ``\boldsymbol{\phi}\in H^1(\mathbb{R}^d)^d`` gives

```math
\begin{aligned}
0=\frac{\partial\mathcal{L}}{\partial\boldsymbol{q}}(\boldsymbol{\phi})&=\int_{\Omega} \boldsymbol{C\varepsilon}(\boldsymbol{v})\boldsymbol{\varepsilon}(\boldsymbol{\phi})~\mathrm{d}\boldsymbol{x}+\int_{\Gamma_R}\boldsymbol{w}(\boldsymbol{v})\cdot\boldsymbol{\phi}~\mathrm{d}s-\int_\Omega \boldsymbol{f}\cdot\boldsymbol{\phi}~\mathrm{d}\boldsymbol{x}-\int_{\Gamma_N}\boldsymbol{g}\cdot\boldsymbol{\phi}~\mathrm{d}s\\
&\quad-\int_{\Gamma_D}\boldsymbol{\phi}\cdot\boldsymbol{C\varepsilon}(\boldsymbol{v})\boldsymbol{n}+\boldsymbol{v}\cdot\boldsymbol{C\varepsilon}(\boldsymbol{\phi})\boldsymbol{n}~\mathrm{d}s\\
&=\int_\Omega\boldsymbol{\phi}\cdot(-\mathrm{div}(\boldsymbol{C\varepsilon}(\boldsymbol{u}))-\boldsymbol{f})~\mathrm{d}\boldsymbol{x}+\int_{\Gamma_R}\boldsymbol{\phi}\cdot(\boldsymbol{C\varepsilon}(\boldsymbol{v})\boldsymbol{n}+\boldsymbol{w}(\boldsymbol{v}))~\mathrm{d}s\\
&\quad+\int_{\Gamma_N}\boldsymbol{\phi}\cdot(\boldsymbol{C\varepsilon}(\boldsymbol{v})\boldsymbol{n}-\boldsymbol{g})~\mathrm{d}s-\int_{\Gamma_D}\boldsymbol{v}\cdot\boldsymbol{C\varepsilon}(\boldsymbol{\phi})\boldsymbol{n}~\mathrm{d}s+\int_{\Gamma_0}\boldsymbol{\phi}\cdot\boldsymbol{C\varepsilon}(\boldsymbol{v})\boldsymbol{n}~\mathrm{d}s
\end{aligned}
```

after applying integration by parts. Under a suitable variations of ``\boldsymbol{\phi}``, the state equation and boundary conditions are generated as required. In other words, ``\boldsymbol{v}`` is given by the solution ``\boldsymbol{u}`` to the equations of state. We can derive the adjoint equation by again requiring stationarity of the Lagrangian and taking a partial derivative of ``\mathcal{L}`` with respect to ``\boldsymbol{v}`` in the direction ``\boldsymbol{\phi}\in H^1(\mathbb{R}^d)^d``. This gives

```math
\begin{aligned}
0=\frac{\partial\mathcal{L}}{\partial\boldsymbol{v}}(\boldsymbol{\phi})&=\int_\Omega \boldsymbol{\phi}\cdot j^\prime(\boldsymbol{v})~\mathrm{d}\boldsymbol{x}+\int_{\Gamma_N} \boldsymbol{\phi}\cdot l_1^\prime(\boldsymbol{v})~\mathrm{d}s+\int_{\Gamma_R} \boldsymbol{\phi}\cdot l_2^\prime(\boldsymbol{v})~\mathrm{d}s+\int_{\Omega} \boldsymbol{C\varepsilon}(\boldsymbol{\phi})\boldsymbol{\varepsilon}(\boldsymbol{q})~\mathrm{d}\boldsymbol{x}\\
&\quad+\int_{\Gamma_R}\boldsymbol{\phi}\cdot w^\prime(\boldsymbol{v})\cdot\boldsymbol{q}~\mathrm{d}s-\int_{\Gamma_D}\boldsymbol{q}\cdot\boldsymbol{C\varepsilon}(\boldsymbol{\phi})\boldsymbol{n}+\boldsymbol{\phi}\cdot\boldsymbol{C\varepsilon}(\boldsymbol{q})\boldsymbol{n}~\mathrm{d}s\\
&=\int_{\Omega} \boldsymbol{\phi}\cdot(-\mathrm{div}(\boldsymbol{C\varepsilon}(\boldsymbol{q}))-j^\prime(\boldsymbol{v}))~\mathrm{d}\boldsymbol{x}+\int_{\Gamma_N} \boldsymbol{\phi}\cdot (\boldsymbol{C\varepsilon}(\boldsymbol{q})\boldsymbol{n}+l_1^\prime(\boldsymbol{v}))~\mathrm{d}s\\
&\quad+\int_{\Gamma_R} \boldsymbol{\phi}\cdot (\boldsymbol{C\varepsilon}(\boldsymbol{q})\boldsymbol{n}+w^\prime(\boldsymbol{v})\cdot\boldsymbol{q}+l_2^\prime(\boldsymbol{v}))~\mathrm{d}s-\int_{\Gamma_D}\boldsymbol{q}\cdot\boldsymbol{C\varepsilon}(\boldsymbol{\phi})\boldsymbol{n}~\mathrm{d}s\\
&\quad+\int_{\Gamma_0}\boldsymbol{\phi}\cdot\boldsymbol{C\varepsilon}(\boldsymbol{q})\boldsymbol{n}~\mathrm{d}s
\end{aligned}
```

where integration by parts has been applied. The adjoint equations are then generated under suitable variations of ``\boldsymbol{\phi}``, while the previous result ``\boldsymbol{v}=\boldsymbol{u}`` implies that we can identify a unique ``\boldsymbol{q}=\boldsymbol{\lambda}`` that satisfies stationaity. In particular,

```math
\begin{aligned}
-\mathrm{div}(\boldsymbol{C\varepsilon}(\boldsymbol{\lambda})) &= j^\prime(\boldsymbol{u})\text{ on }\Omega, \\
\boldsymbol{\lambda} &= \boldsymbol{0}\text{ on }\Gamma_D,\\
\boldsymbol{C\varepsilon}(\boldsymbol{\lambda})\boldsymbol{n} &= \boldsymbol{0} \text{ on }\Gamma_0,\\
\boldsymbol{C\varepsilon}(\boldsymbol{\lambda})\boldsymbol{n} &= -l_1^\prime(\boldsymbol{u})\text{ on }\Gamma_N,\\
\boldsymbol{C\varepsilon}(\boldsymbol{\lambda})\boldsymbol{n} &= -w^\prime(\boldsymbol{u})\cdot\boldsymbol{\lambda}-l_2^\prime(\boldsymbol{u}) \text{ on }\Gamma_R.\\
\end{aligned}
```

The shape derivative of ``J(\Omega)=\mathcal{L}(\Omega,\boldsymbol{u},\boldsymbol{\lambda})`` then follows by application of the chainrule along with the shape derivative results for ``J_1`` and ``J_2`` above:

```math
\begin{aligned}
J^\prime(\Omega)(\boldsymbol{\theta})&=\frac{\partial\mathcal{L}}{\partial\Omega}(\Omega,\boldsymbol{u},\boldsymbol{\lambda})(\boldsymbol{\theta})+\cancel{\frac{\partial\mathcal{L}}{\partial\boldsymbol{v}}(\Omega,\boldsymbol{u},\boldsymbol{\lambda})}(\boldsymbol{u}^\prime(\boldsymbol{\theta}))+\cancel{\frac{\partial\mathcal{L}}{\partial\boldsymbol{q}}(\Omega,\boldsymbol{u},\boldsymbol{\lambda})}(\boldsymbol{\lambda}^\prime(\boldsymbol{\theta}))\\
&=\int_{\Gamma_0} (\boldsymbol{C\varepsilon}(\boldsymbol{u})\boldsymbol{\varepsilon}(\boldsymbol{\lambda})+j(\boldsymbol{u}) -\boldsymbol{f}\cdot\boldsymbol{\lambda})~\boldsymbol{\theta}\cdot\boldsymbol{n}~\mathrm{d}s.
\end{aligned}
```

As required. Note that in the above, we have used that ``\boldsymbol{\theta}\cdot\boldsymbol{n}=0`` on ``\Gamma_N``, ``\Gamma_R``, and ``\Gamma_D``.


### Gâteaux derivative in ``\varphi``
Let us now return to derivatives of ``J`` with respect to ``\varphi`` over the whole computational domain. As previously, suppose that we rewrite ``J`` as 

```math
\hat{\mathcal{J}}(\varphi)=\int_D (1-H(\varphi))j(\boldsymbol{u})~\mathrm{d}\boldsymbol{x}+\int_{\Gamma_N} l_1(\boldsymbol{u})~\mathrm{d}s+\int_{\Gamma_R} l_2(\boldsymbol{u})~\mathrm{d}s
```

where ``\boldsymbol{u}`` satisfies the state equations as previously with relaxation over the whole computational domain ``D`` as

```math
\begin{aligned}
-\mathrm{div}(\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{u})) &= (1-H(\varphi))\boldsymbol{f}\text{ on }D, \\
\boldsymbol{u} &= \boldsymbol{0}\text{ on }\Gamma_D,\\
\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{u})\boldsymbol{n} &= \boldsymbol{0}\text{ on }\Gamma_0,\\
\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{u})\boldsymbol{n} &= \boldsymbol{g}\text{ on }\Gamma_N,\\
\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{u})\boldsymbol{n} &= -\boldsymbol{w}(\boldsymbol{u})\text{ on }\Gamma_R,\\
\end{aligned}
```

and admits a weak form: *Find* ``\boldsymbol{u}\in H^1_{\Gamma_D}(\Omega)^d`` *such that*

```math
\int_D \chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{u})\boldsymbol{\varepsilon}(\boldsymbol{v})~\mathrm{d}\boldsymbol{x}+\int_{\Gamma_R}\boldsymbol{w}(\boldsymbol{u})\cdot\boldsymbol{v}~\mathrm{d}s=\int_D(1-H(\varphi))\boldsymbol{f}\cdot\boldsymbol{v}~\mathrm{d}\boldsymbol{x}+\int_{\Gamma_N}\boldsymbol{g}\cdot\boldsymbol{v}~\mathrm{d}s,~\forall \boldsymbol{v}\in H^1_{\Gamma_D}(\Omega)^d.
```

At this stage it is important to note that this is almost verbatim as the case without relaxation over ``D``. Indeed, we may follow Céa's method as previously with only a minor adjustment to the Lagrangian that relaxes it over ``D``:

```math
\begin{aligned}
\hat{\mathcal{L}}(\varphi,\boldsymbol{v},\boldsymbol{q})=&\int_D (1-H(\varphi))j(\boldsymbol{v})~\mathrm{d}\boldsymbol{x}+\int_{\Gamma_N} l_1(\boldsymbol{v})~\mathrm{d}s+\int_{\Gamma_R} l_2(\boldsymbol{v})~\mathrm{d}s+\int_{D}\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{v})\boldsymbol{\varepsilon}(\boldsymbol{q})~\mathrm{d}\boldsymbol{x}\\
&+\int_{\Gamma_R}\boldsymbol{w}(\boldsymbol{v})\cdot\boldsymbol{q}~\mathrm{d}s-\int_D (1-H(\varphi)) \boldsymbol{f}\cdot\boldsymbol{q}~\mathrm{d}\boldsymbol{x}-\int_{\Gamma_N}\boldsymbol{g}\cdot\boldsymbol{q}~\mathrm{d}s\\
&-\int_{\Gamma_D}\boldsymbol{q}\cdot\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{v})\boldsymbol{n}+\boldsymbol{v}\cdot\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{q})\boldsymbol{n}~\mathrm{d}s.\\
\end{aligned}
```

As a result the partial derivatives of ``\hat{\mathcal{L}}`` in ``\boldsymbol{v}`` and ``\boldsymbol{q}`` are the same as previously up to relaxation over ``D``, i.e.,

```math
\begin{aligned}
0=\frac{\partial\hat{\mathcal{L}}}{\partial\boldsymbol{q}}(\boldsymbol{\phi})&=\int_D\boldsymbol{\phi}\cdot(-\mathrm{div}(\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{u}))-(1-H(\varphi))\boldsymbol{f})~\mathrm{d}\boldsymbol{x}+\int_{\Gamma_R}\boldsymbol{\phi}\cdot(\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{v})\boldsymbol{n}+\boldsymbol{w}(\boldsymbol{v}))~\mathrm{d}s\\
&\quad+\int_{\Gamma_N}\boldsymbol{\phi}\cdot(\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{v})\boldsymbol{n}-\boldsymbol{g})~\mathrm{d}s-\int_{\Gamma_D}\boldsymbol{v}\cdot\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{\phi})\boldsymbol{n}~\mathrm{d}s+\int_{\Gamma_0}\boldsymbol{\phi}\cdot\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{v})\boldsymbol{n}~\mathrm{d}s
\end{aligned}
```

and

```math
\begin{aligned}
0=\frac{\partial\hat{\mathcal{L}}}{\partial\boldsymbol{v}}(\boldsymbol{\phi})&=\int_D \boldsymbol{\phi}\cdot(-\mathrm{div}(\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{q}))-(1-H(\varphi))j^\prime(\boldsymbol{v}))~\mathrm{d}\boldsymbol{x}+\int_{\Gamma_N} \boldsymbol{\phi}\cdot (\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{q})\boldsymbol{n}+l_1^\prime(\boldsymbol{v}))~\mathrm{d}s\\
&\quad+\int_{\Gamma_R} \boldsymbol{\phi}\cdot (\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{q})\boldsymbol{n}+w^\prime(\boldsymbol{v})\cdot\boldsymbol{q}+l_2^\prime(\boldsymbol{v}))~\mathrm{d}s-\int_{\Gamma_D}\boldsymbol{q}\cdot\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{\phi})\boldsymbol{n}~\mathrm{d}s\\
&\quad+\int_{\Gamma_0}\boldsymbol{\phi}\cdot\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{q})\boldsymbol{n}~\mathrm{d}s.
\end{aligned}
```

Therefore, we may identify ``\boldsymbol{v}=\boldsymbol{u}`` and ``\boldsymbol{q}=\boldsymbol{\lambda}`` where the latter satisfies the adjoint equation

```math
\begin{aligned}
-\mathrm{div}(\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{\lambda})) &= (1-H(\varphi))j^\prime(\boldsymbol{u})\text{ on }\Omega, \\
\boldsymbol{\lambda} &= \boldsymbol{0}\text{ on }\Gamma_D,\\
\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{\lambda})\boldsymbol{n} &= \boldsymbol{0} \text{ on }\Gamma_0,\\
\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{\lambda})\boldsymbol{n} &= -l_1^\prime(\boldsymbol{u})\text{ on }\Gamma_N,\\
\chi_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{\lambda})\boldsymbol{n} &= -w^\prime(\boldsymbol{u})\cdot\boldsymbol{\lambda}-l_2^\prime(\boldsymbol{u}) \text{ on }\Gamma_R.\\
\end{aligned}
```

This exactly as previously up to relaxation over ``D``. Finally, The derivative of ``\hat{J}(\varphi)=\hat{\mathcal{L}}(\varphi,\boldsymbol{u},\boldsymbol{\lambda})`` then follows by application of the chainrule as previously:

```math
\begin{aligned}
\hat{J}^\prime(\varphi)(v)&=\frac{\partial\hat{\mathcal{L}}}{\partial\varphi}(\varphi,\boldsymbol{u},\boldsymbol{\lambda})(v)+\cancel{\frac{\partial\hat{\mathcal{L}}}{\partial\boldsymbol{v}}(\varphi,\boldsymbol{u},\boldsymbol{\lambda})}(\boldsymbol{u}^\prime(v))+\cancel{\frac{\partial\hat{\mathcal{L}}}{\partial\boldsymbol{q}}(\varphi,\boldsymbol{u},\boldsymbol{\lambda})}(\boldsymbol{\lambda}^\prime(v))\\
&=\int_D v(\chi^\prime_\epsilon(\varphi)\boldsymbol{C\varepsilon}(\boldsymbol{u})\boldsymbol{\varepsilon}(\boldsymbol{\lambda})+(-H^\prime(\varphi))j(\boldsymbol{u})- (-H^\prime(\varphi))\boldsymbol{f}\cdot\boldsymbol{\lambda})~\mathrm{d}\boldsymbol{x}\\
&=-\int_D v(\boldsymbol{C\varepsilon}(\boldsymbol{u})\boldsymbol{\varepsilon}(\boldsymbol{\lambda})+j(\boldsymbol{u})- \boldsymbol{f}\cdot\boldsymbol{\lambda})H^\prime(\varphi)~\mathrm{d}\boldsymbol{x}+\epsilon\int_D v\boldsymbol{C\varepsilon}(\boldsymbol{u})\boldsymbol{\varepsilon}(\boldsymbol{\lambda})H^\prime(\varphi)~\mathrm{d}\boldsymbol{x}
\end{aligned}
```

where we have used that ``v=0`` on ``\Gamma_D`` as previously. As previously taking ``\boldsymbol{\theta}=v\boldsymbol{n}`` and relaxing the shape derivative of ``J`` over ``D`` with a signed distance function ``\varphi`` yields: 

!!! tip "Result II"
    ```math
    {\huge{|}} J^\prime(\Omega)(v\boldsymbol{n}) - \hat{\mathcal{J}}^\prime(\varphi)(v){\huge{|}}=O(\epsilon).
    ```

As required.

## What is not captured by a Gâteaux derivative at ``\varphi``?
Owing to a fixed computational regime we do not capture a variation of the domain ``\Omega`` in ``\varphi``, so the Gâteaux derivative of certain types of functionals in this context will not match the shape derivative. For example, in the discussion above the shape derivative of ``J_2`` and the Gâteaux derivative of ``J_{2\Omega}`` fail to match even in the relaxed setting. Regardless of this, accurate resolution of ``J_2`` is difficult owing to the appearance of mean curvature. This problem is further exacerbated when a discretisation of the boundary is not available.

In addition, functionals of the signed distance function posed over the whole bounding domain ``D`` admit a special structure under shape differentiation ([Paper](https://doi.org/10.1051/m2an/2019056)). Such cases are not captured by a Gâteaux derivative at ``\varphi`` under relaxation.

!!! note
    In future, we plan to implement CellFEM via GridapEmbedded in LevelSetTopOpt. This will enable Gâteaux derivative of the mapping

    ```math
    \varphi \mapsto \int_{\Omega(\varphi)}f(\varphi)~\mathrm{d}\boldsymbol{x} + \int_{\Omega(\varphi)^\complement}f(\varphi)~\mathrm{d}\boldsymbol{x}.
    ```

    We expect this to rectify the discussion above.