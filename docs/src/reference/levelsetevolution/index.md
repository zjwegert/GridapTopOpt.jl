# LevelSetEvolution
In GridapTopOpt, the level-set function is evolved and reinitialised using implementations of `LevelSetEvolution`. We provide the following implementations of this:
- [Finite differences](finitediff.md)
- [Unfitted finite element methods](unfitted.md)

## Custom `LevelSetEvolution`
To implement a custom level-set evolution method, the following methods can be extended. For example, one could consider Reaction-Diffusion-based evolution of the level set function. This can be solved with a finite element method and so we can implement a new type that inherits from `LevelSetEvolution` independently of the `Stencil` types.

```@docs
GridapTopOpt.LevelSetEvolution
GridapTopOpt.evolve!(::GridapTopOpt.LevelSetEvolution,φ,args...)
GridapTopOpt.reinit!(::GridapTopOpt.LevelSetEvolution,φ,args...)
GridapTopOpt.get_dof_Δ(::GridapTopOpt.LevelSetEvolution)
```