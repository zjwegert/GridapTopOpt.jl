# GridapTopOpt - `Wegert_et_al_2024` branch
This branch contains the scripts and results for the referenced paper below. These are ordered as follows:
- `scripts/` contains `Paper_scripts/` and `Benchmarks/` that are used to generate the results from Section 4 and 5, respectively.
- `results/` contains the benchmark results and job logs for Section 5 in `Benchmarks/`. The results for Section 4 are held in a [separate repository](https://github.com/zjwegert/Wegert_et_al_2024_Results) as the visualisation files (`.vtu`) are large.

**Note**:
- Currently we include the `Manifest.toml` to ensure the correct branch of `GridapSolvers` is included when instantiating the package. This will be removed in future.
- This branch is protected to ensure the code in `src/` matches that of the paper (TODO).

---

GridapTopOpt is computational toolbox for level set-based topology optimisation implemented in Julia and the [Gridap](https://github.com/gridap/Gridap.jl) package ecosystem. See the documentation and following publication for further details:

> Zachary J. Wegert, Jordi Manyer, Connor Mallon, Santiago Badia, and Vivien J. Challis (2024). "GridapTopOpt.jl: A scalable Julia toolbox for level set-based topology optimisation". In preparation.

## Citation
In order to give credit to the `GridapTopOpt` contributors, we ask that you please reference the above paper along with the required citations for [Gridap](https://github.com/gridap/Gridap.jl?tab=readme-ov-file#how-to-cite-gridap).