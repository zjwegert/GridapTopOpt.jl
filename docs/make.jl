# ## Local generate
# cd("./docs/")
# using Pkg; Pkg.activate(".")
# Pkg.develop(path="../")
# ##

using Documenter
using GridapTopOpt

makedocs(
    sitename = "GridapTopOpt.jl",
    format = Documenter.HTML(
      # prettyurls = false, # <- uncomment for live documentation
      collapselevel = 1,
    ),
    warnonly = [:cross_references,:missing_docs],
    checkdocs = :exports,
    modules = [GridapTopOpt],
    pages = [
      "Home" => "index.md",
      "Getting Started" => "getting-started.md",
      "Examples" => [
        "Introductory examples" => "examples/index.md",
        "Topology optimisation on unfitted meshes" => "examples/Unfitted-TO-with-Laplace.md",
        "FSI with CutFEM" => "examples/Fluid-structure_interaction_with_CutFEM.md",
      ],
      "Breaking changes" => "breaking-changes.md",
      "Reference" => [
        "reference/optimisers.md",
        "reference/statemaps.md",
        "reference/levelsetevolution.md",
        "reference/velext.md",
        "reference/io.md",
        "reference/utilities.md",
        "reference/benchmarking.md"
      ],
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/zjwegert/GridapTopOpt.jl.git",
)
