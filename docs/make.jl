using Documenter
using LevelSetTopOpt

makedocs(
    sitename = "LevelSetTopOpt.jl",
    format = Documenter.HTML(prettyurls = false),
    modules = [LevelSetTopOpt],
    pages = [
      "Home" => "index.md",
      "Usage" => [
        "usage/getting-started.md",
        "usage/ad.md",
        "usage/petsc.md",
        "usage/mpi-mode.md",
      ],
      "Tutorials" => [
        "tutorials/minimum_thermal_compliance.md"
      ],
      "Reference" => [
        "reference/optimisers.md",
        "reference/chainrules.md",
        "reference/advection.md",
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
#=deploydocs(
    repo = "<repository url>"
)=#
