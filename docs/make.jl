## Local generate
# cd("./docs/")
# using Pkg; Pkg.activate(".")

using Documenter
using GridapTopOpt

makedocs(
    sitename = "GridapTopOpt.jl",
    format = Documenter.HTML(
      prettyurls = false,
    ),
    modules = [GridapTopOpt],
    pages = [
      "Home" => "index.md",
      "Getting Started" => "getting-started.md",
      "Reference" => [
        "reference/optimisers.md",
        "reference/chainrules.md",
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
#=deploydocs(
    repo = "<repository url>"
)=#