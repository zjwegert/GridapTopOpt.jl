using Documenter
using LevelSetTopOpt

makedocs(
    sitename = "LevelSetTopOpt",
    format = Documenter.HTML(prettyurls = false),
    modules = [LevelSetTopOpt]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
