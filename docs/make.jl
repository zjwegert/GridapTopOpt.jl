using Documenter
using LSTO_Distributed

makedocs(
    sitename = "LSTO_Distributed",
    format = Documenter.HTML(),
    modules = [LSTO_Distributed]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
