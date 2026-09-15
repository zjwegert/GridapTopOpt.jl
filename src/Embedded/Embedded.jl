const CUT = 0

# Util
include("EmbeddedCollections.jl")

# Isolated volumes
include("IsolatedVolumes/IsolatedVolumes.jl")
include("IsolatedVolumes/PolytopalCutters.jl")

# Polytopal cutters (DifferentiableTriangulation & DifferentiableEmbeddedBoundary)
include("PolytopalCutters/PolytopalCutters.jl")