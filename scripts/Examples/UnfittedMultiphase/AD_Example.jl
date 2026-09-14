using Gridap, Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt

# This script demonstrates how to use automatic shape differentiation for problems
# described by multiple level-set functions.

# See the following paper for further details:
# > Zachary J. Wegert, Martin Berggren, and Vivien J. Challis (2026). "Shape calculus and automatic differentiation
#   for multi-phase level-set topology optimisation with unfitted finite elements". arXiv:...

# Background mesh
base_model = CartesianDiscreteModel((0,1,0,1,0,1),(15,15,15))
ref_model = refine(UnstructuredDiscreteModel(base_model), refinement_method = "barycentric")
model = get_model(ref_model)
# Level-set functions and geometries Dφᵢ
order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
Vφᵢ = MultiFieldFESpace([TestFESpace(model,reffe) for i in 1:3])
φ₁ = x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-0.25
φ₂ = x->sqrt((x[1]-0.75)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-0.1
φ₃ = x->sqrt((0.25-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2))^2 + (x[3]-0.5)^2) - 0.025
φhᵢ = interpolate([φ₁,φ₂,φ₃],Vφᵢ)
Dφ₁ = DiscreteGeometryFromFEFunction(φhᵢ[1],model)
Dφ₂ = DiscreteGeometryFromFEFunction(φhᵢ[2],model)
Dφ₃ = DiscreteGeometryFromFEFunction(φhᵢ[3],model)
# Triangulation of Ω and Γ
Ω_geo = Dφ₁ ∩ !Dφ₂ ∩ !Dφ₃
cutgeo = cut(PolytopalLevelSetCutter(),model,Ω_geo)
Ω = DifferentiableTriangulation(cutgeo,Ω_geo)
Γ = DifferentiableEmbeddedBoundary(cutgeo,Ω_geo,Dφ₃)
dΩ = Measure(Ω,2*order)
dΓ = Measure(Γ,2*order)
# Functional and gradient
J(φhᵢ) = ∫(1)dΩ + ∫(1)dΓ
dJ = gradient(J,φhᵢ)
vec_dJ = assemble_vector(dJ,Vφᵢ)