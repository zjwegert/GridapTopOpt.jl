using Gridap
using Gridap.CellData

function lazy_collect(a)
    c = array_cache(a)
    for i in eachindex(a)
        getindex!(c,a,i)
    end
end

model = CartesianDiscreteModel((0,1,0,1),(2,2))

Ω_space = Triangulation(model,[1,2])
#Ω_space = Triangulation(model,[3,4])

reffe = ReferenceFE(lagrangian,Float64,1)
V = FESpace(Ω_space,reffe)
assem = SparseMatrixAssembler(V,V)

Ω = Triangulation(model)
dΩ = Measure(Ω,2)
j(u) = ∫(1.0)dΩ

uh = zero(V)
c = get_contribution(gradient(j,uh),Ω)

collect(c) # Fails
lazy_collect(c)

assemble_vector(assem,collect_cell_vector(V,gradient(j,uh)))
