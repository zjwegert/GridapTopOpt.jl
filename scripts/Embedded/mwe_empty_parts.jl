
using Gridap
using Gridap.Geometry, Gridap.FESpaces

using GridapEmbedded
using GridapEmbedded.Interfaces, GridapEmbedded.LevelSetCutters

using GridapDistributed, PartitionedArrays

using GridapTopOpt

_num_cells(t) = num_cells(t)
_num_cells(t::Gridap.Geometry.AppendedTriangulation) = (num_cells(t.a), num_cells(t.b))

D = 3
np = (D==2) ? (2,1) : (2,1,1)
ranks = with_mpi() do distribute 
  distribute([1,2])
end

nc = (D==2) ? (4,2) : (4,2,2)
domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
model = simplexify(CartesianDiscreteModel(ranks,np,domain,nc))

reffe_m = ReferenceFE(lagrangian,Float64,1)
M = FESpace(model,reffe_m)

θ = 0.1
for θ in [0.1,0.4,0.6,0.9]
  println("θ = $θ")
  ls(x) = ifelse(x[1] > θ,1.0,-1.0)
  mh = interpolate(ls,M)

  geo = DiscreteGeometry(mh,model)
  cutgeo = cut(model,geo)

  # Ωin = Triangulation(cutgeo,PHYSICAL)
  Ωin = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),M)
  #local_views(Ωin)
  #display(map(_num_cells,local_views(Ωin)))
  Ωac = Triangulation(cutgeo,ACTIVE)
  #display(map(_num_cells,local_views(Ωac)))

  reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},1)
  reffe_p = ReferenceFE(lagrangian,Float64,0)
  U = FESpace(Ωac,reffe_u)
  P = FESpace(Ωac,reffe_p)
  X = MultiFieldFESpace([U,P])
  #display(map(num_free_dofs,local_views(U)))

  f = zero(VectorValue{D,Float64})
  dΩin = Measure(Ωin,2)
  a(u,v) = ∫(∇(u) ⊙ ∇(v))dΩin
  b((u,p),(v,q)) = ∫(∇(u) ⊙ ∇(v) + p*q)dΩin
  l((v,q)) = ∫(v⋅f)dΩin
  assemble_matrix(a,U,U)
  assemble_matrix(b,X,X)

  uh = zero(X)
  r(x) = b(uh,x) - l(x)
  assemble_vector(gradient(r,uh),X)
  println("done")
end
