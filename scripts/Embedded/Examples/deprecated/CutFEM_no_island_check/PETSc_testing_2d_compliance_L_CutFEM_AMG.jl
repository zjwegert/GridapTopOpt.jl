using Gridap,GridapTopOpt, GridapSolvers
using Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters

using GridapTopOpt: StateParamMap

using GridapPETSc, SparseMatricesCSR

function main(n;order=1,γg=0.1)
  _model = CartesianDiscreteModel((0,1,0,1),(n,n))
  base_model = UnstructuredDiscreteModel(_model)
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = ref_model.model
  f_Γ_D(x) = x[2] ≈ 1.0
  f_Γ_N(x) = (x[1] ≈ 1 && 0.2 - eps() <= x[2] <= 0.3 + eps())
  update_labels!(1,model,f_Γ_D,"Gamma_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")

  f(x) = ~(0.5 + eps() < x[1] < 1 - eps() && 0.5 + eps() < x[2] < 1 - eps());
  mask = GridapTopOpt.mark_nodes(f,model)
  mask_in = findall(isone,mask)
  topo = get_grid_topology(model)
  cell_to_nodes = Gridap.Geometry.get_faces(topo,2,0);
  cell_mask = findall(x -> all(in.(x, Ref(mask_in))), cell_to_nodes)
  model = UnstructuredDiscreteModel(DiscreteModelPortion(model,cell_mask))

  el_Δ = get_el_Δ(_model)
  h = maximum(el_Δ)

  ## Triangulations and measures
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΓ_N = Measure(Γ_N,2*order)

  ## Levet-set function space and derivative regularisation space
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)

  ## Levet-set function
  φh = interpolate(x->-cos(8π*x[1])*cos(8π*x[2])-0.2,V_φ)
  cutgeo = cut(model,DiscreteGeometry(φh,model))
  strategy = AggregateCutCellsByThreshold(1)
  aggregates = aggregate(strategy,cutgeo)
  Ωact = Triangulation(cutgeo,ACTIVE)
  Ωin = Triangulation(cutgeo,PHYSICAL)
  dΩin = Measure(Ωin,2*order)

  Γg = GhostSkeleton(cutgeo)
  dΓg = Measure(Γg,2*order)
  n_Γg = get_normal_vector(Γg)

  Γ_D = BoundaryTriangulation(model,tags="Gamma_D")
  Λ_D = SkeletonTriangulation(Γ_D)
  nΛ_D = get_normal_vector(Λ_D)
  dΛ_D = Measure(Λ_D,2*order)

  ## Weak form
  function lame_parameters(E,ν)
    λ = (E*ν)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    (λ, μ)
  end

  E = 1.0
  ν = 0.3
  λ, μ = lame_parameters(E,ν)
  σ(ε) = λ*tr(ε)*one(ε) + 2*μ*ε

  g = VectorValue(0,-1)
  a(u,v,φ) = ∫(ε(v) ⊙ (σ ∘ ε(u)))dΩin +
    ∫((γg*h)*jump(nΛ_D⋅∇(v)) ⋅ jump(nΛ_D⋅∇(u)))dΛ_D +
    ∫((γg*h^3)*jump(n_Γg⋅∇(v)) ⋅ jump(n_Γg⋅∇(u)))dΓg
  l(v,φ) = ∫(v⋅g)dΓ_N

  Vstd = TestFESpace(Ωact,ReferenceFE(lagrangian,VectorValue{2,Float64},order);dirichlet_tags=["Gamma_D"])
  V = AgFEMSpace(Vstd,aggregates)
  U = TrialFESpace(V,VectorValue(0.0,0.0))

  ls = ElasticitySolver(V)
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  assem = SparseMatrixAssembler(Tm,Tv,U,V)

  op = AffineFEOperator((u,v)->a(u,v,φh),v->l(v,φh),U,V,assem)
  uh = solve(ls,op)
end

options = "-ksp_converged_reason";
GridapPETSc.with(args=split(options)) do
  main(50)
end;