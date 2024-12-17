using Gridap,GridapTopOpt, GridapSolvers
using Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters

using GridapTopOpt: StateParamMap

using GridapDistributed, GridapPETSc, GridapSolvers,
  PartitionedArrays, GridapTopOpt, SparseMatricesCSR

using GridapSolvers: NewtonSolver

function main(mesh_partition,distribute,el_size,path)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  if i_am_main(ranks)
    rm(path,force=true,recursive=true)
    mkpath(path)
  end

  n = maximum(el_size)
  order = 1
  γ = 0.1
  max_steps = floor(Int,order*n/5)
  vf = 0.4
  α_coeff = 4max_steps*γ
  iter_mod = 1

  _model = CartesianDiscreteModel(ranks,mesh_partition,(0,1,0,1),(n,n))
  base_model = UnstructuredDiscreteModel(_model)
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = Adaptivity.get_model(ref_model)
  el_Δ = get_el_Δ(_model)
  h = maximum(el_Δ)
  h_refine = maximum(el_Δ)/2
  f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= 0.2 + eps() || x[2] >= 0.8 - eps()))
  f_Γ_N(x) = (x[1] ≈ 1 && 0.4 - eps() <= x[2] <= 0.6 + eps())
  update_labels!(1,model,f_Γ_D,"Gamma_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")

  ## Triangulations and measures
  Ω = Triangulation(model)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΩ = Measure(Ω,2*order)
  dΓ_N = Measure(Γ_N,2*order)
  vol_D = sum(∫(1)dΩ)

  ## Levet-set function space and derivative regularisation space
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)
  V_φ = TestFESpace(model,reffe_scalar)

  ## Levet-set function
  φh = interpolate(x->-cos(4π*x[1])*cos(4π*x[2])-0.4,V_φ)
  Ωs = EmbeddedCollection(model,φh) do cutgeo,_
    Ωin = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_IN),V_φ)
    Ωout = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
    Γ = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
    Γg = GhostSkeleton(cutgeo)
    Ωact = Triangulation(cutgeo,ACTIVE)
    (;
      :Ωin  => Ωin,
      :dΩin => Measure(Ωin,2*order),
      :Ωout  => Ωout,
      :dΩout => Measure(Ωout,2*order),
      :Γg   => Γg,
      :dΓg  => Measure(Γg,2*order),
      :n_Γg => get_normal_vector(Γg),
      :Γ    => Γ,
      :dΓ   => Measure(Γ,2*order),
      :Ωact => Ωact
    )
  end

  ## Weak form
  ϵ = 1e-3
  a(u,v,φ) = ∫(∇(v)⋅∇(u))Ωs.dΩin + ∫(ϵ*∇(v)⋅∇(u))Ωs.dΩout
  l(v,φ) = ∫(v)dΓ_N

  ## Optimisation functionals
  J(u,φ) = ∫(∇(u)⋅∇(u))Ωs.dΩin
  Vol(u,φ) = ∫(1/vol_D)Ωs.dΩin - ∫(vf/vol_D)dΩ
  dVol(q,u,φ) = ∫(-1/vol_D*q/(norm ∘ (∇(φ))))Ωs.dΓ

  ## Setup solver and FE operators
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  solver = PETScLinearSolver()

  V = TestFESpace(Ω,reffe_scalar;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,0.0)
  state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
    ls = solver,adjoint_ls = solver
  )
  pcfs = PDEConstrainedFunctionals(J,[Vol],state_map;analytic_dC=(dVol,))

  ## Evolution Method
  evo = CutFEMEvolve(V_φ,Ωs,dΩ,h;max_steps,
    ode_ls = solver,
    assembler=SparseMatrixAssembler(Tm,Tv,V_φ,V_φ))
  reinit = StabilisedReinit(V_φ,Ωs,dΩ,h;
    stabilisation_method=ArtificialViscosity(3.0),
    assembler=SparseMatrixAssembler(Tm,Tv,V_φ,V_φ),
    nls = NewtonSolver(LUSolver();maxiter=50,rtol=1.e-14,verbose=i_am_main(ranks)))
  ls_evo = UnfittedFEEvolution(evo,reinit)
  reinit!(ls_evo,φh)

  ## Hilbertian extension-regularisation problems
  α = α_coeff*(h_refine/order)^2
  a_hilb(p,q) =∫(α*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(
    a_hilb, U_reg, V_reg;
    assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls = solver
  )

  ## Optimiser
  converged(m) = GridapTopOpt.default_al_converged(
    m;
    L_tol = 0.01*h_refine,
    C_tol = 0.01
  )
  optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;debug=true,
    γ,verbose=i_am_main(ranks),constraint_names=[:Vol],converged)
  for (it,uh,φh,state) in optimiser
    if iszero(it % iter_mod)
      writevtk(Ω,path*"Omega$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"velh"=>FEFunction(V_φ,state.vel)])
      writevtk(Ωs.Ωin,path*"Omega_in$it",cellfields=["uh"=>uh])
    end
    write_history(path*"/history.txt",optimiser.history;ranks=ranks)
  end
  it = get_history(optimiser).niter; uh = get_state(pcfs)
  writevtk(Ω,path*"Omega$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
  writevtk(Ωs.Ωin,path*"Omega_in$it",cellfields=["uh"=>uh])
end

with_mpi() do distribute
  write_dir="./results/FCM_thermal_compliance_ALM_MPI/"
  mesh_partition = (2,2)
  el_size = (50,50)
  solver_options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true
    -ksp_converged_reason -ksp_rtol 1.0e-12"

  GridapPETSc.with(args=split(solver_options)) do
    main(mesh_partition,distribute,el_size,write_dir)
  end
end