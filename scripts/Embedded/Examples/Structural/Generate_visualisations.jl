using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt

using GridapDistributed,PartitionedArrays,GridapPETSc

global ncpus = parse(Int64,ARGS[1])
global result_path = ARGS[2]
global mesh_name = ARGS[3]
global I0 =  parse(Int64,ARGS[4])
global IF =  parse(Int64,ARGS[5])
global Imod = parse(Int64,ARGS[6])

function main(ranks)
  _tmp = split(result_path,"/")
  path = "./results/VISUALISATION_$(_tmp[end])/"
  files_path = path*"data/"
  i_am_main(ranks) && mkpath(files_path)

  vf = 0.3
  D = 3

  # Load mesh
  model = GmshDiscreteModel(ranks,result_path*"/Mesh/$mesh_name")
  model = UnstructuredDiscreteModel(model)
  f_diri(x) =
    (cos(pi/3)<=x[1]<=cos(pi/6) && abs(x[2] - sqrt(1-x[1]^2))<1e-4) ||
    (cos(7pi/6)<=x[1]<=cos(2pi/3) && abs(x[2] - sqrt(1-x[1]^2))<1e-4) ||
    (cos(pi/3)<=x[1]<=cos(pi/6) && abs(x[2] - -sqrt(1-x[1]^2))<1e-4) ||
    (cos(7pi/6)<=x[1]<=cos(2pi/3) && abs(x[2] - -sqrt(1-x[1]^2))<1e-4)
  update_labels!(1,model,f_diri,"Gamma_D_new")

  # Get triangulation and element size
  Ω_bg = Triangulation(model)
  hₕ = get_element_diameter_field(model)

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)
  φh = interpolate(-1,V_φ)
  pload!(result_path*"/data/LSF_$I0",get_free_dof_values(φh))

  # Check LS
  GridapTopOpt.correct_ls!(φh)

  # Setup integration meshes and measures
  order = 1
  degree = 2*(order+1)
  Γ_N = BoundaryTriangulation(model,tags=["Gamma_N",])
  dΓ_N = Measure(Γ_N,degree)
  dΩ_bg = Measure(Ω_bg,degree)
  Ω_data = EmbeddedCollection(model,φh) do cutgeo,cutgeo_facets,_φh
    Ω = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
    Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
    Γg = GhostSkeleton(cutgeo)
    Ω_act = Triangulation(cutgeo,ACTIVE)
    # Isolated volumes
    φ_cell_values = map(get_cell_dof_values,local_views(_φh))
    ψ,_ = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Gamma_D_new",])
    (;
      :Ω_act => Ω_act,
      :Ω     => Ω,
      :dΩ    => Measure(Ω,degree),
      :Γg    => Γg,
      :dΓg   => Measure(Γg,degree),
      :n_Γg  => get_normal_vector(Γg),
      :Γ     => Γ,
      :dΓ    => Measure(Γ,degree),
      :n_Γ   => get_normal_vector(Γ), # Note, need to recompute inside obj/constraints to compute derivs
      :ψ     => ψ
    )
  end

  # Setup spaces
  reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
  function build_spaces(Ω_act)
    V = TestFESpace(Ω_act,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_D_new",])
    U = TrialFESpace(V)
    return U,V
  end

  ### Weak form
  # Material parameters
  function lame_parameters(E,ν)
    λ = (E*ν)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    (λ, μ)
  end
  λs, μs = lame_parameters(1.0,0.3)
  # Stabilization
  α_Gd = 1e-7
  k_d = 1.0
  γ_Gd(h) = α_Gd*(λs + μs)*h^3
  # Terms
  σ(ε) = λs*tr(ε)*one(ε) + 2*μs*ε
  a_s_Ω(d,s) = ε(s) ⊙ (σ ∘ ε(d)) # Elasticity
  j_s_k(d,s) = mean(γ_Gd ∘ hₕ)*(jump(Ω_data.n_Γg ⋅ ∇(s)) ⋅ jump(Ω_data.n_Γg ⋅ ∇(d)))
  v_s_ψ(d,s) = (k_d*Ω_data.ψ)*(d⋅s) # Isolated volume term
  g((x,y,z)) = 100VectorValue(-y,x,0.0)

  a(d,s,φ) = ∫(a_s_Ω(d,s) + v_s_ψ(d,s))Ω_data.dΩ + ∫(j_s_k(d,s))Ω_data.dΓg
  l(s,φ) = ∫(s⋅g)dΓ_N

  ## Optimisation functionals
  vol_D = sum(∫(1)dΩ_bg)
  iso_vol_frac(φ) = ∫(Ω_data.ψ/vol_D)Ω_data.dΩ
  J_comp(d,φ) = ∫(ε(d) ⊙ (σ ∘ ε(d)))Ω_data.dΩ
  Vol(d,φ) = ∫(1/vol_D)Ω_data.dΩ - ∫(vf/vol_D)dΩ_bg
  dVol(q,d,φ) = ∫(-1/vol_D*q/(abs(Ω_data.n_Γ ⋅ ∇(φ))))Ω_data.dΓ

  ## Setup solver and FE operators
  solver = PETScLinearSolver()

  al_keys = [:J,:Vol]
  al_bundles = Dict(:C => [:Vol,])
  history = GridapTopOpt.OptimiserHistory(Float64,al_keys,al_bundles,1000,i_am_main(ranks))

  t = PTimer(ranks);

  its = []
  k = 0

  for it = I0:IF
    if it % Imod != 0
      continue
    end
    push!(its,it); k = k+1
    pload!(result_path*"/data/LSF_$it",get_free_dof_values(φh))
    update_collection!(Ω_data,φh)
    U,V = build_spaces(Ω_data.Ω_act)
    op = AffineFEOperator((u,v)->a(u,v,φh),v->l(v,φh),U,V)
    uh = solve(solver,op)

    _J = sum(J_comp(uh,φh))
    _Vol = sum(Vol(uh,φh))
    push!(history,(_J,_Vol))

    tic!(t;barrier=true)
    write_history(path*"/history.txt",history;ranks)
    writevtk(Ω_bg,files_path*"Omega_act_$it",cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ψ"=>Ω_data.ψ])
    writevtk(Ω_data.Ω,files_path*"Omega_in_$it",cellfields=["uh"=>uh])
    toc!(t,"Write")

    if k>1 && i_am_main(ranks)
      _it = its[k-1]
      run(`tar -czf $files_path/data_$_it.tar.gz $files_path/Omega_in_$_it $files_path/Omega_in_$_it.pvtu $files_path/Omega_act_$_it $files_path/Omega_act_$_it.pvtu`)
      run(`rm -r $files_path/Omega_in_$_it $files_path/Omega_act_$_it`)
      run(`rm -r $files_path/Omega_in_$_it.pvtu $files_path/Omega_act_$_it.pvtu`)
    end
  end
  if i_am_main(ranks)
    _it = its[end]
    run(`tar -czf $files_path/data_$_it.tar.gz $files_path/Omega_in_$_it $files_path/Omega_in_$_it.pvtu $files_path/Omega_act_$_it $files_path/Omega_act_$_it.pvtu`)
    run(`rm -r $files_path/Omega_in_$_it $files_path/Omega_act_$_it`)
    run(`rm -r $files_path/Omega_in_$_it.pvtu $files_path/Omega_act_$_it.pvtu`)
  end
end

with_mpi() do distribute
  ranks = distribute(LinearIndices((ncpus,)))
  petsc_options = "-ksp_converged_reason -ksp_error_if_not_converged true -pc_type lu -pc_factor_mat_solver_type superlu_dist"
  GridapPETSc.with(;args=split(petsc_options)) do
    main(ranks)
  end
end
