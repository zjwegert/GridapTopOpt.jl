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

  vf = 0.025
  D = 3

  # Load gmsh mesh (Currently need to update mesh.geo and these values concurrently)
  L = 4.0;
  H = 1.0;
  x0 = 2;
  l = 1.0;
  w = 0.1;
  a = 0.7;
  b = 0.1;
  cw = 0.1;
  vol_D = L*H

  model = GmshDiscreteModel(ranks,result_path*"/Mesh/$mesh_name")
  model = UnstructuredDiscreteModel(model)

  Ω_act = Triangulation(model)
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

  dΩ_act = Measure(Ω_act,degree)
  Γf_D = BoundaryTriangulation(model,tags="Gamma_f_D")
  dΓf_D = Measure(Γf_D,degree)
  Ω = EmbeddedCollection(model,φh) do cutgeo,cutgeo_facets,_φh
    Ωs = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
    Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
    Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
    Γg = GhostSkeleton(cutgeo)
    Ω_act_s = Triangulation(cutgeo,ACTIVE)
    Ω_act_f = Triangulation(cutgeo,ACTIVE_OUT)
    Γi = SkeletonTriangulation(cutgeo_facets,ACTIVE_OUT)
    # Isolated volumes
    φ_cell_values = map(get_cell_dof_values,local_views(_φh))
    ψ_s,_ = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Gamma_Bottom"])
    _,ψ_f = GridapTopOpt.get_isolated_volumes_mask_polytopal(model,φ_cell_values,["Gamma_f_D"])
    (;
      :Ωs       => Ωs,
      :dΩs      => Measure(Ωs,degree),
      :Ωf       => Ωf,
      :dΩf      => Measure(Ωf,degree),
      :Γg       => Γg,
      :dΓg      => Measure(Γg,degree),
      :n_Γg     => get_normal_vector(Γg),
      :Γ        => Γ,
      :dΓ       => Measure(Γ,degree),
      :n_Γ      => get_normal_vector(Γ), # Note, need to recompute inside obj/constraints to compute derivs
      :Ω_act_s  => Ω_act_s,
      :dΩ_act_s => Measure(Ω_act_s,degree),
      :Ω_act_f  => Ω_act_f,
      :dΩ_act_f => Measure(Ω_act_f,degree),
      :Γi       => Γi,
      :dΓi      => Measure(Γi,degree),
      :n_Γi     => get_normal_vector(Γi),
      :ψ_s      => ψ_s,
      :ψ_f      => ψ_f,
    )
  end

  # Setup spaces
  uin(x) = VectorValue(x[2],0.0,0.0)
  uin_dot_e1(x) = uin(x)⋅VectorValue(1.0,0.0,0.0)

  reffe_u = ReferenceFE(lagrangian,VectorValue{D,Float64},order,space=:P)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1,space=:P)
  reffe_d = ReferenceFE(lagrangian,VectorValue{D,Float64},order)

  function build_spaces(Ω_act_s,Ω_act_f)
    # Test spaces
    V = TestFESpace(Ω_act_f,reffe_u,conformity=:H1,
      dirichlet_tags=["Gamma_f_D","Gamma_Bottom","Gamma_Top",
        "Gamma_Symm","Gamma_Symm_NonDesign","Gamma_Right","Gamma_TopCorners"],
      dirichlet_masks=[(true,true,true),(true,true,true),(false,true,false),
        (false,false,true),(false,false,true),(false,false,true),(false,true,true)])
    Q = TestFESpace(Ω_act_f,reffe_p,conformity=:L2)
    T = TestFESpace(Ω_act_s,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_Bottom","Gamma_Symm","Gamma_Symm_NonDesign"],
      dirichlet_masks=[(true,true,true),(false,false,true),(false,false,true)])

    # Trial spaces
    U = TrialFESpace(V,[uin,[VectorValue(0.0,0.0,0.0) for _ = 1:6]...])
    P = TrialFESpace(Q)
    R = TrialFESpace(T)

    # Multifield spaces
    UP = MultiFieldFESpace([U,P])
    VQ = MultiFieldFESpace([V,Q])
    return (UP,VQ),(R,T)
  end

  ### Weak form
  ## Fluid
  # Properties
  Re = 60 # Reynolds number
  ρ = 1.0 # Density
  cl = a # Characteristic length
  u0_max = sum(∫(uin_dot_e1)dΓf_D)/sum(∫(1)dΓf_D)
  μ = ρ*cl*u0_max/Re # Viscosity
  ν = μ/ρ # Kinematic viscosity

  # Stabilization parameters
  α_Nu = 100
  α_u  = 0.1
  α_p  = 0.25

  γ_Nu(h) = α_Nu*μ/h
  γ_u(h) = α_u*μ*h
  γ_p(h) = α_p*h/μ
  k_p    = 1.0 # (Villanueva and Maute, 2017)
  γ_Nu_h = γ_Nu ∘ hₕ
  γ_u_h = mean(γ_u ∘ hₕ)
  γ_p_h = mean(γ_p ∘ hₕ)

  # Terms
  _I = one(SymTensorValue{3,Float64})
  σf(u,p) = 2μ*ε(u) - p*_I
  a_Ω(∇u,∇v) = μ*(∇u ⊙ ∇v)
  b_Ω(div_v,p) = -p*(div_v)
  ab_Γ(u,∇u,v,∇v,p,q,n) = n ⋅ ( - μ*(∇u ⋅ v + ∇v ⋅ u) + v*p + u*q) + γ_Nu_h*(u⋅v)
  ju(∇u,∇v) = γ_u_h*(jump(Ω.n_Γg ⋅ ∇u) ⋅ jump(Ω.n_Γg ⋅ ∇v))
  jp(p,q) = γ_p_h*(jump(p) * jump(q))
  v_ψ(p,q) = k_p * Ω.ψ_f*p*q

  function a_fluid((),(u,p),(v,q),φ)
    ∇u = ∇(u); ∇v = ∇(v);
    div_u = ∇⋅u; div_v = ∇⋅v
    n_Γ = -get_normal_vector(Ω.Γ)
    return ∫(a_Ω(∇u,∇v) + b_Ω(div_v,p) + b_Ω(div_u,q) + v_ψ(p,q))Ω.dΩf +
      ∫(ab_Γ(u,∇u,v,∇v,p,q,n_Γ))Ω.dΓ +
      ∫(ju(∇u,∇v))Ω.dΓg - ∫(jp(p,q))Ω.dΓi
  end

  l_fluid((),(v,q),φ) =  ∫(0q)Ω.dΩf

  ## Structure
  # Material parameters
  function lame_parameters(E,ν)
    λ = (E*ν)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    (λ, μ)
  end
  λs, μs = lame_parameters(0.1,0.05)
  # Stabilization
  α_Gd = 1e-3
  k_d = 1.0
  γ_Gd(h) = α_Gd*(λs + μs)*h^3
  γ_Gd_h = mean(γ_Gd ∘ hₕ)
  # Terms
  σ(ε) = λs*tr(ε)*_I + 2*μs*ε
  a_s_Ω(d,s) = ε(s) ⊙ (σ ∘ ε(d)) # Elasticity
  j_s_k(d,s) = γ_Gd_h*(jump(Ω.n_Γg ⋅ ∇(s)) ⋅ jump(Ω.n_Γg ⋅ ∇(d)))
  v_s_ψ(d,s) = (k_d*Ω.ψ_s)*(d⋅s) # Isolated volume term

  function a_solid(((u,p),),d,s,φ)
    return ∫(a_s_Ω(d,s))Ω.dΩs +
      ∫(j_s_k(d,s))Ω.dΓg +
      ∫(v_s_ψ(d,s))Ω.dΩs
  end
  function l_solid(((u,p),),s,φ)
    n = -get_normal_vector(Ω.Γ)
    return ∫(-(1-Ω.ψ_s)*(n ⋅ σf(u,p)) ⋅ s)Ω.dΓ
  end

  ## Optimisation functionals
  iso_vol_frac(φ) = ∫(Ω.ψ_s/vol_D)Ω.dΩs
  J_comp(((u,p),d),φ) = ∫(ε(d) ⊙ (σ ∘ ε(d)))Ω.dΩs
  Vol(((u,p),d),φ) = ∫(1/vol_D)Ω.dΩs - ∫(vf/vol_D)dΩ_act
  dVol(q,(u,p,d),φ) = ∫(-1/vol_D*q/(abs(Ω.n_Γ ⋅ ∇(φ))))Ω.dΓ

  ## Staggered operators
  fluid_ls = PETScLinearSolver()
  elast_ls = PETScLinearSolver()
  solver = StaggeredFESolver([fluid_ls,elast_ls]);

  al_keys = [:J,:Vol]
  al_bundles = Dict(:C => [:Vol,])
  history = OptimiserHistory(Float64,al_keys,al_bundles,1000,i_am_main(ranks))

  for it = I0:IJ:IF
    pload!(result_path*"/data/LSF_$it",get_free_dof_values(φh))
    update_collection!(Ω,φh)
    (UP,VQ),(R,T) = build_spaces(Ω.Ω_act_s,Ω.Ω_act_f)
    op = StaggeredAffineFEOperator([(x,u,v)->a_fluid(x,u,v,φh),(x,u,v)->a_solid(x,u,v,φh)],
      [(x,v)->l_fluid(x,v,φh),(x,v)->l_solid(x,v,φh)],[UP,R],[VQ,T])
    uh,ph,dh = solve(solver,op)

    _J = sum(J_comp(((uh,ph),dh),φh))
    _Vol = sum(Vol(((uh,ph),dh),φh))
    push!(history,(_J,_Vol))

    write_history(path*"/history.txt",history;ranks)
    writevtk(Ω_act,files_path*"Omega_act_$it",
      cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"ph"=>ph,"dh"=>dh,"ψ_s"=>Ω.ψ_s,"ψ_f"=>Ω.ψ_f])
    writevtk(Ω.Ωf,files_path*"Omega_f_$it",cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
    writevtk(Ω.Ωs,files_path*"Omega_s_$it",cellfields=["uh"=>uh,"ph"=>ph,"dh"=>dh])
    out_paths = "$files_path/Omega_s_$it $files_path/Omega_f_$it $files_path/Omega_act_$it"
    i_am_main(ranks) && run(`tar -czf $files_path/data_$it.tar.gz $out_paths && rm -r $out_paths`)
  end
end

with_mpi() do distribute
  ranks = distribute(LinearIndices((ncpus,)))
  petsc_options = "-ksp_converged_reason -ksp_error_if_not_converged true -pc_type lu -pc_factor_mat_solver_type superlu_dist"
  GridapPETSc.with(;args=split(petsc_options)) do
    main(ranks)
  end
end
