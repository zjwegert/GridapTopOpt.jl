using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.MultiField, Gridap.TensorValues
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapSolvers, GridapSolvers.BlockSolvers, GridapSolvers.NonlinearSolvers
using GridapGmsh
using GridapTopOpt
using Mustache
using DataFrames
using JLD2

using LinearAlgebra
LinearAlgebra.norm(x::VectorValue,p::Real) = norm(x.data,p)

function create_mesh(fetype,path,mesh_size)
  if fetype==:bodyfitted
    if isfile("$(path)body_fitted_$mesh_size.msh")
      return "$(path)/body_fitted_$mesh_size.msh"
    else
      println("     Creating body-fitted mesh")
      open("$(path)/body_fitted_$mesh_size.geo","w") do f
        content = Mustache.render(read("$(@__DIR__)/body_fitted.geo",String), (;mesh_size,))
        write(f,content)
      end
      run(`gmsh "$(path)/body_fitted_$mesh_size.geo" -2 "$(path)/body_fitted_$mesh_size.msh"`)
      # rm("$(path)/body_fitted_$mesh_size.geo")
      return "$(path)/body_fitted_$mesh_size.msh"
    end
  elseif fetype==:cutfem
    if isfile("$(path)/cutfem_$mesh_size.msh")
      return "$(path)/cutfem_$mesh_size.msh"
    else
      println("     Creating CutFEM background mesh")
      open("$(path)/cutfem_$mesh_size.geo","w") do f
        content = Mustache.render(read("$(@__DIR__)/cutfem.geo",String), (;mesh_size,))
        write(f,content)
      end
      run(`gmsh "$(path)/cutfem_$mesh_size.geo" -2 "$(path)/cutfem_$mesh_size.msh"`)
      # rm("$(path)/cutfem_$mesh_size.geo")
      return "$(path)/cutfem_$mesh_size.msh"
    end
  else
    error("Invalid `fetype`, should be :bodyfitted or :cutfem")
  end
end

function main_cutfem(data,params;R=0.1,c=(0.5,0.2))
  Re,mesh_size,α_Nu,α_SUPG,α_GPμ,α_GPp,α_GPu,βp,βμ = params

  path = "./results/NavierStokes-2D-Verification/"
  data_path = "$path/vtk_data/"
  mkpath(data_path)

  # Create the mesh
  mesh_file =create_mesh(:cutfem,path,mesh_size)

  # Load mesh
  model = GmshDiscreteModel(mesh_file)

  Ω_act = Triangulation(model)
  hₕ = CellField(get_element_diameters(model),Ω_act)

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)

  x0,y0 = c # Disk centroid
  φh = interpolate(((x,y),)->sqrt((x-x0)^2+(y-y0)^2)-R,V_φ)

  # Setup integration meshes and measures
  order = 1
  degree = 2*(order+1)

  dΩ_act = Measure(Ω_act,degree)
  Γf_D = BoundaryTriangulation(model,tags="Gamma_f_D")
  Γf_N = BoundaryTriangulation(model,tags="Gamma_f_N")
  dΓf_D = Measure(Γf_D,degree)
  dΓf_N = Measure(Γf_N,degree)
  Ω = EmbeddedCollection(model,φh) do cutgeo,_
    Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
    Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
    Γg = GhostSkeleton(cutgeo)
    Ω_act_f = Triangulation(cutgeo,ACTIVE_OUT)
    (;
      :Ωf      => Ωf,
      :dΩf     => Measure(Ωf,degree),
      :Γg      => Γg,
      :dΓg     => Measure(Γg,degree),
      :n_Γg    => get_normal_vector(Γg),
      :Γ       => Γ,
      :dΓ      => Measure(Γ,degree),
      :Ω_act_f => Ω_act_f,
      :dΩ_act_f => Measure(Ω_act_f,degree),
      :ψ_f     => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_f_D"];groups=((GridapTopOpt.CUT,OUT),IN)),
    )
  end

  # Setup spaces
  uin(x) = VectorValue(16x[2]*(1/2-x[2]),0.0)
  uin_dot_e1(x) = uin(x)⋅VectorValue(1.0,0.0)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order,space=:P)
  reffe_p = ReferenceFE(lagrangian,Float64,order,space=:P)

  V = TestFESpace(Ω.Ω_act_f,reffe_u,conformity=:H1,
    dirichlet_tags=["Gamma_f_D","Gamma_NoSlipTop","Gamma_NoSlipBottom"])
  Q = TestFESpace(Ω.Ω_act_f,reffe_p,conformity=:H1)

  # Trial spaces
  U = TrialFESpace(V,[uin,VectorValue(0.0,0.0),VectorValue(0.0,0.0)])
  P = TrialFESpace(Q)

  # Multifield spaces
  X = MultiFieldFESpace([U,P])
  Y = MultiFieldFESpace([V,Q])

  ### Weak form

  ## Fluid
  # Properties
  # Re = 60 # Reynolds number
  ρ = 1.0 # Density
  cl = 2R # Characteristic length
  u0_max = sum(∫(uin_dot_e1)dΓf_D)/sum(∫(1)dΓf_D)
  μ = ρ*cl*u0_max/Re # Viscosity
  ν = μ/ρ # Kinematic viscosity

  # Stabilization parameters
  γ_Nu(h,u)    = α_Nu*(μ/h + ρ*norm(u,Inf)/6) # (Eqn. 13, Villanueva and Maute, 2017)
  τ_SUPG(h,u)  = α_SUPG*((2norm(u)/h)^2 + 9*(4ν/h^2)^2)^-0.5 # (Eqn. 31, Peterson et al., 2018)
  τ_PSPG(h,u)  = τ_SUPG(h,u) # (Sec. 3.2.2, Peterson et al., 2018)
  γ_GPμ(h)     = α_GPμ*μ*h # (Eqn. 32, Villanueva and Maute, 2017)
  γ_GPp(h,u)   = α_GPp*(μ/h+ρ*norm(u,Inf)/6)^-1*h^2 # (Eqn. 35, Villanueva and Maute, 2017)
  γ_GPu(h,un)  = α_GPu*ρ*abs(un)*h^2 # (Eqn. 37, Villanueva and Maute, 2017)
  γ_GPu(h,u,n) = (γ_GPu ∘ (h.plus,(u⋅n).plus) + γ_GPu ∘ (h.minus,(u⋅n).minus))/2
  k_p          = 1.0 # (Villanueva and Maute, 2017)

  # Terms
  δ = one(SymTensorValue{2,Float64})
  σ_f(ε,p) = -p*δ + 2μ*ε
  σ_f_β(ε,p) = -βp*p*δ + βμ*2μ*ε

  conv(u,∇u) = (∇u') ⋅ u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

  r_conv(u,v) = ρ*v ⋅ (conv∘(u,∇(u)))
  r_Ωf((u,p),(v,q)) = ε(v) ⊙ (σ_f ∘ (ε(u),p)) + q*(∇⋅u)
  r_Γ((u,p),(v,q),n,w) = -v⋅((σ_f ∘ (ε(u),p))⋅n) - u⋅((σ_f_β ∘ (ε(v),q))⋅n) + (γ_Nu ∘ (hₕ,w))*u⋅v
  r_ψ(p,q) = k_p * Ω.ψ_f*p*q
  r_SUPG((u,p),(v,q),w) = ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (u,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅
    (ρ*(conv∘(u,∇(u))) + ∇(p) - μ*Δ(u))
  r_SUPG_picard((u,p),(v,q),w) = ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (w,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅
    (ρ*(conv∘(w,∇(u))) + ∇(p) - μ*Δ(u))
  r_GP_μ(u,v) = mean(γ_GPμ ∘ hₕ)*jump(Ω.n_Γg ⋅ ∇(u)) ⋅ jump(Ω.n_Γg ⋅ ∇(v))
  r_GP_p(p,q,w) = mean(γ_GPp ∘ (hₕ,w))*jump(Ω.n_Γg ⋅ ∇(p)) * jump(Ω.n_Γg ⋅ ∇(q))
  r_GP_u(u,v,w,n) = γ_GPu(hₕ,w,n)*jump(Ω.n_Γg ⋅ ∇(u)) ⋅ jump(Ω.n_Γg ⋅ ∇(v))

  dr_conv(u,du,v) = ρ*v ⋅ (dconv∘(du,∇(du),u,∇(u)))
  dr_SUPG((u,p),(du,dp),(v,q),w) =
    ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (du,∇(v))))⋅(ρ*(conv∘(u,∇(u))) + ∇(p) - μ*Δ(u)) +
    ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (u,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅(ρ*(dconv∘(du,∇(du),u,∇(u))) + ∇(dp) - μ*Δ(du))

  function res_fluid((u,p),(v,q))
    n_Γ = -get_normal_vector(Ω.Γ)
    return ∫(r_conv(u,v) + r_Ωf((u,p),(v,q)))Ω.dΩf +
      ∫(r_SUPG((u,p),(v,q),u))Ω.dΩ_act_f +
      ∫(r_ψ(p,q))Ω.dΩf +
      ∫(r_Γ((u,p),(v,q),n_Γ,u))Ω.dΓ +
      ∫(r_GP_μ(u,v) + r_GP_p(p,q,u) + r_GP_u(u,v,u,Ω.n_Γg))Ω.dΓg
  end

  function jac_fluid_picard((u,p),(du,dp),(v,q))
    n_Γ = -get_normal_vector(Ω.Γ)
    return ∫(ρ*v ⋅ (conv∘(u,∇(du))) + r_Ωf((du,dp),(v,q)))Ω.dΩf +
      ∫(r_SUPG_picard((du,dp),(v,q),u))Ω.dΩ_act_f +
      ∫(r_ψ(dp,q))Ω.dΩf +
      ∫(r_Γ((du,dp),(v,q),n_Γ,u))Ω.dΓ +
      ∫(r_GP_μ(du,v) + r_GP_p(dp,q,u) + r_GP_u(du,v,u,Ω.n_Γg))Ω.dΓg
  end

  function jac_fluid_newton((u,p),(du,dp),(v,q))
    n_Γ = -get_normal_vector(Ω.Γ)
    return ∫(dr_conv(u,du,v) + r_Ωf((du,dp),(v,q)))Ω.dΩf +
      ∫(dr_SUPG((u,p),(du,dp),(v,q),u))Ω.dΩ_act_f +
      ∫(r_ψ(dp,q))Ω.dΩf +
      ∫(r_Γ((du,dp),(v,q),n_Γ,u))Ω.dΓ +
      ∫(r_GP_μ(du,v) + r_GP_p(dp,q,u) + r_GP_u(du,v,u,Ω.n_Γg))Ω.dΓg
  end

  op = FEOperator(res_fluid,jac_fluid_newton,X,Y)
  solver = NewtonSolver(LUSolver();maxiter=100,rtol=1.e-14,verbose=true)
  uh,ph = solve(solver,op);

  fname="cutfem_Re$(Re)_msh$(mesh_size)_aNU$(α_Nu)_aSUPG$(round(α_SUPG,sigdigits=2))_aGPmu$(α_GPμ)_aGPp$(α_GPp)_aGPu$(α_GPu)_Bp$(βp)_Bmu$(βμ)"
  writevtk(Ω.Ωf,data_path*fname,
    cellfields=["uh"=>uh,"ph"=>ph])

  n_Γ = -get_normal_vector(Ω.Γ)
  mass_flow_rate = sum(∫(uh ⋅ n_Γ)Ω.dΓ)
  surface_force = sum(∫(σ_f(ε(uh),ph) ⋅ n_Γ)Ω.dΓ)
  pressure_difference = sum(∫(ph+ρ/2*(uh⋅uh))dΓf_D-∫(ph+ρ/2*(uh⋅uh))dΓf_N)

  println("Mass flow rate: ",mass_flow_rate)
  println("Surface force: ",surface_force)
  println("Pressure difference: ",pressure_difference)
  push!(data,("cutfem",mesh_size,α_Nu,α_SUPG,α_GPμ,α_GPp,α_GPu,βp,βμ,mass_flow_rate,surface_force,pressure_difference,Re))
  jldsave((@__DIR__)*"/data.jld2";data)
  GC.gc()
  nothing
end

function main_fcm(data,params;R=0.1,c=(0.5,0.2))
  Re,mesh_size,α_Nu,α_SUPG,α_GPμ,α_GPp,α_GPu,βp,βμ = params

  path = "./results/NavierStokes-2D-Verification/"
  data_path = "$path/vtk_data/"
  mkpath(data_path)

  # Create the mesh
  mesh_file =create_mesh(:cutfem,path,mesh_size)

  # Load mesh
  model = GmshDiscreteModel(mesh_file)

  Ω_act = Triangulation(model)
  hₕ = CellField(get_element_diameters(model),Ω_act)

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)

  x0,y0 = c # Disk centroid
  φh = interpolate(((x,y),)->sqrt((x-x0)^2+(y-y0)^2)-R,V_φ)

  # Setup integration meshes and measures
  order = 1
  degree = 2*(order+1)

  dΩ_act = Measure(Ω_act,degree)
  Γf_D = BoundaryTriangulation(model,tags="Gamma_f_D")
  Γf_N = BoundaryTriangulation(model,tags="Gamma_f_N")
  dΓf_D = Measure(Γf_D,degree)
  dΓf_N = Measure(Γf_N,degree)
  Ω = EmbeddedCollection(model,φh) do cutgeo,_
    Ωs = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
    Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
    Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
    Γg = GhostSkeleton(cutgeo)
    Ω_act_s = Triangulation(cutgeo,ACTIVE)
    Ω_act_f = Triangulation(cutgeo,ACTIVE_OUT)
    (;
      :Ωs      => Ωs,
      :dΩs     => Measure(Ωs,degree),
      :Ωf      => Ωf,
      :dΩf     => Measure(Ωf,degree),
      :Γg      => Γg,
      :dΓg     => Measure(Γg,degree),
      :n_Γg    => get_normal_vector(Γg),
      :Γ       => Γ,
      :dΓ      => Measure(Γ,degree),
      :Ω_act_s => Ω_act_s,
      :dΩ_act_s => Measure(Ω_act_s,degree),
      :Ω_act_f => Ω_act_f,
      :dΩ_act_f => Measure(Ω_act_f,degree)
    )
  end

  # Setup spaces
  uin(x) = VectorValue(16x[2]*(1/2-x[2]),0.0)
  uin_dot_e1(x) = uin(x)⋅VectorValue(1.0,0.0)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order,space=:P)
  reffe_p = ReferenceFE(lagrangian,Float64,order,space=:P)

  V = TestFESpace(Ω_act,reffe_u,conformity=:H1,
    dirichlet_tags=["Gamma_f_D","Gamma_NoSlipTop","Gamma_NoSlipBottom"])
  Q = TestFESpace(Ω_act,reffe_p,conformity=:H1)

  # Trial spaces
  U = TrialFESpace(V,[uin,VectorValue(0.0,0.0),VectorValue(0.0,0.0)])
  P = TrialFESpace(Q)

  # Multifield spaces
  X = MultiFieldFESpace([U,P])
  Y = MultiFieldFESpace([V,Q])

  ### Weak form

  ## Fluid
  # Properties
  # Re = 60 # Reynolds number
  ρ = 1.0 # Density
  cl = 2R # Characteristic length
  u0_max = sum(∫(uin_dot_e1)dΓf_D)/sum(∫(1)dΓf_D)
  μ = ρ*cl*u0_max/Re # Viscosity
  ν = μ/ρ # Kinematic viscosity

  # Stabilization parameters
  γ_Nu         = α_Nu*(μ/0.01^2)
  τ_SUPG(h,u)  = α_SUPG*((2norm(u)/h)^2 + 9*(4ν/h^2)^2)^-0.5 # (Eqn. 31, Peterson et al., 2018)
  τ_PSPG(h,u)  = τ_SUPG(h,u) # (Sec. 3.2.2, Peterson et al., 2018)

  # Terms
  δ = one(SymTensorValue{2,Float64})
  σ_f(ε,p) = -p*δ + 2μ*ε
  σ_f_β(ε,p) = -βp*p*δ + βμ*2μ*ε

  conv(u,∇u) = (∇u') ⋅ u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

  r_conv(u,v) = ρ*v ⋅ (conv∘(u,∇(u)))
  r_Ωf((u,p),(v,q)) = ε(v) ⊙ (σ_f ∘ (ε(u),p)) + q*(∇⋅u)

  # r_SUPG((u,p),(v,q),w) = ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (u,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅
  #   (ρ*(conv∘(u,∇(u))) + ∇(p) - μ*Δ(u))
  # r_SUPG_picard((u,p),(v,q),w) = ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (w,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅
  #   (ρ*(conv∘(w,∇(u))) + ∇(p) - μ*Δ(u))

  # dr_conv(u,du,v) = ρ*v ⋅ (dconv∘(du,∇(du),u,∇(u)))
  # dr_SUPG((u,p),(du,dp),(v,q),w) =
  #   ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (du,∇(v))))⋅(ρ*(conv∘(u,∇(u))) + ∇(p) - μ*Δ(u)) +
  #   ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (u,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅(ρ*(dconv∘(du,∇(du),u,∇(u))) + ∇(dp) - μ*Δ(du))

  # function res_fluid((u,p),(v,q))
  #   return ∫(r_conv(u,v) + r_Ωf((u,p),(v,q)))Ω.dΩf +
  #     ∫(r_conv(u,v) + r_Ωf((u,p),(v,q)) + γ_Nu*(u⋅v))Ω.dΩs +
  #     ∫(r_SUPG((u,p),(v,q),u))dΩ_act
  # end

  # function jac_fluid_picard((u,p),(du,dp),(v,q))
  #   return ∫(ρ*v ⋅ (conv∘(u,∇(du))) + r_Ωf((du,dp),(v,q)))Ω.dΩs +
  #   ∫(ρ*v ⋅ (conv∘(u,∇(du))) + r_Ωf((du,dp),(v,q)) + γ_Nu*(du⋅v))Ω.dΩf +
  #     ∫(r_SUPG_picard((du,dp),(v,q),u))dΩ_act
  # end

  # function jac_fluid_newton((u,p),(du,dp),(v,q))
  #   return ∫(dr_conv(u,du,v) + r_Ωf((du,dp),(v,q)))Ω.dΩf +
  #     ∫(dr_conv(u,du,v) + r_Ωf((du,dp),(v,q)) + γ_Nu*(du⋅v))Ω.dΩs +
  #     ∫(dr_SUPG((u,p),(du,dp),(v,q),u))dΩ_act
  # end

  # Additional Brinkmann terms in SUPG based on 10.1002/nme.3151
  r_SUPG((u,p),(v,q),w;IN_Ωf=1) = (IN_Ωf*(τ_SUPG ∘ (hₕ,w))*(conv ∘ (u,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅
    (ρ*(conv∘(u,∇(u))) + ∇(p) - μ*Δ(u) + (1-IN_Ωf)*γ_Nu*u)
  r_SUPG_picard((u,p),(v,q),w;IN_Ωf=1) = (IN_Ωf*(τ_SUPG ∘ (hₕ,w))*(conv ∘ (w,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅
    (ρ*(conv∘(w,∇(u))) + ∇(p) - μ*Δ(u) + (1-IN_Ωf)*γ_Nu*u)

  dr_conv(u,du,v) = NS*ρ*v ⋅ (dconv∘(du,∇(du),u,∇(u)))
  dr_SUPG((u,p),(du,dp),(v,q),w;IN_Ωf=1) =
    (IN_Ωf*(τ_SUPG ∘ (hₕ,w))*(conv ∘ (du,∇(v))))⋅(ρ*(conv∘(u,∇(u))) + ∇(p) - μ*Δ(u) + (1-IN_Ωf)*γ_Nu*u) +
    (IN_Ωf*(τ_SUPG ∘ (hₕ,w))*(conv ∘ (u,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅(ρ*(dconv∘(du,∇(du),u,∇(u))) + ∇(dp) - μ*Δ(du) + (1-IN_Ωf)*γ_Nu*du)

  function res_fluid((u,p),(v,q))
    return ∫(r_conv(u,v) + r_Ωf((u,p),(v,q)) + r_SUPG((u,p),(v,q),u))Ω.dΩf +
      ∫(r_conv(u,v) + r_Ωf((u,p),(v,q)) + γ_Nu*(u⋅v) + r_SUPG((u,p),(v,q),u;IN_Ωf=0))Ω.dΩs
  end

  function jac_fluid_newton((u,p),(du,dp),(v,q))
    return ∫(dr_conv(u,du,v) + r_Ωf((du,dp),(v,q)) + dr_SUPG((u,p),(du,dp),(v,q),u))Ω.dΩf +
      ∫(dr_conv(u,du,v) + r_Ωf((du,dp),(v,q)) + γ_Nu*(du⋅v) + dr_SUPG((u,p),(du,dp),(v,q),u;IN_Ωf=0))Ω.dΩs
  end

  function jac_fluid_picard((u,p),(du,dp),(v,q))
    return ∫(ρ*v ⋅ (conv∘(u,∇(du))) + r_Ωf((du,dp),(v,q)) + r_SUPG_picard((du,dp),(v,q),u))Ω.dΩs +
      ∫(ρ*v ⋅ (conv∘(u,∇(du))) + r_Ωf((du,dp),(v,q)) + γ_Nu*(du⋅v) + r_SUPG_picard((du,dp),(v,q),u;IN_Ωf=0))Ω.dΩf
  end

  jac_fluid_AD((),x,dx,y,φ) = jacobian(_x->res_fluid(_x,y),x)

  op = FEOperator(res_fluid,jac_fluid_newton,X,Y)
  solver = NewtonSolver(LUSolver();maxiter=100,rtol=1.e-14,verbose=true)
  uh,ph = solve(solver,op);

  fname="fcm_Re$(Re)_msh$(mesh_size)_aNU$(α_Nu)_aSUPG$(round(α_SUPG,sigdigits=2))_aGPmu$(α_GPμ)_aGPp$(α_GPp)_aGPu$(α_GPu)_Bp$(βp)_Bmu$(βμ)"
  writevtk(Ω.Ωf,data_path*fname,
    cellfields=["uh"=>uh,"ph"=>ph])
  fname="fcm_GAMMA_Re$(Re)_msh$(mesh_size)_aNU$(α_Nu)_aSUPG$(round(α_SUPG,sigdigits=2))_aGPmu$(α_GPμ)_aGPp$(α_GPp)_aGPu$(α_GPu)_Bp$(βp)_Bmu$(βμ)"
  writevtk(Ω.Γ,data_path*fname,
    cellfields=["uh"=>uh,"ph"=>ph])

  n_Γ = -get_normal_vector(Ω.Γ)
  mass_flow_rate = sum(∫(uh ⋅ n_Γ)Ω.dΓ)
  surface_force = sum(∫(σ_f(ε(uh),ph) ⋅ n_Γ)Ω.dΓ)
  pressure_difference = sum(∫(ph+ρ/2*(uh⋅uh))dΓf_D-∫(ph+ρ/2*(uh⋅uh))dΓf_N)

  println("Mass flow rate: ",mass_flow_rate)
  println("Surface force: ",surface_force)
  println("Pressure difference: ",pressure_difference)
  push!(data,("fcm",mesh_size,α_Nu,α_SUPG,α_GPμ,α_GPp,α_GPu,βp,βμ,mass_flow_rate,surface_force,pressure_difference,Re))
  jldsave((@__DIR__)*"/data.jld2";data)
  GC.gc()
  nothing
end

function main_bodyfitted(data,params;R=0.1,c=(0.5,0.2))
  Re,mesh_size,α_Nu,α_SUPG,α_GPμ,α_GPp,α_GPu,βp,βμ = params

  path = "./results/NavierStokes-2D-Verification/"
  data_path = "$path/vtk_data/"
  mkpath(data_path)

  # Create the mesh
  mesh_file =create_mesh(:bodyfitted,path,mesh_size)

  # Load mesh
  model = GmshDiscreteModel(mesh_file)

  Ω = Triangulation(model)
  hₕ = CellField(get_element_diameters(model),Ω)

  # Cut the background model
  reffe_scalar = ReferenceFE(lagrangian,Float64,1)
  V_φ = TestFESpace(model,reffe_scalar)

  x0,y0 = c # Disk centroid
  φh = interpolate(((x,y),)->-sqrt((x-x0)^2+(y-y0)^2)+R,V_φ)

  # Setup integration meshes and measures
  order = 1
  degree = 2*(order+1)

  dΩ = Measure(Ω,degree)
  Γf_D = BoundaryTriangulation(model,tags="Gamma_f_D")
  Γf_N = BoundaryTriangulation(model,tags="Gamma_f_N")
  dΓf_D = Measure(Γf_D,degree)
  dΓf_N = Measure(Γf_N,degree)
  Γ = BoundaryTriangulation(model,tags="Gamma_fsi")
  dΓ = Measure(Γ,degree)
  n_Γ = get_normal_vector(Γ)

  # Setup spaces
  uin(x) = VectorValue(16x[2]*(1/2-x[2]),0.0)
  uin_dot_e1(x) = uin(x)⋅VectorValue(1.0,0.0)

  reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order,space=:P)
  reffe_p = ReferenceFE(lagrangian,Float64,order,space=:P)

  V = TestFESpace(Ω,reffe_u,conformity=:H1,
    dirichlet_tags=["Gamma_f_D","Gamma_NoSlipTop","Gamma_NoSlipBottom","Gamma_fsi"])
  Q = TestFESpace(Ω,reffe_p,conformity=:H1)

  # Trial spaces
  U = TrialFESpace(V,[uin,VectorValue(0.0,0.0),VectorValue(0.0,0.0),VectorValue(0.0,0.0)])
  P = TrialFESpace(Q)

  # Multifield spaces
  X = MultiFieldFESpace([U,P])
  Y = MultiFieldFESpace([V,Q])

  ### Weak form

  ## Fluid
  # Properties
  # Re = 60 # Reynolds number
  ρ = 1.0 # Density
  cl = 2R # Characteristic length
  u0_max = sum(∫(uin_dot_e1)dΓf_D)/sum(∫(1)dΓf_D)
  μ = ρ*cl*u0_max/Re # Viscosity
  ν = μ/ρ # Kinematic viscosity

  # Stabilization parameters
  τ_SUPG(h,u)  = α_SUPG*((2norm(u)/h)^2 + 9*(4ν/h^2)^2)^-0.5 # (Eqn. 31, Peterson et al., 2018)
  τ_PSPG(h,u)  = τ_SUPG(h,u) # (Sec. 3.2.2, Peterson et al., 2018)

  # Terms
  δ = one(SymTensorValue{2,Float64})
  σ_f(ε,p) = -p*δ + 2μ*ε
  σ_f_β(ε,p) = -βp*p*δ + βμ*2μ*ε

  conv(u,∇u) = (∇u') ⋅ u
  dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

  r_conv(u,v) = ρ*v ⋅ (conv∘(u,∇(u)))
  r_Ωf((u,p),(v,q)) = ε(v) ⊙ (σ_f ∘ (ε(u),p)) + q*(∇⋅u)
  r_SUPG((u,p),(v,q),w) = ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (u,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅
    (ρ*(conv∘(u,∇(u))) + ∇(p) - μ*Δ(u))
  r_SUPG_picard((u,p),(v,q),w) = ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (w,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅
    (ρ*(conv∘(w,∇(u))) + ∇(p) - μ*Δ(u))

  dr_conv(u,du,v) = ρ*v ⋅ (dconv∘(du,∇(du),u,∇(u)))
  dr_SUPG((u,p),(du,dp),(v,q),w) =
  ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (du,∇(v))))⋅(ρ*(conv∘(u,∇(u))) + ∇(p) - μ*Δ(u)) +
  ((τ_SUPG ∘ (hₕ,w))*(conv ∘ (u,∇(v))) + (τ_PSPG ∘ (hₕ,w))/ρ*∇(q))⋅(ρ*(dconv∘(du,∇(du),u,∇(u))) + ∇(dp) - μ*Δ(du))

  function res_fluid((u,p),(v,q))
    return ∫(r_conv(u,v) + r_Ωf((u,p),(v,q)))dΩ +
      ∫(r_SUPG((u,p),(v,q),u))dΩ
  end

  function jac_fluid_picard((u,p),(du,dp),(v,q))
    return ∫(ρ*v ⋅ (conv∘(u,∇(du))) + r_Ωf((du,dp),(v,q)))dΩ +
      ∫(r_SUPG_picard((du,dp),(v,q),u))dΩ
  end

  function jac_fluid_newton((u,p),(du,dp),(v,q))
    return ∫(dr_conv(u,du,v) + r_Ωf((du,dp),(v,q)))dΩ +
      ∫(dr_SUPG((u,p),(du,dp),(v,q),u))dΩ
  end

  op = FEOperator(res_fluid,jac_fluid_newton,X,Y)
  solver = NewtonSolver(LUSolver();maxiter=100,rtol=1.e-14,verbose=true)
  uh,ph = solve(solver,op);

  fname="bodyfitted_Re$(Re)_msh$(round(mesh_size,sigdigits=2))_aSUPG$(round(α_SUPG,sigdigits=2))"
  writevtk(Ω,data_path*fname,
    cellfields=["uh"=>uh,"ph"=>ph])

  mass_flow_rate = sum(∫(uh⋅n_Γ)dΓ)
  surface_force = sum(∫(σ_f(ε(uh),ph) ⋅ n_Γ)dΓ)
  pressure_difference = sum(∫(ph+ρ/2*(uh⋅uh))dΓf_D-∫(ph+ρ/2*(uh⋅uh))dΓf_N)

  println("Mass flow rate: ",mass_flow_rate)
  println("Surface force: ",surface_force)
  println("Pressure difference: ",pressure_difference)
  push!(data,("bodyfitted",mesh_size,α_Nu,α_SUPG,α_GPμ,α_GPp,α_GPu,βp,βμ,mass_flow_rate,surface_force,pressure_difference,Re))
  jldsave((@__DIR__)*"/data.jld2";data)
  GC.gc()
  nothing
end

# data = DataFrame(mesh_type=String[],mesh_size=Float64[],α_Nu=Float64[],α_SUPG=Float64[],α_GPμ=Float64[],α_GPp=Float64[],α_GPu=Float64[],βp=Float64[],βμ=Float64[],mass_flow_rate=Float64[],surface_force=[],pressure_difference=Float64[])
# params = (;Re=60,mesh_size=0.01,α_Nu=0,α_SUPG=1/3,α_GPμ=0,α_GPp=0,α_GPu=0,βp=0,βμ=0)
# main_bodyfitted(data,params)
# params = (;Re=60,mesh_size=0.005,α_Nu=0,α_SUPG=1/3,α_GPμ=0,α_GPp=0,α_GPu=0,βp=0,βμ=0)
# main_bodyfitted(data,params)
# params = (;Re=60,mesh_size=1e-3,α_Nu=0,α_SUPG=1/3,α_GPμ=0,α_GPp=0,α_GPu=0,βp=0,βμ=0)
# main_bodyfitted(data,params)

# params = (;Re=60,mesh_size=0.01,α_Nu=100,α_SUPG=1/3,α_GPμ=0.05,α_GPp=0.05,α_GPu=0.05,βp=1,βμ=1)
# main_cutfem(data,params)
# params = (;Re=60,mesh_size=0.01,α_Nu=1000,α_SUPG=1/3,α_GPμ=0.05,α_GPp=0.05,α_GPu=0.05,βp=1,βμ=1)
# main_cutfem(data,params)
# params = (;Re=60,mesh_size=0.01,α_Nu=10000,α_SUPG=1/3,α_GPμ=0.05,α_GPp=0.05,α_GPu=0.05,βp=1,βμ=1)
# main_cutfem(data,params)

# params = (;Re=60,mesh_size=0.01,α_Nu=100,α_SUPG=1/3,α_GPμ=0.05,α_GPp=0.05,α_GPu=0.05,βp=-1,βμ=1)
# main_cutfem(data,params)
# params = (;Re=60,mesh_size=0.01,α_Nu=1000,α_SUPG=1/3,α_GPμ=0.05,α_GPp=0.05,α_GPu=0.05,βp=-1,βμ=1)
# main_cutfem(data,params)
# params = (;Re=60,mesh_size=0.01,α_Nu=10000,α_SUPG=1/3,α_GPμ=0.05,α_GPp=0.05,α_GPu=0.05,βp=-1,βμ=1)
# main_cutfem(data,params)

# params = (;Re=60,mesh_size=0.01,α_Nu=100,α_SUPG=1/3,α_GPμ=0.001,α_GPp=0.001,α_GPu=0.001,βp=-1,βμ=1)
# main_cutfem(data,params)
# params = (;Re=60,mesh_size=0.01,α_Nu=1000,α_SUPG=1/3,α_GPμ=0.001,α_GPp=0.001,α_GPu=0.001,βp=-1,βμ=1)
# main_cutfem(data,params)
# params = (;Re=60,mesh_size=0.01,α_Nu=10000,α_SUPG=1/3,α_GPμ=0.001,α_GPp=0.001,α_GPu=0.001,βp=-1,βμ=1)
# main_cutfem(data,params)

# params = (;Re=60,mesh_size=0.005,α_Nu=100,α_SUPG=1/3,α_GPμ=0.01,α_GPp=0.01,α_GPu=0.01,βp=1,βμ=1)
# main_cutfem(data,params)
# params = (;Re=60,mesh_size=0.005,α_Nu=1000,α_SUPG=1/3,α_GPμ=0.01,α_GPp=0.01,α_GPu=0.01,βp=1,βμ=1)
# main_cutfem(data,params)
# params = (;Re=60,mesh_size=0.005,α_Nu=10000,α_SUPG=1/3,α_GPμ=0.01,α_GPp=0.01,α_GPu=0.01,βp=1,βμ=1)
# main_cutfem(data,params)

# params = (;Re=60,mesh_size=0.005,α_Nu=100,α_SUPG=1/3,α_GPμ=0.01,α_GPp=0.01,α_GPu=0.01,βp=-1,βμ=1)
# main_cutfem(data,params)
# params = (;Re=60,mesh_size=0.005,α_Nu=1000,α_SUPG=1/3,α_GPμ=0.01,α_GPp=0.01,α_GPu=0.01,βp=-1,βμ=1)
# main_cutfem(data,params)
# params = (;Re=60,mesh_size=0.005,α_Nu=10000,α_SUPG=1/3,α_GPμ=0.01,α_GPp=0.01,α_GPu=0.01,βp=-1,βμ=1)
# main_cutfem(data,params)

# params = (;Re=60,mesh_size=0.005,α_Nu=100,α_SUPG=1/3,α_GPμ=0.001,α_GPp=0.001,α_GPu=0.001,βp=-1,βμ=1)
# main_cutfem(data,params)
# params = (;Re=60,mesh_size=0.005,α_Nu=1000,α_SUPG=1/3,α_GPμ=0.001,α_GPp=0.001,α_GPu=0.001,βp=-1,βμ=1)
# main_cutfem(data,params)
# params = (;Re=60,mesh_size=0.005,α_Nu=10000,α_SUPG=1/3,α_GPμ=0.001,α_GPp=0.001,α_GPu=0.001,βp=-1,βμ=1)
# main_cutfem(data,params)

# params = (;Re=60,mesh_size=0.005,α_Nu=100,α_SUPG=1/3,α_GPμ=0.001,α_GPp=0.001,α_GPu=0.001,βp=-1,βμ=1)
# main_cutfem(data,params)
# params = (;Re=60,mesh_size=0.005,α_Nu=1000,α_SUPG=1/3,α_GPμ=0.001,α_GPp=0.001,α_GPu=0.001,βp=-1,βμ=1)
# main_cutfem(data,params)
# params = (;Re=60,mesh_size=0.005,α_Nu=10000,α_SUPG=1/3,α_GPμ=0.001,α_GPp=0.001,α_GPu=0.001,βp=-1,βμ=1)
# main_cutfem(data,params)

# params = (;Re=60,mesh_size=0.001,α_Nu=100,α_SUPG=1/3,α_GPμ=0.001,α_GPp=0.001,α_GPu=0.001,βp=-1,βμ=1)
# main_cutfem(data,params)
# params = (;Re=60,mesh_size=0.001,α_Nu=1000,α_SUPG=1/3,α_GPμ=0.001,α_GPp=0.001,α_GPu=0.001,βp=-1,βμ=1)
# main_cutfem(data,params)
# params = (;Re=60,mesh_size=0.001,α_Nu=10000,α_SUPG=1/3,α_GPμ=0.001,α_GPp=0.001,α_GPu=0.001,βp=-1,βμ=1)
# main_cutfem(data,params)

data = JLD2.load((@__DIR__)*"/data.jld2")["data"]
params = (;Re=60,mesh_size=0.005,α_Nu=2.5*100,α_SUPG=1/3,α_GPμ=0,α_GPp=0,α_GPu=0,βp=0,βμ=0)
main_fcm(data,params)
