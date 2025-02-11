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

function main_bodyfitted(data,params;R=0.1,c=(0.5,0.2))
  E,nu,mesh_size,α_Gd = params

  path = "./results/Elast-2D-Verification/"
  data_path = "$path/vtk_data/"
  mkpath(data_path)

  # Create the mesh
  mesh_file = create_mesh(:bodyfitted,path,mesh_size)

  # Load mesh
  model = GmshDiscreteModel(mesh_file)

  Ω = Triangulation(model)

  # Setup integration meshes and measures
  order = 2
  degree = 2*(order+1)

  dΩ = Measure(Ω,degree)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_f_N")
  dΓ_N = Measure(Γ_N,degree)
  Γ = BoundaryTriangulation(model,tags="Gamma_fsi")
  dΓ = Measure(Γ,degree)
  n_Γ = get_normal_vector(Γ)

  # Setup spaces
  reffe_d = ReferenceFE(lagrangian,VectorValue{2,Float64},order,space=:P)

  T = TestFESpace(Ω,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_f_D"])
  R = TrialFESpace(T)

  ### Weak form

  ## Structure
  # Material parameters
  function lame_parameters(E,ν)
    λ = (E*ν)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    (λ, μ)
  end
  λs, μs = lame_parameters(E,nu)
  g = VectorValue(0.0,-0.01)
  g2 = VectorValue(0.1,0.0)
  # Terms
  σ(ε) = λs*tr(ε)*one(ε) + 2*μs*ε
  a_Ω(d,s) = ε(s) ⊙ (σ ∘ ε(d)) # Elasticity

  function a_solid(d,s)
    return ∫(a_Ω(d,s))dΩ
  end
  function l_solid(s)
    return ∫(g ⋅ s)dΓ_N + ∫(g2 ⋅ s)dΓ
  end

  op = AffineFEOperator(a_solid,l_solid,R,T)
  uh = solve(op);

  fname="bodyfitted_elast_E$(E)_nu$(nu)_msh$(round(mesh_size,sigdigits=2))_aGd$(α_Gd)"
  writevtk(Ω,data_path*fname,cellfields=["uh"=>uh])
  fname="bodyfitted_elast_GAMMA_E$(E)_nu$(nu)_msh$(mesh_size)_aGd$(α_Gd)"
  writevtk(Γ,data_path*fname,
    cellfields=["uh"=>uh,"surface_force"=>(σ ∘ ε(uh)) ⋅ n_Γ])

  mass_flow_rate = sum(∫(uh⋅n_Γ)dΓ)
  surface_force = sum(∫((σ ∘ ε(uh)) ⋅ n_Γ)dΓ)

  println("Mass flow rate: ",mass_flow_rate)
  println("Surface force: ",surface_force)
  push!(data,("bodyfitted",mesh_size,E,nu,α_Gd,mass_flow_rate,surface_force))
  jldsave((@__DIR__)*"/data_elast.jld2";data)
  GC.gc()
  nothing
end

function main_cutfem(data,params;R=0.1,c=(0.5,0.2))
  E,nu,mesh_size,α_Gd = params

  path = "./results/Elast-2D-Verification/"
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
  Γ_N = BoundaryTriangulation(model,tags="Gamma_f_N")
  dΓ_N = Measure(Γ_N,degree)
  Ω = EmbeddedCollection(model,φh) do cutgeo,cutgeo_facets
    Ωs = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
    Ωf = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
    Γ  = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
    Γg = GhostSkeleton(cutgeo)
    Ω_act_s = Triangulation(cutgeo,ACTIVE)
    Ω_act_f = Triangulation(cutgeo,ACTIVE_OUT)
    Fi = SkeletonTriangulation(cutgeo_facets,ACTIVE_OUT)
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
      :dΩ_act_f => Measure(Ω_act_f,degree),
      :Fi => Fi,
      :dFi => Measure(Fi,degree),
      :n_Fi    => get_normal_vector(Fi),
      # :ψ_s     => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_s_D"];groups=(IN,(GridapTopOpt.CUT,OUT))),
      # :ψ_f     => GridapTopOpt.get_isolated_volumes_mask(cutgeo,["Gamma_f_D"];groups=(OUT,(GridapTopOpt.CUT,IN))),
    )
  end

  # Setup spaces
  reffe_d = ReferenceFE(lagrangian,VectorValue{2,Float64},order,space=:P)

  T = TestFESpace(Ω.Ω_act_f,reffe_d,conformity=:H1,dirichlet_tags=["Gamma_f_D"])
  R = TrialFESpace(T)

  ### Weak form

  ## Structure
  # Material parameters
  function lame_parameters(E,ν)
    λ = (E*ν)/((1+ν)*(1-2*ν))
    μ = E/(2*(1+ν))
    (λ, μ)
  end
  λs, μs = lame_parameters(0.1,0.05)
  g = VectorValue(0.0,-0.01)
  g2 = VectorValue(0.1,0.0)
  # Stabilization
  k_d = 1.0
  γ_Gd(h) = α_Gd*(λs + μs)*h^3
  # Terms
  σ(ε) = λs*tr(ε)*one(ε) + 2*μs*ε
  a_s_Ω(s,d) = ε(s) ⊙ (σ ∘ ε(d)) # Elasticity
  j_s_k(s,d) = mean(γ_Gd ∘ hₕ)*jump(Ω.n_Γg ⋅ ∇(s)) ⋅ jump(Ω.n_Γg ⋅ ∇(d))
  v_s_χ(s,d) = k_d*Ω.χ_s*d⋅s # Isolated volume term

  function a_solid(d,s)
    return ∫(a_s_Ω(s,d))Ω.dΩf + ∫(j_s_k(s,d))Ω.dΓg
  end
  function l_solid(s)
    return ∫(g ⋅ s)dΓ_N + ∫(g2 ⋅ s)Ω.dΓ
  end

  op = AffineFEOperator(a_solid,l_solid,R,T)
  uh = solve(op);

  n_Γ = -get_normal_vector(Ω.Γ)

  fname="cutfem_elast_E$(E)_nu$(nu)_msh$(round(mesh_size,sigdigits=2))_aGd$(α_Gd)"
  writevtk(Ω.Ωf,data_path*fname,cellfields=["uh"=>uh])
  fname="cutfem_elast_GAMMA_E$(E)_nu$(nu)_msh$(mesh_size)_aGd$(α_Gd)"
  writevtk(Ω.Γ,data_path*fname,
    cellfields=["uh"=>uh,"surface_force"=>(σ ∘ ε(uh)) ⋅ n_Γ])

  mass_flow_rate = sum(∫(uh⋅n_Γ)Ω.dΓ)
  surface_force = sum(∫((σ ∘ ε(uh)) ⋅ n_Γ)Ω.dΓ)

  println("Mass flow rate: ",mass_flow_rate)
  println("Surface force: ",surface_force)
  push!(data,("cutfem",mesh_size,E,nu,α_Gd,mass_flow_rate,surface_force))
  jldsave((@__DIR__)*"/data_elast.jld2";data)
  GC.gc()
  nothing
end

data = DataFrame(mesh_type=String[],mesh_size=Float64[],E=Float64[],nu=Float64[],α_Gd=Float64[],mass_flow_rate=Float64[],surface_force=[])
params = (;E=0.1,nu=0.05,mesh_size=0.01,α_Gd=0.0)
main_bodyfitted(data,params)
params = (;E=0.1,nu=0.05,mesh_size=0.005,α_Gd=0.0)
main_bodyfitted(data,params)
params = (;E=0.1,nu=0.05,mesh_size=0.0025,α_Gd=0.0)
main_bodyfitted(data,params)

params = (;E=0.1,nu=0.05,mesh_size=0.01,α_Gd=1e-7)
main_cutfem(data,params)
params = (;E=0.1,nu=0.05,mesh_size=0.005,α_Gd=1e-7)
main_cutfem(data,params)
params = (;E=0.1,nu=0.05,mesh_size=0.0025,α_Gd=1e-7)
main_cutfem(data,params)


# data = JLD2.load((@__DIR__)*"/data_elast.jld2")["data"]