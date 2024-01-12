using Gridap, GridapDistributed, GridapPETSc, PartitionedArrays, LSTO_Distributed

"""
  ...
""" 
function main()
  ## Parameters
  order = 2;
  xmax,ymax=(1.0,1.0)
  prop_Γ_N = 0.4;
  dom = (0,xmax,0,ymax);
  el_size = (500,500);
  γ = 0.1;
  γ_reinit = 0.5;
  max_steps = floor(Int,minimum(el_size)/10)
  tol = 1/(order^2*10)*prod(inv,minimum(el_size))
  C = isotropic_2d(1.,0.3);
  η_coeff = 2;
  α_coeff = 4;
  g = VectorValue(0,-1);
  path = dirname(dirname(@__DIR__))*"/results/minimum_length_scale"

  ## FE Setup
  model = CartesianDiscreteModel(dom,el_size);
  Δ = get_Δ(model)
  f_Γ_D(x) = (x[1] ≈ 0.0) ? true : false;
  f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= 
    ymax/2+ymax*prop_Γ_N/4 + eps()) ? true : false;
  update_labels!(1,model,f_Γ_D,"Gamma_D")
  update_labels!(2,model,f_Γ_N,"Gamma_N")

  ## Triangulations and measures
  Ω = Triangulation(model)
  Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
  dΩ = Measure(Ω,2order)
  dΓ_N = Measure(Γ_N,2order)

  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D"])
  U = TrialFESpace(V,VectorValue(0.0,0.0))
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
  U_reg = TrialFESpace(V_reg,0)

  ## Create FE functions
  φh = interpolate(gen_lsf(4,0.2),V_φ);
  φ = get_free_dof_values(φh)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = 2*maximum(Δ),ϵₘ=0.0)
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  a(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(v)))dΩ
  l(v,φ,dΩ,dΓ_N) = ∫(v⋅g)dΓ_N
  res(u,v,φ,dΩ,dΓ_N) = a(u,v,φ,dΩ,dΓ_N) - l(v,φ,dΩ,dΓ_N)

  ## Optimisation functionals
  stencil = AdvectionStencil(FirstOrderStencil(2,Float64),model,V_φ,Δ./order,max_steps,tol)
  reinit!(stencil,φ,γ_reinit)

  uh = interpolate(x->x[1],V_φ)
  # f = φ->∫((DH ∘ φ)*(norm ∘ φ)*uh)dΩ;
  f = φ->∫((I ∘ φ)*uh)dΩ; # <- this corresponds to J_1Ω
  df = gradient(f,φh)
  dfh = FEFunction(V_φ, assemble_vector(df,V_φ))

  # df_analytic = q->∫(q*(∇(uh)⋅∇(φh)+laplacian(φh)*uh)*(DH ∘ φh)*(norm ∘ φh))dΩ
  df_analytic = q->∫(uh*q*(DH ∘ φh))dΩ#*(norm ∘ ∇(φh)))dΩ # <- assume |∇φ| = 1
  df_analytic_h = FEFunction(V_φ, assemble_vector(df_analytic,V_φ))

  make_dir(path)
  writevtk(Ω,"$path/TestVariational",cellfields=["dfh_analytic"=>df_analytic_h,"dfh"=>dfh])

  ## When it doesn't work:
  """
  Take
    J(Ω) = ∫ j(φ) dD
  where D is the whole bounding domain and φ is a signed distance function. Then,
    J'(Ω)(θ) = ... a very complicated expression.
"""

  return nothing
 


  P_minT = (u,φ,dΩ,dΓ_N) -> ∫(φ*φ*(_max ∘ φ)*(_max ∘ φ))dΩ # ∫(H ∘ φ)dΩ

  function dP_minT(q,uu,_φ,dΩ,dΓ_N)
    jp = 2_φ*(_max ∘ _φ)*(_max ∘ _φ)-_φ*_φ*(_max ∘ _φ) #DH ∘ _φ

    ω(_φ,_Δφ) = 1/(1 + 100*abs(_φ)^3.5*abs(_Δφ)^3.5)

    bi_form(u,v) = ∫(u*v*(DH ∘ _φ)*(norm ∘ ∇(_φ)) + (ω ∘ (_φ,laplacian(_φ)))*(∇(_φ) ⋅ ∇(u))*(∇(_φ) ⋅ ∇(v)) )dΩ;
    li_form(v) = ∫(-jp*v)dΩ

    op = AffineFEOperator(bi_form,li_form,V_φ,V_φ)
    xh = solve(op)
    ∫(xh*q*(DH ∘ _φ)*(norm ∘ ∇(_φ)))dΩ
  end

  ## Finite difference solver and level set function
  stencil = AdvectionStencil(FirstOrderStencil(2,Float64),model,V_φ,Δ./order,max_steps,tol)
  reinit!(stencil,φ,γ_reinit)

  ## Setup solver and FE operators
  state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh,dΩ,dΓ_N)
  # pcfs = PDEConstrainedFunctionals(P_minT,state_map)
  pcfs = PDEConstrainedFunctionals(P_minT,[P_minT],state_map,analytic_dC=[dP_minT])
  J_val,C_val,dJ,dC = Gridap.evaluate!(pcfs,φ)

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

  make_dir(path)
  # optimiser = AugmentedLagrangian(φ,pcfs,stencil,vel_ext,interp,el_size,γ,γ_reinit);
  # for history in optimiser
  #   it,Ji,_,_= last(history)
  #   print_history(it,["J"=>Ji])
  #   write_history(history,path*"/history.csv")
  #   uhi = get_state(pcfs)
  #   write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh))];iter_mod=1)
  # end
  # it,Ji,_,_ = last(optimiser.history)
  # print_history(it,["J"=>Ji])
  # write_history(optimiser.history,path*"/history.csv")
  # uhi = get_state(pcfs)
  # write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh))];iter_mod=1)

  @show J_val

  writevtk(Ω,"$path/TestVariational",cellfields=["φh"=>φh,"dJ"=>FEFunction(U_reg,dJ),"dJ_analytic"=>FEFunction(U_reg,dC[1])])
  maximum(abs,dJ-dC[1]),maximum(abs,dJ-dC[1])/maximum(abs,dC[1])
end

main()