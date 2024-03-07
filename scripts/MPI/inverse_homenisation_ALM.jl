using Gridap, Gridap.MultiField, GridapDistributed, GridapPETSc, GridapSolvers, 
  PartitionedArrays, LevelSetTopOpt, SparseMatricesCSR

"""
  (MPI) Maximum bulk modulus inverse homogenisation with augmented Lagrangian method in 2D.

  Optimisation problem:
      Min J(Ω) = -κ(Ω)
        Ω
    s.t., Vol(Ω) = vf,
          ⎡For unique εᴹᵢ, find uᵢ∈V=H¹ₚₑᵣ(Ω)ᵈ, 
          ⎣∫ ∑ᵢ C ⊙ ε(uᵢ) ⊙ ε(vᵢ) dΩ = ∫ -∑ᵢ C ⊙ ε⁰ᵢ ⊙ ε(vᵢ) dΩ, ∀v∈V.
""" 
function main(mesh_partition,distribute,el_size)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  ## Parameters
  order = 1
  xmax,ymax=(1.0,1.0)
  dom = (0,xmax,0,ymax)
  γ = 0.05
  γ_reinit = 0.5
  max_steps = floor(Int,minimum(el_size)/10)
  tol = 1/(5order^2)/minimum(el_size)
  C = isotropic_elast_tensor(2,1.,0.3)
  η_coeff = 2
  α_coeff = 4
  vf = 0.5
  path = dirname(dirname(@__DIR__))*"/results/inverse_homenisation_ALM_MPI"
  i_am_main(ranks) && mkdir(path)

  ## FE Setup
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size,isperiodic=(true,true));
  el_Δ = get_el_Δ(model)
  f_Γ_D(x) = iszero(x)
  update_labels!(1,model,f_Γ_D,"origin")

  ## Triangulations and measures
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2order)
  vol_D = sum(∫(1)dΩ)

  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["origin"])
  U = TrialFESpace(_V,VectorValue(0.0,0.0))
  V_reg = V_φ = TestFESpace(model,reffe_scalar)
  U_reg = TrialFESpace(V_reg)

  ## Create FE functions
  lsf_fn(x) = max(initial_lsf(2,0.4)(x),initial_lsf(2,0.4;b=VectorValue(0,0.5))(x));
  φh = interpolate(lsf_fn,V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  εᴹ = (TensorValue(1.,0.,0.,0.),     # ϵᵢⱼ⁽¹¹⁾≡ϵᵢⱼ⁽¹⁾
        TensorValue(0.,0.,0.,1.),     # ϵᵢⱼ⁽²²⁾≡ϵᵢⱼ⁽²⁾
        TensorValue(0.,1/2,1/2,0.))   # ϵᵢⱼ⁽¹²⁾≡ϵᵢⱼ⁽³⁾

  a(u,v,φ,dΩ) = ∫((I ∘ φ) * C ⊙ ε(u) ⊙ ε(v))dΩ
  l = [(v,φ,dΩ) -> ∫(-(I ∘ φ) * C ⊙ εᴹ[i] ⊙ ε(v))dΩ for i in 1:3]

  ## Optimisation functionals
  _C(C,ε_p,ε_q) = C ⊙ ε_p ⊙ ε_q;
  _K(C,(u1,u2,u3),εᴹ) = (_C(C,ε(u1)+εᴹ[1],εᴹ[1]) + _C(C,ε(u2)+εᴹ[2],εᴹ[2]) + 2*_C(C,ε(u1)+εᴹ[1],εᴹ[2]))/4
  _v_K(C,(u1,u2,u3),εᴹ) = (_C(C,ε(u1)+εᴹ[1],ε(u1)+εᴹ[1]) + _C(C,ε(u2)+εᴹ[2],ε(u2)+εᴹ[2]) + 2*_C(C,ε(u1)+εᴹ[1],ε(u2)+εᴹ[2]))/4    

  J(u,φ,dΩ) = ∫(-(I ∘ φ)*_K(C,u,εᴹ))dΩ
  dJ(q,u,φ,dΩ) = ∫(_v_K(C,u,εᴹ)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;
  Vol(u,φ,dΩ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ;
  dVol(q,u,φ,dΩ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  stencil = AdvectionStencil(FirstOrderStencil(2,Float64),model,V_φ,tol,max_steps)

  ## Setup solver and FE operators
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  solver = ElasticitySolver(V)
  
  state_map = RepeatingAffineFEStateMap(
    3,a,l,U,V,V_φ,U_reg,φh,dΩ;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
    ls = solver,adjoint_ls = solver
  )
  pcfs = PDEConstrainedFunctionals(J,[Vol],state_map;analytic_dJ=dJ,analytic_dC=[dVol])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ
  vel_ext = VelocityExtension(
    a_hilb,U_reg,V_reg,
    assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls = PETScLinearSolver()
  )
  
  ## Optimiser
  optimiser = AugmentedLagrangian(pcfs,stencil,vel_ext,φh;
    γ,γ_reinit,verbose=i_am_main(ranks),constraint_names=[:Vol])
  for (it, uh, φh) in optimiser
    write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi)|"=>(norm ∘ ∇(φh))])
    write_history(path*"/history.txt",optimiser.history;ranks=ranks)
  end
  it = get_history(optimiser).niter; uh = get_state(pcfs)
  write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi)|"=>(norm ∘ ∇(φh))];iter_mod=1)
end

with_mpi() do distribute
  mesh_partition = (2,2)
  el_size = (200,200)
  hilb_solver_options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
    -ksp_converged_reason -ksp_rtol 1.0e-12 -mat_block_size 3
    -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_coarse_sub_pc_type cholesky"
    
  GridapPETSc.with(args=split(hilb_solver_options)) do
    main(mesh_partition,distribute,el_size)
  end
end;