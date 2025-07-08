using Gridap, Gridap.MultiField, GridapDistributed, GridapPETSc, GridapSolvers,
  PartitionedArrays, GridapTopOpt, SparseMatricesCSR

global elx       = parse(Int,ARGS[1])
global ely       = parse(Int,ARGS[2])
global elz       = parse(Int,ARGS[3])
global Px        = parse(Int,ARGS[4])
global Py        = parse(Int,ARGS[5])
global Pz        = parse(Int,ARGS[6])
global write_dir = ARGS[7]

"""
  (MPI) Maximum bulk modulus inverse homogenisation with augmented Lagrangian method in 3D.

  Optimisation problem:
      Min J(Ω) = -κ(Ω)
        Ω
    s.t., Vol(Ω) = vf,
          ⎡For unique εᴹᵢ, find uᵢ∈V=H¹ₚₑᵣ(Ω)ᵈ,
          ⎣∫ ∑ᵢ C ⊙ ε(uᵢ) ⊙ ε(vᵢ) dΩ = ∫ -∑ᵢ C ⊙ ε⁰ᵢ ⊙ ε(vᵢ) dΩ, ∀v∈V.
"""
function main(mesh_partition,distribute,el_size,path)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  ## Parameters
  order = 1
  xmax,ymax,zmax=(1.0,1.0,1.0)
  dom = (0,xmax,0,ymax,0,zmax)
  γ = 0.1
  γ_reinit = 0.5
  max_steps = floor(Int,order*minimum(el_size)/10)
  tol = 1/(5order^2)/minimum(el_size)
  C = isotropic_elast_tensor(3,1.,0.3)
  η_coeff = 2
  α_coeff = 4max_steps*γ
  vf = 0.5
  iter_mod = 10
  i_am_main(ranks) && mkpath(path)

  ## FE Setup
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size,isperiodic=(true,true,true))
  el_Δ = get_el_Δ(model)
  f_Γ_D(x) = iszero(x)
  update_labels!(1,model,f_Γ_D,"origin")

  ## Triangulations and measures
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*order)
  vol_D = sum(∫(1)dΩ)

  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["origin"])
  U = TrialFESpace(V,VectorValue(0.0,0.0,0.0))
  V_reg = V_φ = TestFESpace(model,reffe_scalar)
  U_reg = TrialFESpace(V_reg)

  ## Create FE functions
  lsf_fn(x) = cos(2π*x[1]) + cos(2π*x[2]) + cos(2π*x[3])
  φh = interpolate(lsf_fn,V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  εᴹ = (TensorValue(1.,0.,0.,0.,0.,0.,0.,0.,0.),           # ϵᵢⱼ⁽¹¹⁾≡ϵᵢⱼ⁽¹⁾
        TensorValue(0.,0.,0.,0.,1.,0.,0.,0.,0.),           # ϵᵢⱼ⁽²²⁾≡ϵᵢⱼ⁽²⁾
        TensorValue(0.,0.,0.,0.,0.,0.,0.,0.,1.),           # ϵᵢⱼ⁽³³⁾≡ϵᵢⱼ⁽³⁾
        TensorValue(0.,0.,0.,0.,0.,1/2,0.,1/2,0.),         # ϵᵢⱼ⁽²³⁾≡ϵᵢⱼ⁽⁴⁾
        TensorValue(0.,0.,1/2,0.,0.,0.,1/2,0.,0.),         # ϵᵢⱼ⁽¹³⁾≡ϵᵢⱼ⁽⁵⁾
        TensorValue(0.,1/2,0.,1/2,0.,0.,0.,0.,0.))         # ϵᵢⱼ⁽¹²⁾≡ϵᵢⱼ⁽⁶⁾

  a(u,v,φ) = ∫((I ∘ φ) * C ⊙ ε(u) ⊙ ε(v))dΩ
  l = [(v,φ) -> ∫(-(I ∘ φ) * C ⊙ εᴹ[i] ⊙ ε(v))dΩ for i in 1:6]

  ## Optimisation functionals
  Cᴴ(r,s,u,φ) = ∫((I ∘ φ)*(C ⊙ (ε(u[r])+εᴹ[r]) ⊙ εᴹ[s]))dΩ
  dCᴴ(r,s,q,u,φ) = ∫(-q*(C ⊙ (ε(u[r])+εᴹ[r]) ⊙ (ε(u[s])+εᴹ[s]))*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  κ(u,φ) = -1/9*(Cᴴ(1,1,u,φ)+Cᴴ(2,2,u,φ)+Cᴴ(3,3,u,φ)+
    2*(Cᴴ(1,2,u,φ)+Cᴴ(1,3,u,φ)+Cᴴ(2,3,u,φ)))
  dκ(q,u,φ) = -1/9*(dCᴴ(1,1,q,u,φ)+dCᴴ(2,2,q,u,φ)+dCᴴ(3,3,q,u,φ)+
    2*(dCᴴ(1,2,q,u,φ)+dCᴴ(1,3,q,u,φ)+dCᴴ(2,3,q,u,φ)))
  Vol(u,φ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ;
  dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  evo = FiniteDifferenceEvolver(FirstOrderStencil(3,Float64),model,V_φ;max_steps)
  reinit = FiniteDifferenceReinitialiser(FirstOrderStencil(3,Float64),model,V_φ;tol,γ_reinit)
  ls_evo = LevelSetEvolution(evo,reinit)

  ## Setup solver and FE operators
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  solver = ElasticitySolver(V)

  state_map = RepeatingAffineFEStateMap(
    6,a,l,U,V,V_φ,φh,dΩ;
    assem_U = SparseMatrixAssembler(Tm,Tv,U,V),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,V,U),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,V_φ,V_φ),
    ls = solver, adjoint_ls = solver
  )
  pcfs = PDEConstrainedFunctionals(κ,[Vol],state_map;analytic_dJ=dκ,analytic_dC=[dVol])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_Δ)
  a_hilb(p,q) = ∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(
    a_hilb,U_reg,V_reg,
    assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
    ls = PETScLinearSolver()
  )

  ## Optimiser
  optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;
    γ,verbose=i_am_main(ranks),constraint_names=[:Vol])
  for (it, uh, φh) in optimiser
    data = ["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh))]
    iszero(it % iter_mod) && writevtk(Ω,path*"out$it",cellfields=data)
    write_history(path*"/history.txt",optimiser.history;ranks=ranks)
  end
  it = get_history(optimiser).niter; uh = get_state(pcfs)
  writevtk(Ω,path*"out$it",cellfields=["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh))])
end

with_mpi() do distribute
  mesh_partition = (Px,Py,Pz)
  el_size = (elx,ely,elz)
  hilb_solver_options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true
    -ksp_converged_reason -ksp_rtol 1.0e-12"

  GridapPETSc.with(args=split(hilb_solver_options)) do
    main(mesh_partition,distribute,el_size,write_dir)
  end
end