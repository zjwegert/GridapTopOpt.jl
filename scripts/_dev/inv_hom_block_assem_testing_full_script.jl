using Gridap, GridapDistributed, GridapPETSc, PartitionedArrays, LevelSetTopOpt
using GridapSolvers: LinearSolvers
using Gridap.MultiField: BlockMultiFieldStyle

"""
  (Serial) Maximum bulk modulus inverse homogenisation with augmented Lagrangian method in 2D.

  Optimisation problem:
      Min J(Ω) = -κ(Ω)
        Ω
    s.t., Vol(Ω) = Vf,
          ⎡For unique εᴹᵢ, find uᵢ∈V=H¹ₚₑᵣ(Ω)ᵈ, 
          ⎣∫ ∑ᵢ C ⊙ ε(uᵢ) ⊙ ε(vᵢ) dΩ = ∫ -∑ᵢ C ⊙ ε⁰ᵢ ⊙ ε(vᵢ) dΩ, ∀v∈V.
""" 
function main(diag_block)
  ## Parameters
  order = 2;
  xmax,ymax=(1.0,1.0)
  dom = (0,xmax,0,ymax);
  el_size = (200,200);
  γ = 0.05;
  γ_reinit = 0.5;
  max_steps = floor(Int,minimum(el_size)/10)
  tol = 1/(order^2*10)/minimum(el_size)
  C = isotropic_elast_tensor(2,1.,0.3);
  η_coeff = 2;
  α_coeff = 4;
  path = dirname(dirname(@__DIR__))*"/results/main_inverse_homogenisation_ALM"

  ## FE Setup
  model = CartesianDiscreteModel(dom,el_size,isperiodic=(true,true));
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
  _V = TestFESpace(model,reffe;dirichlet_tags=["origin"])
  _U = TrialFESpace(_V,VectorValue(0.0,0.0))
  mfs = BlockMultiFieldStyle()
  U = MultiFieldFESpace([_U,_U,_U];style=mfs);
  V = MultiFieldFESpace([_V,_V,_V];style=mfs);
  V_reg = V_φ = TestFESpace(model,reffe_scalar)
  U_reg = TrialFESpace(V_reg)

  ## Create FE functions
  lsf_fn = x->max(initial_lsf(2,0.4)(x),initial_lsf(2,0.4;b=VectorValue(0,0.5))(x));
  φh = interpolate(lsf_fn,V_φ);
  φ = get_free_dof_values(φh)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  εᴹ = (TensorValue(1.,0.,0.,0.),     # ϵᵢⱼ⁽¹¹⁾≡ϵᵢⱼ⁽¹⁾
        TensorValue(0.,0.,0.,1.),     # ϵᵢⱼ⁽²²⁾≡ϵᵢⱼ⁽²⁾
        TensorValue(0.,1/2,1/2,0.))   # ϵᵢⱼ⁽¹²⁾≡ϵᵢⱼ⁽³⁾

  a(u,v,φ,dΩ) = ∫((I ∘ φ)*sum(C ⊙ ε(u[i]) ⊙ ε(v[i]) for i = 1:length(u)))dΩ
  l(v,φ,dΩ) = ∫(-(I ∘ φ)*sum(C ⊙ εᴹ[i] ⊙ ε(v[i]) for i = 1:length(v)))dΩ;
  res(u,v,φ,dΩ) = a(u,v,φ,dΩ) - l(v,φ,dΩ)

  ## Optimisation functionals
  _C(C,ε_p,ε_q) = C ⊙ ε_p ⊙ ε_q;
  _K(C,(u1,u2,u3),εᴹ) = (_C(C,ε(u1)+εᴹ[1],εᴹ[1]) + _C(C,ε(u2)+εᴹ[2],εᴹ[2]) + 2*_C(C,ε(u1)+εᴹ[1],εᴹ[2]))/4
  _v_K(C,(u1,u2,u3),εᴹ) = (_C(C,ε(u1)+εᴹ[1],ε(u1)+εᴹ[1]) + _C(C,ε(u2)+εᴹ[2],ε(u2)+εᴹ[2]) + 2*_C(C,ε(u1)+εᴹ[1],ε(u2)+εᴹ[2]))/4    

  J = (u,φ,dΩ) -> ∫(-(I ∘ φ)*_K(C,u,εᴹ))dΩ
  dJ = (q,u,φ,dΩ) -> ∫(-_v_K(C,u,εᴹ)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;
  Vol = (u,φ,dΩ) -> ∫(((ρ ∘ φ) - 0.5)/vol_D)dΩ;
  dVol = (q,u,φ,dΩ) -> ∫(1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  stencil = AdvectionStencil(FirstOrderStencil(2,Float64),model,V_φ,el_size./order,max_steps,tol)
  reinit!(stencil,φ,γ_reinit)

  ## Setup solver and FE operators
  P = LinearSolvers.JacobiLinearSolver()
  pcg = LinearSolvers.CGSolver(P;rtol=10e-12,verbose=true)

  assem = if diag_block
    DiagonalBlockMatrixAssembler(SparseMatrixAssembler(U,V)) 
  else
    SparseMatrixAssembler(U,V)
  end

  state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh,dΩ;
    ls = pcg,
    adjoint_ls = pcg,
    assem_U=assem,
    assem_adjoint=assem)
  pcfs = PDEConstrainedFunctionals(J,[Vol],state_map;analytic_dJ=dJ,analytic_dC=[dVol])

  J_init,C_init,dJ,dC = Gridap.evaluate!(pcfs,φ)
  u_vec = pcfs.state_map.fwd_caches[4]

  return pcfs,[J_init,C_init,dJ,dC],u_vec

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_Δ)
  a_hilb(p,q) =∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)
  
  ## Optimiser
  make_dir(path)
  optimiser = AugmentedLagrangian(φ,pcfs,stencil,vel_ext,interp,el_size,γ,γ_reinit);
  for history in optimiser
    it,Ji,Ci,Li = last(history)
    λi = optimiser.λ; Λi = optimiser.Λ
    print_history(it,["J"=>Ji,"C"=>Ci,"L"=>Li,"λ"=>λi,"Λ"=>Λi])
    write_history(history,path*"/history.csv")
    data = ["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh))]
    iszero(it % 10) && writevtk(Ω,path*"out$it",cellfields=data)
  end
  it,Ji,Ci,Li = last(optimiser.history)
  λi = optimiser.λ; Λi = optimiser.Λ
  print_history(it,["J"=>Ji,"C"=>Ci,"L"=>Li,"λ"=>λi,"Λ"=>Λi])
  write_history(optimiser.history,path*"/history.csv")
  writevtk(Ω,path*"/out$it",cellfields=["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi)|"=>(norm ∘ ∇(φh))])
end

using LinearAlgebra
using BlockArrays
function LinearAlgebra.diag(A::BlockArray)
  @assert ndims(A) == 2 "diag only supports matrices currently"
  _blk_axes = blockaxes(A);
  blkrng = (_blk_axes[1]<_blk_axes[2]) ? _blk_axes[1] : _blk_axes[2] 

  mortar(map(I->diag(A[I,I]),blkrng))
end
function Gridap.Algebra.numerical_setup!(ns::LinearSolvers.JacobiNumericalSetup, A::AbstractMatrix)
  ns.inv_diag .= 1.0 ./ diag(A)
end

# The above is a crappy solver for this problem, use AMG instead (see MPI version)

_PSFs,_OBJ_VALS,_U = main(false);
_PSFs_diag,_OBJ_VALS_diag,_U_diag = main(true);



_without = Base.summarysize(_PSFs.state_map.fwd_caches[2])
_with = Base.summarysize(_PSFs_diag.state_map.fwd_caches[2])

_with/_without