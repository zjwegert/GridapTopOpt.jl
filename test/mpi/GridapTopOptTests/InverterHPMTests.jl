module InverterHPMTestsMPI
using Test

using Gridap, Gridap.MultiField, GridapDistributed, GridapSolvers,
  PartitionedArrays, GridapTopOpt, SparseMatricesCSR

"""
  (MPI) Inverter mechanism with Hilbertian projection method in 2D.

  Optimisation problem:
      Min J(Ω) = ηᵢₙ*∫ u⋅e₁ dΓᵢₙ/Vol(Γᵢₙ)
        Ω
    s.t., Vol(Ω) = vf,
            C(Ω) = 0,
          ⎡u∈V=H¹(Ω;u(Γ_D)=0)ᵈ,
          ⎣∫ C ⊙ ε(u) ⊙ ε(v) dΩ + ∫ kₛv⋅u dΓₒᵤₜ = ∫ v⋅g dΓᵢₙ , ∀v∈V.

    where C(Ω) = ∫ -u⋅e₁-δₓ dΓₒᵤₜ/Vol(Γₒᵤₜ). We assume symmetry in the problem to aid
     convergence.
"""
function main(distribute,mesh_partition)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))
  ## Parameters
  order = 1
  dom = (0,1,0,0.5)
  el_size = (20,20)
  γ = 0.1
  γ_reinit = 0.5
  max_steps = floor(Int,order*minimum(el_size)/10)
  tol = 1/(5order^2)/minimum(el_size)
  C = isotropic_elast_tensor(2,1.0,0.3)
  η_coeff = 2
  α_coeff = 4max_steps*γ
  vf = 0.4
  δₓ = 0.2
  ks = 0.1
  g = VectorValue(0.5,0)

  ## FE Setup
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size)
  el_Δ = get_el_Δ(model)
  f_Γ_in(x) = (x[1] ≈ 0.0) && (x[2] <= 0.03 + eps())
  f_Γ_out(x) = (x[1] ≈ 1.0) && (x[2] <= 0.07 + eps())
  f_Γ_D(x) = (x[1] ≈ 0.0) && (x[2] >= 0.4)
  f_Γ_D2(x) = (x[2] ≈ 0.0)
  update_labels!(1,model,f_Γ_in,"Gamma_in")
  update_labels!(2,model,f_Γ_out,"Gamma_out")
  update_labels!(3,model,f_Γ_D,"Gamma_D")
  update_labels!(4,model,f_Γ_D2,"SymLine")

  ## Triangulations and measures
  Ω = Triangulation(model)
  Γ_in = BoundaryTriangulation(model,tags="Gamma_in")
  Γ_out = BoundaryTriangulation(model,tags="Gamma_out")
  dΩ = Measure(Ω,2order)
  dΓ_in = Measure(Γ_in,2order)
  dΓ_out = Measure(Γ_out,2order)
  vol_D = sum(∫(1)dΩ)
  vol_Γ_in = sum(∫(1)dΓ_in)
  vol_Γ_out = sum(∫(1)dΓ_out)

  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;dirichlet_tags=["Gamma_D","SymLine"],
    dirichlet_masks=[(true,true),(false,true)])
  U = TrialFESpace(V,[VectorValue(0.0,0.0),VectorValue(0.0,0.0)])
  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_in","Gamma_out"])
  U_reg = TrialFESpace(V_reg,[0,0])

  ## Create FE functions
  lsf_fn(x) = min(max(initial_lsf(6,0.2)(x),-sqrt((x[1]-1)^2+(x[2]-0.5)^2)+0.2),
    sqrt((x[1])^2+(x[2]-0.5)^2)-0.1)
  φh = interpolate(lsf_fn,V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  a(u,v,φ) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(v)))dΩ + ∫(ks*(u⋅v))dΓ_out
  l(v,φ) = ∫(v⋅g)dΓ_in

  ## Optimisation functionals
  e₁ = VectorValue(1,0)
  J(u,φ) = ∫((u⋅e₁)/vol_Γ_in)dΓ_in
  Vol(u,φ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ
  dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  UΓ_out(u,φ) = ∫((u⋅-e₁-δₓ)/vol_Γ_out)dΓ_out

  ## Finite difference solver and level set function
  evo = FiniteDifferenceEvolver(FirstOrderStencil(2,Float64),model,V_φ;max_steps)
  reinit = FiniteDifferenceReinitialiser(FirstOrderStencil(2,Float64),model,V_φ;tol,γ_reinit)
  ls_evo = LevelSetEvolution(evo,reinit)

  ## Setup solver and FE operators
  state_map = AffineFEStateMap(a,l,U,V,V_φ)
  pcfs = PDEConstrainedFunctionals(J,[Vol,UΓ_out],state_map,analytic_dC=[dVol,nothing])

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_Δ)
  a_hilb(p,q) = ∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

  ## Optimiser
  optimiser = HilbertianProjection(pcfs,ls_evo,vel_ext,φh;
    γ,verbose=i_am_main(ranks),debug=true,constraint_names=[:Vol,:UΓ_out])

  # Do a few iterations
  vars, state = iterate(optimiser)
  vars, state = iterate(optimiser,state)
  true
end

# Test that these run successfully
with_mpi() do distribute
  @test main(distribute,(2,2))
end

end # module