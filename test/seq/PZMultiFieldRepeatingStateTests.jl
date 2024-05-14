module PZMultiFieldRepeatingStateTests
using Test

using Gridap, GridapDistributed, GridapPETSc, GridapSolvers,
  PartitionedArrays, GridapTopOpt, SparseMatricesCSR

using Gridap.TensorValues, Gridap.Helpers, Gridap.MultiField

## Parameters
function main(;AD,use_mfs)
  el_size = (20,20)
  order = 1
  xmax,ymax=(1.0,1.0)
  dom = (0,xmax,0,ymax)
  γ = 0.1
  γ_reinit = 0.5
  max_steps = floor(Int,order*minimum(el_size)/10)
  tol = 1/(5order^2)/minimum(el_size)
  η_coeff = 2
  α_coeff = 4*max_steps*γ
  vf = 0.5

  ## FE Setup
  model = CartesianDiscreteModel(dom,el_size,isperiodic=(true,true))
  el_Δ = get_el_Δ(model)
  f_Γ_D(x) = iszero(x)
  update_labels!(1,model,f_Γ_D,"origin")

  ## Triangulations and measures
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*order)
  vol_D = sum(∫(1)dΩ)

  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["origin"])
  U = TrialFESpace(V,VectorValue(0.0,0.0))
  Q = TestFESpace(model,reffe_scalar;conformity=:H1,dirichlet_tags=["origin"])
  P = TrialFESpace(Q,0)
  if use_mfs
    mfs = BlockMultiFieldStyle()
    UP = MultiFieldFESpace([U,P];style=mfs)
    VQ = MultiFieldFESpace([V,Q];style=mfs)
  else
    UP = MultiFieldFESpace([U,P])
    VQ = MultiFieldFESpace([V,Q])
  end

  V_φ = TestFESpace(model,reffe_scalar)
  V_reg = TestFESpace(model,reffe_scalar)
  U_reg = TrialFESpace(V_reg)

  ## Create FE functions
  φh = interpolate(initial_lsf(2,0.2),V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ),ϵ=10^-9)
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  ## Material tensors
  C, e, κ = PZT5A_2D();
  k0 = norm(C.data,Inf);
  α0 = norm(e.data,Inf);
  β0 = norm(κ.data,Inf);
  γ0 = β0*k0/α0^2;

  ## Weak forms
  εᴹ = (SymTensorValue(1.0,0.0,0.0),
        SymTensorValue(0.0,0.0,1.0),
        SymTensorValue(0.0,1/2,0))
  Eⁱ = (VectorValue(1.0,0.0,),
        VectorValue(0.0,1.0))

  a((u,ϕ),(v,q),φ,dΩ) = ∫((I ∘ φ) * (1/k0*((C ⊙ ε(u)) ⊙ ε(v)) -
                                    1/α0*((-∇(ϕ) ⋅ e) ⊙ ε(v)) +
                                    -1/α0*((e ⋅² ε(u)) ⋅ -∇(q)) +
                                    -γ0/β0*((κ ⋅ -∇(ϕ)) ⋅ -∇(q))) )dΩ;

  l_ε = [((v,q),φ,dΩ) -> ∫(((I ∘ φ) * (-C ⊙ εᴹ[i] ⊙ ε(v) + k0/α0*(e ⋅² εᴹ[i]) ⋅ -∇(q))))dΩ for i = 1:3];
  l_E = [((v,q),φ,dΩ) -> ∫((I ∘ φ) * ((Eⁱ[i] ⋅ e ⊙ ε(v) + k0/α0*(κ ⋅ Eⁱ[i]) ⋅ -∇(q))))dΩ for i = 1:2];
  l = [l_ε; l_E]

  function Cᴴ(r,s,uϕ,φ,dΩ)
    u_s = uϕ[2s-1]; ϕ_s = uϕ[2s]
    ∫(1/k0 * (I ∘ φ) * (((C ⊙ (1/k0*ε(u_s) + εᴹ[s])) ⊙ εᴹ[r]) - ((-1/α0*∇(ϕ_s) ⋅ e) ⊙ εᴹ[r])))dΩ;
  end

  function DCᴴ(r,s,q,uϕ,φ,dΩ)
    u_r = uϕ[2r-1]; ϕ_r = uϕ[2r]
    u_s = uϕ[2s-1]; ϕ_s = uϕ[2s]
    ∫(- 1/k0 * q * (
      (C ⊙ (1/k0*ε(u_s) + εᴹ[s])) ⊙ (1/k0*ε(u_r) + εᴹ[r]) -
      (-1/α0*∇(ϕ_s) ⋅ e) ⊙ (1/k0*ε(u_r) + εᴹ[r]) -
      (e ⋅² (1/k0*ε(u_s) + εᴹ[s])) ⋅ (-1/α0*∇(ϕ_r)) -
      (κ ⋅ (-1/α0*∇(ϕ_s))) ⋅ (-1/α0*∇(ϕ_r))
      ) * (DH ∘ φ) * (norm ∘ ∇(φ))
    )dΩ;
  end

  Bᴴ(uϕ,φ,dΩ) = 1/4*(Cᴴ(1,1,uϕ,φ,dΩ)+Cᴴ(2,2,uϕ,φ,dΩ)+2*Cᴴ(1,2,uϕ,φ,dΩ))
  DBᴴ(q,uϕ,φ,dΩ) = 1/4*(DCᴴ(1,1,q,uϕ,φ,dΩ)+DCᴴ(2,2,q,uϕ,φ,dΩ)+2*DCᴴ(1,2,q,uϕ,φ,dΩ))

  J(uϕ,φ,dΩ) = -1*Bᴴ(uϕ,φ,dΩ)
  DJ(q,uϕ,φ,dΩ) = -1*DBᴴ(q,uϕ,φ,dΩ)
  C1(uϕ,φ,dΩ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ;
  DC1(q,uϕ,φ,dΩ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  stencil = HamiltonJacobiEvolution(FirstOrderStencil(2,Float64),model,V_φ,tol,max_steps)

  ## Setup solver and FE operators
  state_map = RepeatingAffineFEStateMap(5,a,l,UP,VQ,V_φ,U_reg,φh,dΩ)
  pcfs = if AD
    PDEConstrainedFunctionals(J,[C1],state_map;analytic_dJ=DJ,analytic_dC=[DC1])
  else
    PDEConstrainedFunctionals(J,[C1],state_map)
  end

  ## Hilbertian extension-regularisation problems
  α = α_coeff*maximum(el_Δ)
  a_hilb(p,q) = ∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
  vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

  # ## Optimiser
  optimiser = HilbertianProjection(pcfs,stencil,vel_ext,φh;γ,γ_reinit,verbose=true)
  vars, state = iterate(optimiser)
  vars, state = iterate(optimiser,state)
  true
end

function PZT5A_2D()
  ε_0 = 8.854e-12;
  C_Voigt = [12.0400e10  7.51000e10     0.0
              7.51000e10  11.0900e10     0.0
                0.0         0.0      2.1000e10]
  e_Voigt = [0.0       0.0       12.30000
              -5.40000   15.80000   0.0]
  K_Voigt = [540*ε_0        0
                0    830*ε_0]
  C = voigt2tensor4(C_Voigt)
  e = voigt2tensor3(e_Voigt)
  κ = voigt2tensor2(K_Voigt)
  C,e,κ
end

"""
  Given a material constant given in Voigt notation,
  return a SymFourthOrderTensorValue using ordering from Gridap
"""
function voigt2tensor4(A::Array{M,2}) where M
  if isequal(size(A),(3,3))
    return SymFourthOrderTensorValue(A[1,1], A[3,1], A[2,1],
                                     A[1,3], A[3,3], A[2,3],
                                     A[1,2], A[3,2], A[2,2])
  elseif isequal(size(A),(6,6))
    return SymFourthOrderTensorValue(A[1,1], A[6,1], A[5,1], A[2,1], A[4,1], A[3,1],
                                     A[1,6], A[6,6], A[5,6], A[2,6], A[4,6], A[3,6],
                                     A[1,5], A[6,5], A[5,5], A[2,5], A[4,5], A[3,5],
                                     A[1,2], A[6,2], A[5,2], A[2,2], A[4,2], A[3,2],
                                     A[1,4], A[6,4], A[5,4], A[2,4], A[4,4], A[3,4],
                                     A[1,3], A[6,3], A[5,3], A[2,3], A[4,3], A[3,3])
  else
      @notimplemented
  end
end

"""
  Given a material constant given in Voigt notation,
  return a ThirdOrderTensorValue using ordering from Gridap
"""
function voigt2tensor3(A::Array{M,2}) where M
  if isequal(size(A),(2,3))
    return ThirdOrderTensorValue(A[1,1], A[2,1], A[1,3], A[2,3], A[1,3], A[2,3], A[1,2], A[2,2])
  elseif isequal(size(A),(3,6))
    return ThirdOrderTensorValue(
      A[1,1], A[2,1], A[3,1], A[1,6], A[2,6], A[3,6], A[1,5], A[2,5], A[3,5],
      A[1,6], A[2,6], A[3,6], A[1,2], A[2,2], A[3,2], A[1,4], A[2,4], A[3,4],
      A[1,5], A[2,5], A[3,5], A[1,4], A[2,4], A[3,4], A[1,3], A[2,3], A[3,3])
  else
    @notimplemented
  end
end

"""
  Given a material constant given in Voigt notation,
  return a SymTensorValue using ordering from Gridap
"""
function voigt2tensor2(A::Array{M,2}) where M
  return TensorValue(A)
end

# Test that these run successfully
@test main(;AD=true)
@test main(;AD=false)
@test main(;AD=true,use_mfs=true)
@test main(;AD=false,use_mfs=true)

end # module