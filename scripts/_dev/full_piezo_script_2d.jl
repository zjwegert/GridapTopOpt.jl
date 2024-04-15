using LevelSetTopOpt, Gridap

using Gridap, Gridap.TensorValues, Gridap.Geometry, Gridap.FESpaces, Gridap.Helpers, Gridap.Algebra
using GridapDistributed
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code
using PartitionedArrays
using SparseMatricesCSR
using ChainRulesCore

using GridapSolvers
using Gridap.MultiField

## Piezo helpers
function PZT5A(D::Int)
  ε_0 = 8.854e-12;
  if D == 3
    C_Voigt = [12.0400e10  7.52000e10  7.51000e10  0.0        0.0        0.0
          7.52000e10  12.0400e10  7.51000e10  0.0        0.0        0.0
          7.51000e10  7.51000e10  11.0900e10  0.0        0.0        0.0
          0.0         0.0         0.0          2.1000e10  0.0        0.0
          0.0         0.0         0.0          0.0        2.1000e10  0.0
          0.0         0.0         0.0          0.0        0.0       2.30e10]
    e_Voigt = [0.0       0.0       0.0        0.0       12.30000  0.0
                0.0       0.0       0.0        12.30000  0.0       0.0
                -5.40000  -5.40000   15.80000   0.0       0.0       0.0]
    K_Voigt = [540*ε_0     0        0
                  0      540*ε_0    0
                  0        0    830*ε_0]
  elseif D == 2
    C_Voigt = [12.0400e10  7.51000e10     0.0
                7.51000e10  11.0900e10     0.0
                  0.0         0.0      2.1000e10]
    e_Voigt = [0.0       0.0       12.30000
                -5.40000   15.80000   0.0]
    K_Voigt = [540*ε_0        0
                  0    830*ε_0]
  else
    @notimplemented
  end
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

## Parameters
el_size = (100,100)
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
path = dirname(dirname(@__DIR__))*"/results/2d_PZ_inverse_homenisation_ALM"

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
# mfs = BlockMultiFieldStyle()
UP = MultiFieldFESpace([U,P])#;style=mfs)
VQ = MultiFieldFESpace([V,Q])#;style=mfs)

V_φ = TestFESpace(model,reffe_scalar)
V_reg = TestFESpace(model,reffe_scalar)
U_reg = TrialFESpace(V_reg)

## Create FE functions
φh = interpolate(initial_lsf(4,0.2),V_φ)

## Interpolation and weak form
interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))#,ϵ=10^-9)
I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

## Material tensors
C, e, κ = PZT5A(2);
k0 = norm(C.data,Inf); 
α0 = norm(e.data,Inf); 
β0 = norm(κ.data,Inf);
γ = β0*k0/α0^2;

## Weak forms
εᴹ = (SymTensorValue(1.0,0.0,0.0),
      SymTensorValue(0.0,0.0,1.0),
      SymTensorValue(0.0,1/2,0))
Eⁱ = (VectorValue(1.0,0.0,),
      VectorValue(0.0,1.0))

a((u,ϕ),(v,q),φ,dΩ) = ∫((I ∘ φ) * (1/k0*((C ⊙ ε(u)) ⊙ ε(v)) - 
                                   1/α0*((-∇(ϕ) ⋅ e) ⊙ ε(v)) +
                                  -1/α0*((e ⋅² ε(u)) ⋅ -∇(q)) + 
                                  -γ/β0*((κ ⋅ -∇(ϕ)) ⋅ -∇(q))) )dΩ;

l_ε = [((v,q),φ,dΩ) -> ∫(((I ∘ φ) * (-C ⊙ εᴹ[i] ⊙ ε(v) + k0/α0*(e ⋅² εᴹ[i]) ⋅ -∇(q))))dΩ for i = 1:3];
l_E = [((v,q),φ,dΩ) -> ∫((I ∘ φ) * ((Eⁱ[i] ⋅ e ⊙ ε(v) + k0/α0*(κ ⋅ Eⁱ[i]) ⋅ -∇(q))))dΩ for i = 1:2];
l = [l_ε; l_E]

state_map = RepeatingAffineFEStateMap(5,a,l,UP,VQ,V_φ,U_reg,φh,dΩ)

x  = state_map(φh)
U  = LevelSetTopOpt.get_trial_space(state_map)
xh = FEFunction(U,x)
# u1,ϕ1,u2,ϕ2,u3,ϕ3,u4,ϕ4,u5,ϕ5 = xh


function Cᴴ(r,s,uϕ,φ,dΩ) # (normalised by 1/k0)
  u_s = uϕ[2s-1]; ϕ_s = uϕ[2s]
  ∫(1/k0 * (I ∘ φ) * (((C ⊙ (1/k0*ε(u_s) + εᴹ[s])) ⊙ εᴹ[r]) + ((1/α0*∇(ϕ_s) ⋅ e) ⊙ εᴹ[r])))dΩ;
end

function eᴴ(i,s,uϕ,φ,dΩ) # (normalised by 1/α0)
  u_s = uϕ[2s-1]; ϕ_s = uϕ[2s]
  ∫(1/α0 * (I ∘ φ) * (((e ⋅² (1/k0*ε(u_s) + εᴹ[s])) ⋅ Eⁱ[i]) + ((κ ⋅ (-1/α0*∇(ϕ_s))) ⋅ Eⁱ[i])))dΩ;
end

function κᴴ(i,j,uϕ,φ,dΩ) # (normalised by 1/β0)
  u_j = uϕ[6+2j-1]; ϕ_j = uϕ[6+2j]
  ∫(1/β0 * (I ∘ φ) * (((e ⋅² (1/k0*ε(u_j)) ⋅ Eⁱ[i]) + ((κ ⋅ (-1/α0*∇(ϕ_j) + Eⁱ[j])) ⋅ Eⁱ[i]))))dΩ;
end

function dᴴ(uϕ,φ,dΩ)
  _Cᴴ(r,s) = sum(Cᴴ(r,s,uϕ,φ,dΩ))
  _eᴴ(i,s) = sum(eᴴ(i,s,uϕ,φ,dΩ))
  (_Cᴴ(2,2)*_eᴴ(2,1) + _Cᴴ(1,1)*_eᴴ(2,2) - _Cᴴ(2,1)*(_eᴴ(2,1) + _eᴴ(2,2)))/
    (_Cᴴ(1,1)*_Cᴴ(2,2) - _Cᴴ(2,1)^2)
end

# map(rs->sum(Cᴴ(rs[1],rs[2],xh,φh,dΩ)), CartesianIndices((1:3, 1:3)))
# map(is->sum(eᴴ(is[1],is[2],xh,φh,dΩ)), CartesianIndices((1:2, 1:3)))
# map(ij->sum(κᴴ(ij[1],ij[2],xh,φh,dΩ)), CartesianIndices((1:2, 1:2)))
dᴴ(xh,φh,dΩ)#*α0/k0

C_mat = map(rs->sum(Cᴴ(rs[1],rs[2],xh,φh,dΩ)), CartesianIndices((1:3, 1:3)))
e_mat = map(is->sum(eᴴ(is[1],is[2],xh,φh,dΩ)), CartesianIndices((1:2, 1:3)))
d_mat = e_mat*inv(C_mat);
dh = d_mat[2,1] + d_mat[2,2]

nothing
# ## Optimisation functionals
# J((u, p),φ,dΩ) = ∫(0)dΩ
# dJ(q,(u, p),φ,dΩ) = ∫(0q)dΩ
# Vol((u, p),φ,dΩ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ;
# dVol(q,(u, p),φ,dΩ) = ∫(1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

# ## Finite difference solver and level set function
# stencil = AdvectionStencil(FirstOrderStencil(2,Float64),model,V_φ,tol,max_steps)

# ## Setup solver and FE operators
# state_map = RepeatingAffineFEStateMap(5,a,l,UP,VQ,V_φ,U_reg,φh,dΩ)
# pcfs = PDEConstrainedFunctionals(J,[Vol],state_map;analytic_dJ=dJ,analytic_dC=[dVol])

# evaluate!(pcfs,φh)

# ## Hilbertian extension-regularisation problems
# α = α_coeff*maximum(el_Δ)
# a_hilb(p,q) = ∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
# vel_ext = VelocityExtension(
#   a_hilb,U_reg,V_reg,
#   assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
#   ls = PETScLinearSolver()
# )

# ## Optimiser
# make_dir(path;ranks=ranks)
# optimiser = AugmentedLagrangian(pcfs,stencil,vel_ext,φh;γ,γ_reinit,verbose=i_am_main(ranks))
# for (it, uh, φh) in optimiser
#   write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi)|"=>(norm ∘ ∇(φh)),"uh"=>uh])
#   write_history(path*"/history.txt",optimiser.history;ranks=ranks)
# end
# it = optimiser.history.niter; uh = get_state(optimiser.problem)
# write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi)|"=>(norm ∘ ∇(φh)),"uh"=>uh];iter_mod=1)


function dh(stuff)
  C(M,N) = @notimplemented
  e(i,M) = @notimplemented
  dh = (C23^2*e31 - C22*C33*e31 + C13^2*e32 + C11*C23*e32 - C11*C33*e32 + 
        C12*C33*(e31 + e32) + C12^2*e33 + C11*(-C22 + C23)*e33 - C12*C23*(e31 + e33) + 
        C13*(-(C23*(e31 + e32)) + C22*(e31 + e33) - C12*(e32 + e33)))/(
        C13^2*C22 - 2*C12*C13*C23 + C12^2*C33 + C11*(C23^2 - C22*C33)) 
end
