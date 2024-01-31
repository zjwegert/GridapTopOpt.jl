using Gridap, Gridap.TensorValues, Gridap.Geometry, Gridap.FESpaces, Gridap.Helpers, Gridap.Algebra
using GridapDistributed
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code
using PartitionedArrays
using SparseMatricesCSR
using ChainRulesCore
using LSTO_Distributed

using GridapSolvers
using Gridap.MultiField

function main(mesh_partition,distribute,el_size)
  ranks = distribute(LinearIndices((prod(mesh_partition),)))

  ## Parameters
  order = 1
  xmax,ymax,zmax=(1.0,1.0,1.0)
  dom = (0,xmax,0,ymax,0,zmax)
  γ = 0.05
  γ_reinit = 0.5
  max_steps = floor(Int,minimum(el_size)/3)
  tol = 1/(2order^2)*prod(inv,minimum(el_size))
  C = isotropic_3d(1.,0.3)
  η_coeff = 2
  α_coeff = 4
  vf = 0.5
  path = dirname(dirname(@__DIR__))*"/results/MPI_main_3d_PZ_inverse_homenisation_ALM"

  ## FE Setup
  model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size,isperiodic=(true,true,true))
  Δ = get_Δ(model)
  f_Γ_D(x) = iszero(x)
  update_labels!(1,model,f_Γ_D,"origin")

  ## Triangulations and measures
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2*order)
  vol_D = sum(∫(1)dΩ)

  ## Spaces
  reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["origin"])
  U = TrialFESpace(V,VectorValue(0.0,0.0,0.0))
  Q = TestFESpace(model,reffe_scalar;conformity=:H1,dirichlet_tags=["origin"])
  P = TrialFESpace(Q,0)

  mfs = BlockMultiFieldStyle()
  UP = MultiFieldFESpace([U,P];style=mfs)
  VQ = MultiFieldFESpace([V,Q];style=mfs)

  ## Create FE functions
  lsf_fn(x) cos(2π*x[1]) + cos(2π*x[2]) + cos(2π*x[3])
  φh = interpolate(lsf_fn,V_φ)

  ## Interpolation and weak form
  interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ))
  I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

  ## Material tensors
  C, e, κ = PZT5A(3);
  k0 = norm(C.data,Inf); 
  α0 = norm(e.data,Inf); 
  β0 = norm(κ.data,Inf);
  γ = β0*k0/α0^2;

  ## Weak forms
  εᴹ = (TensorValue(1.,0.,0.,0.,0.,0.,0.,0.,0.),           # ϵᵢⱼ⁽¹¹⁾≡ϵᵢⱼ⁽¹⁾
        TensorValue(0.,0.,0.,0.,1.,0.,0.,0.,0.),           # ϵᵢⱼ⁽²²⁾≡ϵᵢⱼ⁽²⁾
        TensorValue(0.,0.,0.,0.,0.,0.,0.,0.,1.),           # ϵᵢⱼ⁽³³⁾≡ϵᵢⱼ⁽³⁾
        TensorValue(0.,0.,0.,0.,0.,1/2,0.,1/2,0.),         # ϵᵢⱼ⁽²³⁾≡ϵᵢⱼ⁽⁴⁾
        TensorValue(0.,0.,1/2,0.,0.,0.,1/2,0.,0.),         # ϵᵢⱼ⁽¹³⁾≡ϵᵢⱼ⁽⁵⁾
        TensorValue(0.,1/2,0.,1/2,0.,0.,0.,0.,0.))         # ϵᵢⱼ⁽¹²⁾≡ϵᵢⱼ⁽⁶⁾
  Eⁱ = (VectorValue(1.,0.,0.),
        VectorValue(0.,1.,0),
        VectorValue(0.,0.,1.))

  a((u, ϕ),(v, q),φ,dΩ) = ∫((I ∘ φ) * (1/k0*((C ⊙ ε(u)) ⊙ ε(v)) - 
                                       1/α0*((-∇(ϕ) ⋅ e) ⊙ ε(v)) +
                                      -1/α0*((e ⋅² ε(u)) ⋅ -∇(q)) + 
                                      -γ/β0*((κ ⋅ -∇(ϕ)) ⋅ -∇(q))) )dΩ;

  l_ε = [(v, q) -> ∫((I ∘ φ) * (-C ⊙ εᴹ[i] ⊙ ε(v) + k0/α0*(e ⋅² εᴹ[i]) ⋅ -∇(q)))dΩ for i = 1:6];
  l_E = [(v, q) -> ∫(I ∘ φ) * ((Eⁱ[i] ⋅ e ⊙ ε(v) + k0/α0*(κ ⋅ Eⁱ[i]) ⋅ -∇(q)))dΩ for i = 1:3];
  l = [l_ε; l_E]

  ## Optimisation functionals
  J(u,φ,dΩ) = @notimplemented # ∫(-(I ∘ φ)*_K(C,u,εᴹ))dΩ
  dJ(q,u,φ,dΩ) = @notimplemented # ∫(-_v_K(C,u,εᴹ)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
  Vol(u,φ,dΩ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ;
  dVol(q,u,φ,dΩ) = ∫(1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

  ## Finite difference solver and level set function
  stencil = AdvectionStencil(FirstOrderStencil(3,Float64),model,V_φ,tol,max_steps)
  reinit!(stencil,φh,γ_reinit)

  ## Setup solver and FE operators
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  Tv = Vector{PetscScalar}
  solver_u = ElasticitySolver(V)
  solver_ϕ = PETScLinearSolver()
  P = BlockDiagonalPreconditioner([solver_u,solver_ϕ])
  solver = GridapSolvers.LinearSolvers.GMRESSolver(100;Pr=P,rtol=1.e-8,verbose=i_am_main(ranks))


  state_map = RepeatingAffineFEStateMap(
    9,a,l,UP,VQ,V_φ,U_reg,φh,dΩ;
    assem_U = SparseMatrixAssembler(Tm,Tv,UP,VQ),
    assem_adjoint = SparseMatrixAssembler(Tm,Tv,VQ,UP),
    assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
    ls = solver, adjoint_ls = solver
  )
  pcfs = PDEConstrainedFunctionals(J,[Vol],state_map;analytic_dJ=dJ,analytic_dC=[dVol])

  J, C, dJ, dC = Gridap.evaluate!(pcfs,φh)

  # ## Hilbertian extension-regularisation problems
  # α = α_coeff*maximum(Δ)
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
  #   write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh"=>uh])
  #   write_history(path*"/history.txt",optimiser.history;ranks=ranks)
  # end
  # it = optimiser.history.niter; uh = get_state(optimiser.problem)
  # write_vtk(Ω,path*"/struc_$it",it,["phi"=>φh,"H(phi)"=>(H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh"=>uh];iter_mod=1)
end;

with_mpi() do distribute
  main((2,2,2),distribute,(25,25,25))
end;

function dh(stuff)
  C(M,N) = @notimplemented
  e(i,M) = @notimplemented
  dh = (C23^2*e31 - C22*C33*e31 + C13^2*e32 + C11*C23*e32 - C11*C33*e32 + 
        C12*C33*(e31 + e32) + C12^2*e33 + C11*(-C22 + C23)*e33 - C12*C23*(e31 + e33) + 
        C13*(-(C23*(e31 + e32)) + C22*(e31 + e33) - C12*(e32 + e33)))/(
        C13^2*C22 - 2*C12*C13*C23 + C12^2*C33 + C11*(C23^2 - C22*C33)) 
end

## TESTING
model = CartesianDiscreteModel((0,1,0,1),(10,10));
Ω = Triangulation(model)
dΩ = Measure(Ω,2)
V = TestFESpace(model,ReferenceFE(lagrangian,Float64,1))
uh = interpolate(x->x[1]^2*x[2]^2,V);
Cmock = [i*j*∫(uh)dΩ for i = 1:3, j =1:3]

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