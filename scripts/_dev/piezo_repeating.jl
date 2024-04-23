using Gridap, GridapDistributed, GridapPETSc, GridapSolvers, 
  PartitionedArrays, LevelSetTopOpt, SparseMatricesCSR
using Gridap.TensorValues, Gridap.Helpers, Gridap.MultiField
using GridapSolvers.BlockSolvers

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
#function main(mesh_partition,distribute)
  #ranks = distribute(LinearIndices((prod(mesh_partition),)))

mesh_partition = (1,1)
ranks = with_mpi() do distribute
  distribute(LinearIndices((prod(mesh_partition),)))
end

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
path = dirname(@__DIR__)*"/results/2d_PZ_inverse_homenisation_MPI_Nx$(mesh_partition[1])Ny$(mesh_partition[2])/"
#i_am_main(ranks) && mkpath(path)

## FE Setup
model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size,isperiodic=(true,true))
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
mfs = BlockMultiFieldStyle()
UP = MultiFieldFESpace([U,P];style=mfs)
VQ = MultiFieldFESpace([V,Q];style=mfs)

V_φ = TestFESpace(model,reffe_scalar)
V_reg = TestFESpace(model,reffe_scalar)
U_reg = TrialFESpace(V_reg)

## Create FE functions
lsf_fn = initial_lsf(2,0.2)
φh = interpolate(lsf_fn,V_φ)

## Interpolation and weak form
interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ),ϵ=10^-9)
I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

## Material tensors
C, e, κ = PZT5A(2);
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

## Optimisation functionals
# Note, uϕ unpacks as -> uϕ = u1,ϕ1,u2,ϕ2,u3,ϕ3,u4,ϕ4,u5,ϕ5
#  where entries 1,2,3 correspond to εᴹ[1],εᴹ[2],εᴹ[3] and 
#  4,5 correspond to Eⁱ[1],Eⁱ[2].

function Cᴴ(r,s,uϕ,φ,dΩ) # (normalised by 1/k0)
  u_s = uϕ[2s-1]; ϕ_s = uϕ[2s]
  ∫(1/k0 * (I ∘ φ) * (((C ⊙ (1/k0*ε(u_s) + εᴹ[s])) ⊙ εᴹ[r]) - ((-1/α0*∇(ϕ_s) ⋅ e) ⊙ εᴹ[r])))dΩ;
end

function DCᴴ(r,s,q,uϕ,φ,dΩ) # (normalised by 1/k0)
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

function eᴴ(i,s,uϕ,φ,dΩ) # (normalised by 1/α0)
  u_s = uϕ[2s-1]; ϕ_s = uϕ[2s]
  ∫(1/α0 * (I ∘ φ) * (((e ⋅² (1/k0*ε(u_s) + εᴹ[s])) ⋅ Eⁱ[i]) + ((κ ⋅ (-1/α0*∇(ϕ_s))) ⋅ Eⁱ[i])))dΩ;
end

function Deᴴ(i,s,q,uϕ,φ,dΩ) # (normalised by 1/α0)
  u_i = uϕ[6+2i-1]; ϕ_i = uϕ[6+2i]
  u_s = uϕ[2s-1]; ϕ_s = uϕ[2s]
  ∫(- 1/α0 * q * (
    - (C ⊙ (1/k0*ε(u_s) + εᴹ[s])) ⊙ (1/k0*ε(u_i)) +
    (-1/α0*∇(ϕ_s) ⋅ e) ⊙ (1/k0*ε(u_i)) +
    (e ⋅² (1/k0*ε(u_s) + εᴹ[s])) ⋅ (-1/α0*∇(ϕ_i) + Eⁱ[i]) + 
    (κ ⋅ (-1/α0*∇(ϕ_s))) ⋅ (-1/α0*∇(ϕ_i) + Eⁱ[i])
    ) * (DH ∘ φ) * (norm ∘ ∇(φ))
  )dΩ;
end

function κᴴ(i,j,uϕ,φ,dΩ) # (normalised by 1/β0)
  u_j = uϕ[6+2j-1]; ϕ_j = uϕ[6+2j]
  ∫(1/β0 * (I ∘ φ) * (((e ⋅² (1/k0*ε(u_j)) ⋅ Eⁱ[i]) + ((κ ⋅ (-1/α0*∇(ϕ_j) + Eⁱ[j])) ⋅ Eⁱ[i]))))dΩ;
end

function Dκᴴ(i,j,q,uϕ,φ,dΩ) # (normalised by 1/β0)
  u_i = uϕ[6+2i-1]; ϕ_i = uϕ[6+2i]
  u_j = uϕ[6+2j-1]; ϕ_j = uϕ[6+2j]
  ∫(- 1/β0 * q * (
    - (C ⊙ (1/k0*ε(u_j))) ⊙ (1/k0*ε(u_i)) +
    ((-1/α0*∇(ϕ_j) + Eⁱ[j]) ⋅ e) ⊙ (1/k0*ε(u_i)) +
    (e ⋅² (1/k0*ε(u_j))) ⋅ (-1/α0*∇(ϕ_i) + Eⁱ[i]) + 
    (κ ⋅ (-1/α0*∇(ϕ_j) + Eⁱ[j])) ⋅ (-1/α0*∇(ϕ_i) + Eⁱ[i])
    ) * (DH ∘ φ) * (norm ∘ ∇(φ))
  )dΩ;
end

# Equiv to
#   C_mat = map(rs->sum(Cᴴ(rs[1],rs[2],xh,φh,dΩ)), CartesianIndices((1:3, 1:3)))
#   e_mat = map(is->sum(eᴴ(is[1],is[2],xh,φh,dΩ)), CartesianIndices((1:2, 1:3)))
#   d_mat = e_mat*inv(C_mat);
#   dh = d_mat[2,1] + d_mat[2,2]
function dᴴ(uϕ,φ,dΩ)
  # We leave eᴴ_21 and eᴴ_22 as DomainContribution so that result is a DomainContribution.
  #  Note that this means AD applied to this functional would be incorrect.
  Cᴴ_11 = sum(Cᴴ(1,1,uϕ,φ,dΩ)); 
  Cᴴ_22 = sum(Cᴴ(2,2,uϕ,φ,dΩ)); 
  Cᴴ_21 = sum(Cᴴ(2,1,uϕ,φ,dΩ));
  eᴴ_21 = eᴴ(2,1,uϕ,φ,dΩ); 
  eᴴ_22 = eᴴ(2,2,uϕ,φ,dΩ);
  (Cᴴ_22*eᴴ_21 + Cᴴ_11*eᴴ_22 - Cᴴ_21*(eᴴ_21 + eᴴ_22))*(1/(Cᴴ_11*Cᴴ_22 - Cᴴ_21^2))
end

function Ddᴴ(q,uϕ,φ,dΩ)
  Cᴴ_11 = sum(Cᴴ(1,1,uϕ,φ,dΩ)); DCᴴ_11 = DCᴴ(1,1,q,uϕ,φ,dΩ)
  Cᴴ_22 = sum(Cᴴ(2,2,uϕ,φ,dΩ)); DCᴴ_22 = DCᴴ(2,2,q,uϕ,φ,dΩ)
  Cᴴ_21 = sum(Cᴴ(2,1,uϕ,φ,dΩ)); DCᴴ_21 = DCᴴ(2,1,q,uϕ,φ,dΩ)
  eᴴ_21 = sum(eᴴ(2,1,uϕ,φ,dΩ)); Deᴴ_21 = DCᴴ(2,1,q,uϕ,φ,dΩ) 
  eᴴ_22 = sum(eᴴ(2,2,uϕ,φ,dΩ)); Deᴴ_22 = DCᴴ(2,2,q,uϕ,φ,dΩ)
  (-1*Cᴴ_11^2*eᴴ_22*(DCᴴ_22)+Cᴴ_22^2*(-eᴴ_21*(DCᴴ_11)+Cᴴ_11*(Deᴴ_21))+
    Cᴴ_21^3*((Deᴴ_21)+(Deᴴ_22))+Cᴴ_11*Cᴴ_22*(-1*((eᴴ_21+eᴴ_22)*(DCᴴ_21))+
    Cᴴ_11*(Deᴴ_22))-Cᴴ_21^2*(eᴴ_22*((DCᴴ_11)+(DCᴴ_21))+eᴴ_21*((DCᴴ_21)+(DCᴴ_22))+
    Cᴴ_22*(Deᴴ_21)+Cᴴ_11*(Deᴴ_22))+Cᴴ_21*(Cᴴ_11*(2*eᴴ_22*(DCᴴ_21)+(eᴴ_21+eᴴ_22)*(DCᴴ_22))+
    Cᴴ_22*((eᴴ_21+eᴴ_22)*(DCᴴ_11)+2*eᴴ_21*(DCᴴ_21)-Cᴴ_11*((Deᴴ_21)+(Deᴴ_22)))))*(1/(Cᴴ_21^2-Cᴴ_11*Cᴴ_22)^2)
end

Bᴴ(uϕ,φ,dΩ) = 1/4*(Cᴴ(1,1,uϕ,φ,dΩ)+Cᴴ(2,2,uϕ,φ,dΩ)+2*Cᴴ(1,2,uϕ,φ,dΩ))
DBᴴ(q,uϕ,φ,dΩ) = 1/4*(DCᴴ(1,1,q,uϕ,φ,dΩ)+DCᴴ(2,2,q,uϕ,φ,dΩ)+2*DCᴴ(1,2,q,uϕ,φ,dΩ))


J(uϕ,φ,dΩ) = -1*dᴴ(uϕ,φ,dΩ)
DJ(q,uϕ,φ,dΩ) = -1*Ddᴴ(q,uϕ,φ,dΩ)
C1(uϕ,φ,dΩ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ;
DC1(q,uϕ,φ,dΩ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ
C2(uϕ,φ,dΩ) = Cᴴ(2,2,uϕ,φ,dΩ) - ∫(0.05/vol_D)dΩ
DC2(q,uϕ,φ,dΩ) = DCᴴ(2,2,q,uϕ,φ,dΩ)

# ## Finite difference solver and level set function
stencil = HamiltonJacobiEvolution(FirstOrderStencil(2,Float64),model,V_φ,tol,max_steps)

# ## Setup solver and FE operators
Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
Tv = Vector{PetscScalar}
solver_u = LUSolver() #ElasticitySolver(V) #GridapSolvers.LinearSolvers.ElasticitySolver(V)
solver_ϕ = LUSolver() #PETScLinearSolver()

# P = GridapSolvers.BlockDiagonalSolver([solver_u,solver_ϕ])
# solver = GridapSolvers.LinearSolvers.GMRESSolver(100;Pr=P,rtol=1.e-8,verbose=i_am_main(ranks))
solver = BlockTriangularSolver([solver_u,solver_ϕ])

state_map = RepeatingAffineFEStateMap(5,a,l,UP,VQ,V_φ,U_reg,φh,dΩ;
  assem_U = SparseMatrixAssembler(Tm,Tv,UP,VQ),
  assem_adjoint = SparseMatrixAssembler(Tm,Tv,VQ,UP),
  assem_deriv = SparseMatrixAssembler(Tm,Tv,U_reg,U_reg),
  ls = solver,adjoint_ls = solver
)
pcfs = PDEConstrainedFunctionals(J,[C1,C2],state_map;analytic_dJ=DJ,analytic_dC=[DC1,DC2])

# ## Hilbertian extension-regularisation problems
α = α_coeff*maximum(el_Δ)
a_hilb(p,q) = ∫(α^2*∇(p)⋅∇(q) + p*q)dΩ;
vel_ext = VelocityExtension(
  a_hilb,U_reg,V_reg;
  assem = SparseMatrixAssembler(Tm,Tv,U_reg,V_reg),
  ls = PETScLinearSolver()
)

# ## Optimiser
has_oscillations(m,os_it) = LevelSetTopOpt.default_has_oscillations(m,os_it;itlength=25)
optimiser = HilbertianProjection(pcfs,stencil,vel_ext,φh;γ,γ_reinit,has_oscillations,
  verbose=i_am_main(ranks),ls_enabled=true)
for (it, uh, φh) in optimiser
  u1,ϕ1,u2,ϕ2,u3,ϕ3,u4,ϕ4,u5,ϕ5 = uh
  data = ["φ"=>φh,"H(φ)"=>(H ∘ φh),"|∇(φ)|"=>(norm ∘ ∇(φh)),
    "u1"=>1/k0*u1,"ϕ1"=>1/α0*ϕ1,"u2"=>1/k0*u2,"ϕ2"=>1/α0*ϕ2,"u3"=>1/k0*u3,"ϕ3"=>1/α0*ϕ3,
    "u4"=>1/k0*u4,"ϕ4"=>1/α0*ϕ4,"u5"=>1/k0*u5,"ϕ5"=>1/α0*ϕ5]
  writevtk(Ω,path*"out$it",cellfields=data)
  write_history(path*"/history.txt",optimiser.history)
end
#end

#with_debug() do distribute
#  mesh_partition = (2,2)
#  solver_options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
#    -ksp_converged_reason -ksp_rtol 1.0e-12"
#  GridapPETSc.with(args=split(solver_options)) do
#    main(mesh_partition,distribute)
#  end
#end

# mesh_partition = (2,2)
# ranks = with_debug() do distribute
#   distribute(LinearIndices((prod(mesh_partition),)))
# end

# el_size = (200,200)
# order = 1
# xmax,ymax=(1.0,1.0)
# dom = (0,xmax,0,ymax)
# γ = 0.1
# γ_reinit = 0.5
# max_steps = floor(Int,order*minimum(el_size)/10)
# tol = 1/(5order^2)/minimum(el_size)
# η_coeff = 2
# α_coeff = 4*max_steps*γ
# vf = 0.5

# ## FE Setup
# model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size,isperiodic=(true,true))
# el_Δ = get_el_Δ(model)
# f_Γ_D(x) = iszero(x)
# update_labels!(1,model,f_Γ_D,"origin")

# ## Triangulations and measures
# Ω = Triangulation(model)
# dΩ = Measure(Ω,2*order)
# vol_D = sum(∫(1)dΩ)

# ## Spaces
# reffe = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
# reffe_scalar = ReferenceFE(lagrangian,Float64,order)
# V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["origin"])
# U = TrialFESpace(V,VectorValue(0.0,0.0))
# Q = TestFESpace(model,reffe_scalar;conformity=:H1,dirichlet_tags=["origin"])
# P = TrialFESpace(Q,0)
# mfs = BlockMultiFieldStyle()
# UP = MultiFieldFESpace([U,P];style=mfs)
# VQ = MultiFieldFESpace([V,Q];style=mfs)

# Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
# Tv = Vector{PetscScalar}
# assem = SparseMatrixAssembler(Tm,Tv,UP,VQ)
