using Gridap, Gridap.MultiField, GridapDistributed, GridapPETSc, GridapSolvers, 
  PartitionedArrays, LevelSetTopOpt, SparseMatricesCSR

nothing
  
np = (2,3);

ranks = with_debug() do distribute
    distribute(LinearIndices((prod(np),)))
end;

## Parameters
order = 1;
xmax,ymax=(1.0,1.0)
dom = (0,xmax,0,ymax);
el_size = (10,10);
γ = 0.05;
γ_reinit = 0.5;
max_steps = floor(Int,minimum(el_size)/10)
tol = 1/(order^2*10)/minimum(el_size)
C = isotropic_elast_tensor(2,1.,0.3);
η_coeff = 2;
α_coeff = 4;
path = dirname(dirname(@__DIR__))*"/results/block_testing"

## FE Setup
model = CartesianDiscreteModel(ranks,(2,3),dom,el_size,isperiodic=(true,true));
# model = CartesianDiscreteModel(dom,el_size,isperiodic=(true,true));
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
φh = interpolate(-1,V_φ);
φ = get_free_dof_values(φh)

## Interpolation and weak form
interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

εᴹ = (TensorValue(1.,0.,0.,0.),     # ϵᵢⱼ⁽¹¹⁾≡ϵᵢⱼ⁽¹⁾
    TensorValue(0.,0.,0.,1.),     # ϵᵢⱼ⁽²²⁾≡ϵᵢⱼ⁽²⁾
    TensorValue(0.,1/2,1/2,0.))   # ϵᵢⱼ⁽¹²⁾≡ϵᵢⱼ⁽³⁾

a(u,v,φ,dΩ) = ∫((I ∘ φ)*sum(C ⊙ ε(u[i]) ⊙ ε(v[i]) for i = 1:length(u)))dΩ
l(v,φ,dΩ) = ∫(-(I ∘ φ)*sum(C ⊙ εᴹ[i] ⊙ ε(v[i]) for i ∈ 1:length(v)))dΩ;
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
ls_evo = HamiltonJacobiEvolution(FirstOrderStencil(2,Float64),model,V_φ,el_size./order,max_steps,tol);
reinit!(stencil,φ,γ_reinit)

## Initialise op
uhd = zero(U);
Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
Tv=Vector{PetscScalar}
assem = DiagonalBlockMatrixAssembler(SparseMatrixAssembler(Tm,Tv,U,V));
@time op = AffineFEOperator((u,v) -> a(u,v,φh,dΩ),v -> l(v,φh,dΩ),U,V,assem);
K = get_matrix(op); b = get_vector(op);
x = GridapDistributed.allocate_in_domain(K)

## Initialise adjoint
assem_adjoint = DiagonalBlockMatrixAssembler(SparseMatrixAssembler(V,U));
adjoint_K = assemble_matrix((u,v) -> a(v,u,φh,dΩ),assem_adjoint,V,U);

## Update mat and vec
LevelSetTopOpt._assemble_matrix_and_vector!((u,v) -> a(u,v,φh,dΩ),v -> l(v,φh,dΩ),K,b,assem,U,V,uhd)
# numerical_setup!(...)

## Update adjoint
LevelSetTopOpt.assemble_matrix!((u,v) -> a(v,u,φh,dΩ),adjoint_K,assem_adjoint,V,U)

### Test
@time op_test = AffineFEOperator((u,v) -> a(u,v,φh,dΩ),v -> l(v,φh,dΩ),U,V,SparseMatrixAssembler(U,V))
K_test = get_matrix(op_test); b_test=get_vector(op_test)
x_test = GridapDistributed.allocate_in_domain(K_test)
@show Base.summarysize(K)
@show Base.summarysize(K_test)

using BlockArrays
if typeof(K_test) <: BlockArray
  @assert K_test == K
else
  for I ∈ CartesianIndices(K.blocks)
    @assert K.blocks[I].matrix_partition.items[:] ≈ K_test.blocks[I].matrix_partition.items[:]
  end
end

## Solver
options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
  -ksp_converged_reason -ksp_rtol 1.0e-12 -mat_block_size 3
  -mg_levels_ksp_type chebyshev -mg_levels_esteig_ksp_type cg -mg_coarse_sub_pc_type cholesky";

GridapPETSc.with(args=split(options)) do
  P = BlockDiagonalPreconditioner(map(Vi -> ElasticitySolver(Vi),V))
  solver = GridapSolvers.LinearSolvers.CGSolver(P;rtol=1.e-8,verbose=i_am_main(ranks))
  P2 = BlockDiagonalPreconditioner(map(Vi -> ElasticitySolver(Vi),V))
  solver2 = GridapSolvers.LinearSolvers.CGSolver(P2;rtol=1.e-8,verbose=i_am_main(ranks))

  @time ns = numerical_setup(symbolic_setup(solver,K),K)
  @time ns_test = numerical_setup(symbolic_setup(solver2,K_test),K_test)

  @time numerical_setup!(ns,K)
  @time numerical_setup!(ns_test,K_test)
end