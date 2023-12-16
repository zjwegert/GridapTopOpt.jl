using Gridap, Gridap.MultiField, GridapDistributed, GridapPETSc, GridapSolvers, 
  PartitionedArrays, LSTO_Distributed, SparseMatricesCSR

  
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
tol = 1/(order^2*10)*prod(inv,minimum(el_size))
C = isotropic_2d(1.,0.3);
η_coeff = 2;
α_coeff = 4;
path = dirname(dirname(@__DIR__))*"/results/block_testing"

## FE Setup
model = CartesianDiscreteModel(ranks,(2,3),dom,el_size,isperiodic=(true,true));
Δ = get_Δ(model)
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
lsf_fn = x->max(gen_lsf(2,0.4)(x),gen_lsf(2,0.4;b=VectorValue(0,0.5))(x));
φh = interpolate(lsf_fn,V_φ);
φ = get_free_dof_values(φh)

## Interpolation and weak form
interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ))
I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

εᴹ = (TensorValue(1.,0.,0.,0.),     # ϵᵢⱼ⁽¹¹⁾≡ϵᵢⱼ⁽¹⁾
    TensorValue(0.,0.,0.,1.),     # ϵᵢⱼ⁽²²⁾≡ϵᵢⱼ⁽²⁾
    TensorValue(0.,1/2,1/2,0.))   # ϵᵢⱼ⁽¹²⁾≡ϵᵢⱼ⁽³⁾

a(u,v,φ,dΩ) = ∫((I ∘ φ)*sum(C ⊙ ε(u[i]) ⊙ ε(v[i]) for i ∈ eachindex(εᴹ)))dΩ
a_single(u,v,φ,dΩ) = ∫((I ∘ φ)*(C ⊙ ε(u) ⊙ ε(v)))dΩ
l(v,φ,dΩ) = ∫(-(I ∘ φ)*sum(C ⊙ εᴹ[i] ⊙ ε(v[i]) for i ∈ eachindex(εᴹ)))dΩ;
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
stencil = AdvectionStencil(FirstOrderStencil(2,Float64),model,V_φ,Δ./order,max_steps,tol);
reinit!(stencil,φ,γ_reinit)

## Special assembly option
using BlockArrays

assem = SparseMatrixAssembler(first(U),first(V));
du = get_trial_fe_basis(first(U));
dv = get_fe_basis(first(V));
uhd = zero(first(U))
matcontribs = a_single(du,dv,φh,dΩ);
data = Gridap.FESpaces.collect_cell_matrix(first(U),first(V),matcontribs);
@time A = assemble_matrix(assem,data);

full_assem = SparseMatrixAssembler(U,V);
full_du = get_trial_fe_basis(U);
full_dv = get_fe_basis(V);
full_uhd = zero(U);
full_matcontribs = a(full_du,full_dv,φh,dΩ)
full_data = Gridap.FESpaces.collect_cell_matrix(U,V,full_matcontribs);
@time full_A = assemble_matrix(full_assem,full_data);
# @time full_A = Gridap.FESpaces.allocate_matrix(full_assem,full_data);

full_A_cached = zero(full_A);
copy!(full_A_cached[Block(1,1)],A)
full_A_cached[Block(2,2)] = full_A_cached[Block(1,1)]
full_A_cached[Block(3,3)] = full_A_cached[Block(1,1)]
full_A_cached[Block(2,2)] === full_A_cached[Block(1,1)]
full_A_cached[Block(3,3)] === full_A_cached[Block(1,1)]

fill(full_A_cached,1);

full_A_cached == full_A

# op = AffineFEOperator((u,v) -> a(u,v,φh,dΩ),v -> l(v,φh,dΩ),U,V,SparseMatrixAssembler(U,V))
# K = get_matrix(op);