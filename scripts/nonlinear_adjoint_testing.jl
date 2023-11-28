using Gridap, GridapDistributed, GridapPETSc, PartitionedArrays, LSTO_Distributed

## Parameters
order = 1;
xmax=ymax=1.0
prop_Γ_N = 0.4;
prop_Γ_D = 0.2
dom = (0,xmax,0,ymax);
el_size = (100,100);
γ = 0.1;
γ_reinit = 0.5;
max_steps = floor(Int,minimum(el_size)/10)
tol = 1/(order^2*10)*prod(inv,minimum(el_size)) # <- We can do better than this I think
D = 1;
η_coeff = 2;
α_coeff = 4;
path = "./Results/nonlinear_testing"

## FE Setup
model = CartesianDiscreteModel(dom,el_size);
Δ = get_Δ(model)
f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= ymax*prop_Γ_D + eps() || 
  x[2] >= ymax-ymax*prop_Γ_D - eps())) ? true : false;
f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= 
  ymax/2+ymax*prop_Γ_N/4 + eps()) ? true : false;
update_labels!(1,model,f_Γ_D,"Gamma_D")
update_labels!(2,model,f_Γ_N,"Gamma_N")

## Triangulations and measures
Ω = Triangulation(model)
Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
dΩ = Measure(Ω,2order)
dΓ_N = Measure(Γ_N,2order)

## Spaces
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_D"])
U = TrialFESpace(V,0.0)
V_φ = TestFESpace(model,reffe_scalar)
V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
U_reg = TrialFESpace(V_reg,0)

## Create FE functions
φh = interpolate(gen_lsf(4,0.2),V_φ);
φ = get_free_dof_values(φh)

## Interpolation and weak form
interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(Δ))
I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

K(u) = exp ∘ u;

res(φ,u,v) = ∫((I ∘ φ)*K(u)*∇(u)⋅∇(v))dΩ - ∫(v)dΓ_N

op = FEOperator((u,v)->res(φh,u,v),U,V)
uh = solve(op)

J(φ,u) = ∫((I ∘ φ)*K(u)*∇(u)⋅∇(u))dΩ

dJdφ = ∇(φ -> J(φ,uh))(φh)
dJdφ_vec = assemble_vector(dJdu,V_reg)

## Adjoint solve

dRdu = Gridap.jacobian(op,uh) # = dℜ/du
λh = FEFunction(V, adjoint(dRdu)\dJdu_vec)

dRdφ = ∇(φh->res(φh,uh,λh))(φh)
dRdφ_vec = -assemble_vector(dRdφ,V_reg)

dJdφ_vec + dRdφ_vec