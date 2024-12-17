using Gridap, GridapTopOpt, GridapDistributed, PartitionedArrays

parts = (3,3);
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(parts),)))
end

path="./results/TEST_thermal_compliance_ALM2/"
## Parameters
order = 1
xmax=ymax=1.0
prop_Γ_N = 0.2
prop_Γ_D = 0.2
dom = (0,xmax,0,ymax)
el_size = (200,200)
γ = 0.1
γ_reinit = 0.5
max_steps = floor(Int,order*minimum(el_size)/10)
tol = 1/(5*order^2)/minimum(el_size)
κ = 1
vf = 0.4
η_coeff = 2
α_coeff = 4max_steps*γ
iter_mod = 10
mkpath(path)

## FE Setup
model = CartesianDiscreteModel(ranks,parts,dom,el_size);
el_Δ = get_el_Δ(model)
f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= ymax*prop_Γ_D + eps() ||
    x[2] >= ymax-ymax*prop_Γ_D - eps()))
f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/2 - eps() <= x[2] <=
    ymax/2+ymax*prop_Γ_N/2 + eps())
update_labels!(1,model,f_Γ_D,"Gamma_D")
update_labels!(2,model,f_Γ_N,"Gamma_N")

## Triangulations and measures
Ω = Triangulation(model)
Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
dΩ = Measure(Ω,2*order)
dΓ_N = Measure(Γ_N,2*order)
vol_D = sum(∫(1)dΩ)

## Spaces
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_D"])
U = TrialFESpace(V,0.0)
V_φ = TestFESpace(model,reffe_scalar)
V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
U_reg = TrialFESpace(V_reg,0)

## Create FE functions
φh = interpolate(initial_lsf(4,0.2),V_φ)

## Interpolation and weak form
interp = SmoothErsatzMaterialInterpolation(η = η_coeff*maximum(el_Δ))
I,H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

a(u,v,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(v))dΩ
l(v,φ) = ∫(v)dΓ_N

## Optimisation functionals
J(u,φ) = ∫((I ∘ φ)*κ*∇(u)⋅∇(u))dΩ
dJ(q,u,φ) = ∫(κ*∇(u)⋅∇(u)*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ;
Vol(u,φ) = ∫(((ρ ∘ φ) - vf)/vol_D)dΩ;
dVol(q,u,φ) = ∫(-1/vol_D*q*(DH ∘ φ)*(norm ∘ ∇(φ)))dΩ

state_map = AffineFEStateMap(a,l,U,V,V_φ,V_φ,φh)

# Create StateParamIntegrandWithMeasures
_J = GridapTopOpt.StateParamIntegrandWithMeasure(J,state_map)

function φ_to_j(φ)
  u = state_map(φ)
  _J(u,φ)
end

using Zygote

state_map(get_free_dof_values(φh))

φ_to_j(get_free_dof_values(φh))

_data = Zygote.gradient(φ_to_j,get_free_dof_values(φh))

sum(_data[1])