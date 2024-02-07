using LevelSetTopOpt
using Gridap
using Gridap.CellData: DomainContribution
using Gridap.FESpaces: _gradient
using GridapDistributed: DistributedDomainContribution
import Base.*, Base./

struct FunctionOperation{A} <: Function
  a
  b
  op :: A
end

(F::FunctionOperation)(args...) = F.op(sum(F.a(args...)),sum(F.b(args...)))

function (*)(a::Function,b::Function)
  FunctionOperation(a,b,*)
end

function (*)(a::Function,b::FunctionOperation)
  FunctionOperation(a,b,*)
end

(*)(a::FunctionOperation,b::Function) = b*a

Gridap.gradient(F::FunctionOperation{typeof(*)},xh) = gradient(F.a,xh)*sum(F.b(xh)) + gradient(F.b,xh)*sum(F.a(xh))

## Testing with ChainRules.jl
D = 1
model = CartesianDiscreteModel((0,1,0,1),(10,10));
Ω = Triangulation(model)
Γ_N = BoundaryTriangulation(model,tags=[6,])
dΩ = Measure(Ω,2)
dΓ_N = Measure(Γ_N,2)
reffe_scalar = ReferenceFE(lagrangian,Float64,1)
V = TestFESpace(model,reffe_scalar;dirichlet_tags=[1])
U = TrialFESpace(V,0.0)
V_φ = TestFESpace(model,reffe_scalar)
V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=[6,])
U_reg = TrialFESpace(V_reg,0)

## Create FE functions
φh = interpolate(x->-1,V_φ)
interp = SmoothErsatzMaterialInterpolation(η = 2maximum(get_el_size(model)));
I,_H,DH,ρ = interp.I,interp.H,interp.DH,interp.ρ

a(u,v,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*D*∇(u)⋅∇(v))dΩ
l(v,φ,dΩ,dΓ_N) = ∫(v)dΓ_N

## Optimisation functionals
J1(u,φ,dΩ,dΓ_N) = ∫((I ∘ φ)*D*∇(u)⋅∇(u))dΩ
J2(u,φ,dΩ,dΓ_N) = ∫(ρ ∘ φ)dΩ

J = J1*J2

J_iwm = LevelSetTopOpt.IntegrandWithMeasure(J,(dΩ,dΓ_N));
uh = interpolate(x->x[1]^2*x[2]^2,V);
gradient(J_iwm,[uh,φh],1)

function Gridap.gradient(F::LevelSetTopOpt.IntegrandWithMeasure{<:FunctionOperation},uh::Vector{<:FEFunction},K::Int)
  _f(uk) = F.F(uh[1:K-1]...,uk,uh[K+1:end]...,F.dΩ...)
  return gradient(_f,uh[K])
end

# state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh,dΩ,dΓ_N)
# pcfs = PDEConstrainedFunctionals(J,state_map)

## TESTING
model = CartesianDiscreteModel((0,1,0,1),(10,10));
Ω = Triangulation(model)
dΩ = Measure(Ω,2)
V = TestFESpace(model,ReferenceFE(lagrangian,Float64,1))
uh = interpolate(x->x[1]^2*x[2]^2,V);
F(u) = ∫(u)dΩ;
G(u) = ∫(u*u)dΩ;
H(u) = ∫(cos ∘ u)dΩ;

# d/du F*F
dF = gradient(F,uh);
dFxF = 2dF*sum(F(uh));
dFxF_new = gradient(F*F,uh);
@assert get_array(dFxF_new) == get_array(dFxF)

# d/du F*G
dF = gradient(F,uh);
dG = gradient(G,uh);
dFxG = dF*sum(G(uh)) + dG*sum(F(uh));
dFxG_new = gradient(F*G,uh);
@assert get_array(dFxG) == get_array(dFxG_new)
# d/du F*G*H
dF = gradient(F,uh);
dG = gradient(G,uh);
dH = gradient(H,uh);
dFxGxH = dF*sum(G(uh))*sum(H(uh)) + dG*sum(F(uh))*sum(H(uh)) + dH*sum(F(uh))*sum(G(uh));
dFxGxH_new = gradient(F*G*H,uh);
get_array(dFxG) == get_array(dFxG_new)