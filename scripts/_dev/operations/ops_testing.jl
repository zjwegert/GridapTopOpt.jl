using LevelSetTopOpt
using Gridap
using Gridap.CellData: DomainContribution
using Gridap.FESpaces: _gradient
using GridapDistributed: DistributedDomainContribution
import Base.*, Base./

ContributionTypes = Union{DomainContribution,DistributedDomainContribution}

struct DomainContributionOperation{A}
  a
  b
  op :: A
end

(F::DomainContributionOperation)(args...) = F.op(sum(F.a),sum(F.b))

function (*)(a::ContributionTypes,b::ContributionTypes)
  DomainContributionOperation(a,b,*)
end

function (*)(a::ContributionTypes,b::DomainContributionOperation)
  DomainContributionOperation(a,b,*)
end

(*)(a::DomainContributionOperation,b::ContributionTypes) = b*a

Gridap.gradient(F::DomainContributionOperation{typeof(*)},xh) = gradient(F.a,xh)*sum(F.b(xh)) + gradient(F.b,xh)*sum(F.a(xh))

# function Gridap.FESpaces._gradient(f,xh,fuh::DomainContributionOperation)
#   @show typeof(fuh.a)
#   _gradient(f,xh,fuh.a)#*sum(fuh.b(xh)) + _gradient(f,xh,fuh.b)*sum(fuh.a(xh))
# end

## TESTING
model = CartesianDiscreteModel((0,1,0,1),(10,10));
Ω = Triangulation(model)
dΩ = Measure(Ω,2)
V = TestFESpace(model,ReferenceFE(lagrangian,Float64,1))
uh = interpolate(x->x[1]^2*x[2]^2,V);
F(u) = ∫(u)dΩ;
G(u) = ∫(u*u)dΩ;
# H(u) = ∫(cos ∘ u)dΩ;

struct FunctionOperation
  a
  b
  op
end

function (*)(a::Function,b::Function)
  FunctionOperation(a,b,*)
end

function (*)(a::Function,b::FunctionOperation)
  IntegrandWithMeasureOperation(a,b,*)
end

(*)(a::FunctionOperation,b::Function) = b*a

(F*G)

# d/du F*F
dF = gradient(F,uh);
dFxF = 2dF*sum(F(uh));
dFxF_new = gradient(FG,uh);
@assert get_array(dFxF_new) == get_array(dFxF)

# # d/du F*G
# dF_iwf = gradient(F_iwf,uh);
# dG_iwf = gradient(G_iwf,uh);
# dFxG = dF_iwf*sum(G_iwf(uh)) + dG_iwf*sum(F_iwf(uh));
# dFxG_new = gradient(F_iwf*G_iwf,uh);
# @assert get_array(dFxG) == get_array(dFxG_new)
# # d/du (F*G)*H
# dF_iwf = gradient(F_iwf,uh);
# dG_iwf = gradient(G_iwf,uh);
# dH_iwf = gradient(H_iwf,uh);
# # dFxGxH = dF_iwf*sum(G_iwf(uh)) + dG_iwf*sum(F_iwf(uh));
# dFxGxH_new = gradient(F_iwf*G_iwf*H_iwf,uh);

# sum(F_iwf(uh))*sum(G_iwf(uh))
# (F_iwf*G_iwf)(uh)

# # get_array(dFxG) == get_array(dFxG_new)