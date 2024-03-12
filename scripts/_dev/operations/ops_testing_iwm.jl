using LevelSetTopOpt
using Gridap
using LevelSetTopOpt: IntegrandWithMeasure
import Base.*, Base./

struct IntegrandWithMeasureOperation{A<:Function}
  a
  b
  op :: A
end

(F::IntegrandWithMeasureOperation)(args...) = F.op(sum(F.a(args...)),sum(F.b(args...)))

function (*)(a::IntegrandWithMeasure,b::IntegrandWithMeasure)
  IntegrandWithMeasureOperation(a,b,*)
end

function (*)(a::IntegrandWithMeasure,b::IntegrandWithMeasureOperation)
  IntegrandWithMeasureOperation(a,b,*)
end

(*)(a::IntegrandWithMeasureOperation,b::IntegrandWithMeasure) = b*a

Gridap.gradient(F::IntegrandWithMeasureOperation{typeof(*)},xh) = gradient(F.a,xh)*sum(F.b(xh)) + gradient(F.b,xh)*sum(F.a(xh))

# function (/)(a::Number,b::DomainContribution)
#   DomainContributionOperation(a,b,/)
# end

# function (/)(a::DomainContribution,b::DomainContribution)
#   DomainContributionOperation(a,b,/)
# end

# function (/)(a::DomainContribution,b::DomainContributionOperation)
#   DomainContributionOperation(a,b,/)
# end

# function (/)(a::DomainContributionOperation,b::DomainContribution)
#   DomainContributionOperation(a,b,/)
# end

## TESTING
model = CartesianDiscreteModel((0,1,0,1),(10,10));
Ω = Triangulation(model)
dΩ = Measure(Ω,2)
V = TestFESpace(model,ReferenceFE(lagrangian,Float64,1))
uh = interpolate(x->x[1]^2*x[2]^2,V);
F_iwf = IntegrandWithMeasure((u,dΩ)->∫(u)dΩ,(dΩ,));
G_iwf = IntegrandWithMeasure((u,dΩ)->∫(u*u)dΩ,(dΩ,));
H_iwf = IntegrandWithMeasure((u,dΩ)->∫(cos ∘ u)dΩ,(dΩ,));
# d/du F*F
dF_iwf = gradient(F_iwf,uh);
dFxF = 2dF_iwf*sum(F_iwf(uh));
dFxF_new = gradient(F_iwf*F_iwf,uh);
@assert get_array(dFxF_new) == get_array(dFxF)
# d/du F*G
dF_iwf = gradient(F_iwf,uh);
dG_iwf = gradient(G_iwf,uh);
dFxG = dF_iwf*sum(G_iwf(uh)) + dG_iwf*sum(F_iwf(uh));
dFxG_new = gradient(F_iwf*G_iwf,uh);
@assert get_array(dFxG) == get_array(dFxG_new)
# d/du (F*G)*H
dF_iwf = gradient(F_iwf,uh);
dG_iwf = gradient(G_iwf,uh);
dH_iwf = gradient(H_iwf,uh);
# dFxGxH = dF_iwf*sum(G_iwf(uh)) + dG_iwf*sum(F_iwf(uh));
dFxGxH_new = gradient(F_iwf*G_iwf*H_iwf,uh);

sum(F_iwf(uh))*sum(G_iwf(uh))
(F_iwf*G_iwf)(uh)

# get_array(dFxG) == get_array(dFxG_new)