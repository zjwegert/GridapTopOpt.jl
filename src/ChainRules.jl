"""
    Functional

    A functional over a measure dΩ that enables derivatives in each state.

    - F: (states...,[measures...]) -> ∫(f(states...))dΩ1 + ∫(g(states...))dΩ2 ...
    - dΩ: A measure, can be a lower dimensional measure as well (e.g., boundary measure dΓ).
    - state: Tuple of length N of states. These must be of type FEFunction or DistributedCellField.
"""
struct Functional{N}
    F
    dΩ::Vector
    state::NTuple{N}
    function Functional(F,dΩ::Vector,args...)
        N = length(args)
        new{N}(F,dΩ,args)
    end
end

Functional(F,dΩ::Measure,args...) = Functional(F,[dΩ],args...)
Functional(F,dΩ::GridapDistributed.DistributedMeasure,args...) = Functional(F,[dΩ],args...)

"""
    FunctionalGradient

    Enables differentation of Functional in the Kth state.
"""
struct FunctionalGradient{N,K}
    F::Functional{N}
    function FunctionalGradient(F::Functional{N},K) where N
        @assert 0<K<=N
        new{N,K}(F)
    end
end

function (fg::FunctionalGradient{N,K})(uh::GridapDistributed.DistributedCellField) where {N,K}
    fields = map(i->i==K ? uh : fg.F.state[i],1:N)
    local_fields = map(local_views,fields) |> GridapDistributed.to_parray_of_arrays
    local_measures = map(local_views,fg.F.dΩ) |> GridapDistributed.to_parray_of_arrays
    contribs = map(local_measures,local_fields) do dΩ,lf
        _f = u -> fg.F.F(lf[1:K-1]...,u,lf[K+1:end]...,dΩ...)
        return Gridap.Fields.gradient(_f,lf[K])
    end
    return GridapDistributed.DistributedDomainContribution(contribs)
end

function (fg::FunctionalGradient{N,K})(uh::FEFunction) where {N,K}
    fields = map(i->i==K ? uh : fg.F.state[i],1:N)
    _f = u -> fg.F.F(fields[1:K-1]...,u,fields[K+1:end]...,fg.F.dΩ...)
    return Gridap.Fields.gradient(_f,fields[K])
end

####################################################################

abstract type AbstractStateFunctional end
# Need to decide if this is something we want the user to be able to extend...
# I think yes, but need a clean API.

"""
    SingleStateFunctional

    Assume that we have a Functional of either the following form:
        J₁: (u,ϕ,dΩ) ↦ ∫_Ω f(u(ϕ),ϕ) dΩ,
        J₂: (u,ϕ,dΓ) ↦ ∫_Γ g(u(ϕ),ϕ) dΓ,
        J₃: (u,ϕ,[dΩ,dΓ]) ↦ ∫_Ω f(u(ϕ),ϕ) dΩ + ∫_Γ g(u(ϕ),ϕ) dΓ,
    where u is a state field and ϕ is auxilary. The signature MUST match the weak form.

    The number of states refers to number of solutions to PDEs (e.g., u above).
    We assume that there is only one additional auxilary field ϕ for purpose of AD.
"""
struct SingleStateFunctional{U,V,Vϕ} <: AbstractStateFunctional
    F::Functional{2}
    trial_space::U
    test_space::V
    # assembler::Assembler
    aux_space::Vϕ
end

function (u_to_j::SingleStateFunctional)(u,ϕ) # <- function signature may have to be more general if we want an extendable API
    F=u_to_j.F
    U=u_to_j.trial_space
    V_ϕ=u_to_j.aux_space
    _uh=FEFunction(U,u)
    _φh=FEFunction(V_ϕ,ϕ)
    sum(F.F(_uh,_φh,F.dΩ...))
end

function ChainRulesCore.rrule(u_to_j::SingleStateFunctional,u,ϕ)
    F=u_to_j.F
    U=u_to_j.trial_space
    V_ϕ=u_to_j.aux_space
    _uh=FEFunction(U,u)
    _φh=FEFunction(V_ϕ,ϕ)
    jp=sum(F.F(_uh,_φh,F.dΩ...))
    function u_to_j_pullback(dj)
        djdu = FunctionalGradient(F,1)(_uh)
        djdu_vec = assemble_vector(djdu,U)
        djdϕ = FunctionalGradient(F,2)(_φh)
        djdϕ_vec = assemble_vector(djdϕ,V_ϕ)
        (  NoTangent(), dj*djdu_vec, dj*djdϕ_vec )
    end
    jp, u_to_j_pullback
end

struct AffineFEStateMap{S}
    a::Function
    l::Function
    res::Function
    F::S
    # solver
    # adjoint_solver
end

function (ϕ_to_u::AffineFEStateMap{S} where S<:SingleStateFunctional)(ϕ)
    a=ϕ_to_u.a
    l=ϕ_to_u.l
    res=ϕ_to_u.res
    V_ϕ=ϕ_to_u.F.aux_space
    U=ϕ_to_u.F.trial_space
    V=ϕ_to_u.F.test_space
    meas = ϕ_to_u.F.F.dΩ
    # assem = ϕ_to_u.F.assembler
    op = AffineFEOperator(a(ϕ,meas...),l(ϕ,meas...),U,V)#,assem)
    get_free_dof_values(Gridap.solve(op))
end

function ChainRulesCore.rrule(ϕ_to_u::AffineFEStateMap{S} where S<:SingleStateFunctional,ϕ)
    a=ϕ_to_u.a
    l=ϕ_to_u.l
    res=ϕ_to_u.res
    V_ϕ=ϕ_to_u.F.aux_space
    U=ϕ_to_u.F.trial_space
    V=ϕ_to_u.F.test_space
    meas = ϕ_to_u.F.F.dΩ
    op = AffineFEOperator(a(ϕ,meas...),l(ϕ,meas...),U,V)#,assem)
    adjoint_op = AffineFEOperator(a(ϕ,meas...),l(ϕ,meas...),U,V)#,assem)
    _uh = Gridap.solve(op)
    function ϕ_to_u_pullback(du)
        dϕ = Adjoint(ϕ,_uh,du,adjoint_op,res,ϕ_to_u)     
        ( NoTangent(),dϕ)
    end
    get_free_dof_values(_uh), ϕ_to_u_pullback
end

function Adjoint(ϕ,_uh,du,adjoint_op,res,F::AffineFEStateMap{S} where S<:SingleStateFunctional)
    V_ϕ = F.F.aux_space
    Aᵀ = Gridap.jacobian(adjoint_op,_uh) # = dr/du
    V = adjoint_op.trial
    λ = solve(LUSolver(),Aᵀ,du)
    λh = FEFunction(V,λ)
    _φh = FEFunction(V_ϕ,ϕ)
    res_functional = Functional(res,F.F.F.dΩ,_uh,λh,_φh)
    dϕ() = FunctionalGradient(res_functional,3)(_φh)
    dϕ = -assemble_vector(dϕ(),V_ϕ)
end

## Helpers
function ϕ_to_ϕₕ(ϕ::AbstractArray,Q)
	ϕ = FEFunction(Q,ϕ)
end
function ϕ_to_ϕₕ(ϕ::FEFunction,Q)
	ϕ
end
function ϕ_to_ϕₕ(ϕ::CellField,Q)
	ϕ
end
function ϕ_to_ϕₕ(ϕ::GridapDistributed.DistributedCellField,Q)
	ϕ
end