using ChainRulesCore
using Gridap.Algebra: LinearSolver
using GridapDistributed: DistributedDomainContribution, to_parray_of_arrays

"""
    IntegrandWithMeasure

    Enables partial differentation of an integrand F via Gridap.gradient.
"""
struct IntegrandWithMeasure{A,B<:Tuple}
    F  :: A
    dΩ :: B
end

(F::IntegrandWithMeasure)(args...) = F.F(args...,F.dΩ...)

Gridap.gradient(F::IntegrandWithMeasure,uh) = Gridap.gradient(F,[uh],1)

function Gridap.gradient(F::IntegrandWithMeasure,uh::Vector{<:FEFunction},K::Int)
    @check 0 < K <= length(uh)
    _f(uk) = F.F(uh[1:K-1]...,uk,uh[K+1:end]...,F.dΩ...)
    return Gridap.gradient(_f,uh[K])
end

function Gridap.gradient(F::IntegrandWithMeasure,uh::Vector,K::Int)
    @check 0 < K <= length(uh)
    local_fields = map(local_views,uh) |> to_parray_of_arrays
    local_measures = map(local_views,F.dΩ) |> to_parray_of_arrays
    contribs = map(local_measures,local_fields) do dΩ,lf
        _f = u -> F.F(lf[1:K-1]...,u,lf[K+1:end]...,dΩ...)
        return Gridap.Fields.gradient(_f,lf[K])
    end
    return DistributedDomainContribution(contribs)
end

"""
    StateParamIntegrandWithMeasure

    Assume that we have a Functional of the following form:
        F: (u,φ,[dΩ₁,dΩ₂,...]) ↦ ∫_Ω₁ f(u(φ),φ) dΩ₁ + ∫_Ω₂ g(u(φ),φ) dΩ₂ + ...,
    where u is a state field and φ is auxilary. 
    
    Assumptions:
     - The arguments to F matchs the weak form.
     - The argument u is the solution to an FE problem. This can be a single field or multifield.
     - There is a single auxilary field. Again, this can possibly be a MultiFieldFEFunction.
        E.g., multiple level set functions.
"""
struct StateParamIntegrandWithMeasure{A<:IntegrandWithMeasure,B<:Tuple,C<:Tuple,D<:Tuple}
    F       :: A
    spaces  :: B
    assem   :: C
    caches  :: D
end

function StateParamIntegrandWithMeasure(F::IntegrandWithMeasure,
        U,
        V_φ,
        assem_U,
        assem_φ)

    dΩ = F.dΩ
    φ₀ = zero(V_φ); u₀ = zero(U)
    djdu_vec = assemble_vector(v->F.F(v,φ₀,dΩ...),assem_U,U)
    djdφ_vec = assemble_vector(v->F.F(u₀,v,dΩ...),assem_φ,V_φ)
    assems = (assem_U,assem_φ)
    spaces = (U,V_φ)
    caches = (djdu_vec,djdφ_vec)
    StateParamIntegrandWithMeasure(F,spaces,assems,caches)
end

function (u_to_j::StateParamIntegrandWithMeasure)(u<:T,φ<:T) where T<:AbstractArray
    U,V_φ = u_to_j.spaces
    uh = FEFunction(U,u)
    φh = FEFunction(V_φ,φ)
    return u_to_j.F(uh,φh)
end

function rrule(u_to_j::StateParamIntegrandWithMeasure,u<:T,φ<:T) where T<:AbstractArray
    F=u_to_j.F
    U,V_φ = u_to_j.spaces
    assem_U,assem_φ = u_to_j.assem
    djdu_vec,djdφ_vec = u_to_j.caches
    uh = FEFunction(U,u)
    φh = FEFunction(V_φ,φ)
    fields = [uh,φh];
    function u_to_j_pullback(dj)
        ## Compute ∂F/∂uh(uh) and ∂F/∂φh(φh)
        djdu = ∇(F,fields,1)
        djdu_vecdata = collect_cell_vector(U,djdu)
        assemble_vector!(djdu_vec,assem_U,djdu_vecdata)
        djdφ = ∇(F,fields,2)
        djdφ_vecdata = collect_cell_vector(V_φ,djdφ)
        assemble_vector!(djdφ_vec,assem_φ,djdφ_vecdata)
        djdu_vec .*= dj
        djdφ_vec .*= dj
        (  NoTangent(), djdu_vec, djdφ_vec )
    end
    u_to_j(u,φ), u_to_j_pullback
end

abstract type AbstractFEStateMap end

# Getters
get_state(::AbstractFEStateMap) = @abstractmethod
get_measure(::AbstractFEStateMap) = @abstractmethod;
get_trial_space(::AbstractFEStateMap) = @abstractmethod;
get_test_space(::AbstractFEStateMap) = @abstractmethod;
get_aux_space(::AbstractFEStateMap) = @abstractmethod;
get_state_assembler(::AbstractFEStateMap) = @abstractmethod
get_aux_assembler(::AbstractFEStateMap) = @abstractmethod

# Autodiff methods
(::AbstractFEStateMap)(arg) = @abstractmethod
ChainRulesCore.rrule(::AbstractFEStateMap,arg) = @abstractmethod

struct AffineFEStateMap{A,B,C,D<:Tuple,E<:Tuple,F<:Tuple,G<:Tuple,H<:Tuple} <: AbstractFEStateMap
    a               ::A
    l               ::B
    res             ::C
    dΩ              ::D
    spaces          ::E
    cache           ::F
    fwd_caches      ::G
    adjoint_caches  ::H

    function AffineFEStateMap(
            a,l,res,
            U,V,V_φ,
            dΩ...;
            assem_U = SparseMatrixAssembler(U,V),
            assem_adjoint = SparseMatrixAssembler(V,U),
            assem_φ = SparseMatrixAssembler(V_φ,V_φ),
            ls::LinearSolver = LUSolver(),
            adjoint_ls::LinearSolver = LUSolver())
        
        spaces = (U,V,V_φ)
        ## dφdu cache
        φ₀ = zero(V_φ); u₀ = zero(U)
        dφdu_vec = assemble_vector(v->l(u₀,v,dΩ...),assem_φ,V_φ)

        ## K,b,x
        op = AffineFEOperator((u,v) -> a(u,v,φ₀,dΩ...),v -> l(v,φ₀,dΩ...),U,V,assem_U)
        K = get_matrix(op); b = get_vector(op); 
        x = get_free_dof_values(zero(U))
        
        ## Adjoint K,b,x
        adjoint_K = assemble_matrix((u,v) -> a(v,u,φ₀,dΩ...),assem_adjoint,V,U)
        adjoint_x = get_free_dof_values(zero(V))

        ## Numerical setups
        ns = numerical_setup(symbolic_setup(ls,K),K)
        adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
        
        ## Caches and adjoint caches
        caches = (dφdu_vec,assem_φ)
        fwd_caches = (ns,K,b,x,assem_U)
        adjoint_caches = (adjoint_ns,adjoint_K,adjoint_x,assem_adjoint)
        return new(a,l,res,dΩ,spaces,caches,fwd_caches,adjoint_caches)
    end
end

# Getters
get_state(m::AffineFEStateMap) = FEFunction(get_trial_space(m),m.fwd_caches[4])
get_measure(m::AffineFEStateMap) = m.dΩ;
get_trial_space(m::AffineFEStateMap) = m.spaces[1];
get_test_space(m::AffineFEStateMap) = m.spaces[2];
get_aux_space(m::AffineFEStateMap) = m.spaces[3];
get_state_assembler(m::AffineFEStateMap) = last(m.fwd_caches)
get_aux_assembler(m::AffineFEStateMap) = last(m.caches)

function (φ_to_u::AffineFEStateMap)(φ<:AbstractVector)
    a=φ_to_u.a
    l=φ_to_u.l
    dΩ = φ_to_u.dΩ
    U,V,V_φ = φ_to_u.spaces
    ns,K,b,x,assem_U = φ_to_u.fwd_caches
    φh = FEFunction(V_φ,φ)

    ## Reassemble and solve
    assemble_matrix_and_vector!(a(du,dv,φh,dΩ...),l(dv,φh,dΩ...),K,b,assem_U,U,V)
    numerical_setup!(ns,K)
    solve!(x,ns,b)
    x
end

function ChainRulesCore.rrule(φ_to_u::AffineFEStateMap,φ<:AbstractVector)
    a=φ_to_u.a
    res = F.res
    dΩ = φ_to_u.dΩ
    U,V,V_φ = φ_to_u.spaces
    dφdu_vec,assem_φ = F.caches
    adjoint_ns,adjoint_K,λ,assem_adjoint = φ_to_u.adjoint_caches
    
    ## Forward problem
    φh = FEFunction(V_φ,φ)
    u = φ_to_u(φh)

    ## Adjoint operator
    assemble_matrix!((u,v) -> a(v,u,φ₀,dΩ...),adjoint_K,assem_adjoint,V,U)
    function φ_to_u_pullback(du)
        ## Adjoint Solve
        numerical_setup!(adjoint_ns,adjoint_K)
        solve!(λ,adjoint_ns,du)
        λh = FEFunction(V,λ)
        
        ## Compute grad
        uh = FEFunction(U,u)
        res_functional = IntegrandWithMeasure(res,dΩ)
        dφdu_contrib = ∇(res_functional,[uh,λh,φh],3)
        dφdu_vecdata = collect_cell_vector(V_φ,dφdu_contrib)
        assemble_vector!(dφdu_vec,assem_φ,dφdu_vecdata)
        dφdu_vec .*= -1;
        ( NoTangent(),dφdu_vec)
    end
    u, φ_to_u_pullback
end

"""
    PDEConstrainedFunctionals
"""
struct PDEConstrainedFunctionals{A<:AbstractArray,B<:Union{Function,Nothing},E<:AbstractFEStateMap}
    J :: StateParamIntegrandWithMeasure
    C :: Vector{StateParamIntegrandWithMeasure}
    dJ :: A
    dC :: Vector{A}
    analytic_dJ :: B
    analytic_dC :: Vector{B}
    state_map :: C
end

function PDEConstrainedFunctionals(
        J :: Function,
        C :: Vector{Function},
        state_map :: AbstractFEStateMap;
        analytic_dJ = nothing,
        analytic_dC = fill(nothing,length(C)))

    dΩ = get_measure(state_map)
    U = get_trial_space(state_map)
    V_φ = get_aux_space(state_map)
    assem_U = get_state_assembler(state_map)
    assem_φ = get_aux_assembler(state_map)

    # Create StateParamIntegrandWithMeasures
    spiwm(f) = StateParamIntegrandWithMeasure(
        IntegrandWithMeasure(f,dΩ),U,V_φ,assem_U,assem_φ)
    J_spiwm = spiwm(J)
    C_spiwm = map(spiwm,C)

    # Preallocate
    u₀ = zero(U)
    dJ = assemble_vector(v->J(u₀,v,dΩ...),assem_φ,V_φ)
    dC = map(_->similar(dJ),C)

    return PDEConstrainedFunctionals(
        J_spiwm,C_spiwm,dJ,dC,analytic_dJ,analytic_dC,state_map)
end

get_state(m::PDEConstrainedFunctionals) = get_state(m.state_map)

function evaluate_functionals!(pcf::PDEConstrainedFunctionals,φ<:AbstractVector)
    u = pcf.state_map(φ)
    J = pcf.J; C = pcf.C
    [J(u,φ), map(C(u,φ),1:length(C))]
    return J(u,φ), map(C(u,φ),1:length(C))
end

function evaluate_derivatives!(pcf::PDEConstrainedFunctionals,φ<:AbstractVector)
    _,_,dJ,dC = _evaluate_derivatives(pcf,φ)
    return dJ,dC
end

function evaluate!(pcf::PDEConstrainedFunctionals,φ<:AbstractVector)
    _evaluate_derivatives(pcf,φ)
end

function _evaluate_derivatives(pcf::PDEConstrainedFunctionals,φ<:AbstractVector)
    J = pcf.J; C = pcf.C
    dJ = pcf.dJ; dC = pcf.dC
    analytic_dJ = pcf.dJ; analytic_dC = pcf.dC
    U = get_trial_space(pcf.state_map)
    V_φ = get_aux_space(pcf.state_map)
    aux_assem = get_aux_assembler(pcf.state_map)
    dΩ = get_measure(pcf.state_map)

    ## Foward problem
    u, u_pullback = rrule(pcf.state_map,φ)

    function ∇!(F::StateParamIntegrandWithMeasure,dF,::Nothing)
        # Automatic differentation
        j_val, j_pullback = rrule(F,u,φ); # Compute functional and pull back
        _, dFdu, dFdφ     = j_pullback(1); # Compute dFdu, dFdφ
        _, dφ_adj         = u_pullback(dFdu); # Compute -dFdu*dudφ via adjoint 
        copy!(dF,dφ_adj)
        dF .+= dFdφ
        return j_val
    end
    function ∇!(F::StateParamIntegrandWithMeasure,dF,dF_analytic)
        # Analytic shape derivative
        j_val = F(u,φ)
        uh = FEFunction(U,u)
        φh = FEFunction(V_φ,φ)
        _dF = (q) -> dF_analytic(q,uh,φh,dΩ...)
        assemble_vector!(_dF,dF,aux_assem,V_φ)
        return j_val
    end
    j = ∇!(J,dJ,analytic_dJ)
    c = map(∇!,C,dC,analytic_dC)

    return j,c,dJ,dC
end