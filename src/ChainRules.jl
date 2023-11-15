using ChainRulesCore
using Gridap.Algebra: LinearSolver

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
    local_fields = map(local_views,uh) |> GridapDistributed.to_parray_of_arrays
    local_measures = map(local_views,F.dΩ) |> GridapDistributed.to_parray_of_arrays
    contribs = map(local_measures,local_fields) do dΩ,lf
        _f = u -> F.F(lf[1:K-1]...,u,lf[K+1:end]...,dΩ...)
        return Gridap.Fields.gradient(_f,lf[K])
    end
    return GridapDistributed.DistributedDomainContribution(contribs)
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
        V_φ, #;
        assem_U, # =SparseMatrixAssembler(U,U),
        assem_φ) # =SparseMatrixAssembler(V_φ,V_φ))

    dΩ = F.dΩ
    φ₀ = zero(V_φ); u₀ = zero(U)
    djdu_vec = assemble_vector(v->F.F(v,φ₀,dΩ...),assem_U,U)
    djdφ_vec = assemble_vector(v->F.F(u₀,v,dΩ...),assem_φ,V_φ)
    assems = (assem_U,assem_φ)
    spaces = (U,V_φ)
    caches = (djdu_vec,djdφ_vec)
    StateParamIntegrandWithMeasure(F,spaces,assems,caches)
end

(u_to_j::StateParamIntegrandWithMeasure)(uh,φh) = u_to_j.F(uh,φh)

function rrule(u_to_j::StateParamIntegrandWithMeasure,uh,φh)
    F=u_to_j.F
    U,V_φ = u_to_j.spaces
    assem_U,assem_φ = u_to_j.assem
    djdu_vec,djdφ_vec = u_to_j.caches
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
    u_to_j(_uh,_φh), u_to_j_pullback
end

struct AffineFEStateMap{A,B,C,D<:Tuple,E<:Tuple,F<:Tuple,G<:Tuple,H<:Tuple}
    a               ::A
    l               ::B
    res             ::C
    dΩ              ::D
    spaces          ::E
    cache           ::F
    fwd_caches      ::G
    adjoint_caches  ::H
end

function AffineFEStateMap(
        a,
        l,
        res,
        dΩ::Tuple,
        spaces::Tuple,
        assems::Tuple,
        ls::Gridap.Algebra.LinearSolver,
        adjoint_ls::Gridap.Algebra.LinearSolver)
    
    U,V,V_φ = spaces
    assem_U,assem_adjoint,assem_φ = assems

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
    return AffineFEStateMap(a,l,res,dΩ,spaces,caches,fwd_caches,adjoint_caches)
end

function (φ_to_u::AffineFEStateMap)(φh)
    a=φ_to_u.a
    l=φ_to_u.l
    dΩ = φ_to_u.dΩ
    U,V,_ = φ_to_u.spaces
    ns,K,b,x,assem_U = φ_to_u.fwd_caches
    
    ## Reassemble and solve
    assemble_matrix_and_vector!(a(du,dv,φh,dΩ...),l(dv,φh,dΩ...),K,b,assem_U,U,V)
    numerical_setup!(ns,K)
    solve!(x,ns,b)
    x
end

function ChainRulesCore.rrule(φ_to_u::AffineFEStateMap,φh)
    a=φ_to_u.a
    res = F.res
    dΩ = φ_to_u.dΩ
    U,V,V_φ = φ_to_u.spaces
    dφdu_vec,assem_φ = F.caches
    adjoint_ns,adjoint_K,λ,assem_adjoint = φ_to_u.adjoint_caches
    
    ## Forward problem
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
struct PDEConstrainedFunctionals{A,B}
    J :: StateParamIntegrandWithMeasure
    C :: Vector{StateParamIntegrandWithMeasure}
    dJ :: A
    dC :: Vector{B}
    state_map :: AffineStateMap
end

function PDEConstrainedFunctionals(
        J :: Function,
        C :: Vector{Function},
        a,l,res,
        U,V,V_φ,
        dΩ...;
        assem_U = SparseMatrixAssembler(U,V),
        assem_adjoint = SparseMatrixAssembler(V,U),
        assem_φ = SparseMatrixAssembler(V_φ,V_φ),
        ls::LinearSolver = LUSolver(),
        adjoint_ls::LinearSolver = LUSolver(),
        dJ = nothing,
        dC = fill(nothing,length(C)))

    # Create StateParamIntegrandWithMeasures
    spiwm(f) = StateParamIntegrandWithMeasure(IntegrandWithMeasure(f,dΩ),U,V_φ,assem_U,assem_φ)
    J_spiwm = spiwm(J)
    C_spiwm = map(spiwm,C)
 
    # Create AffineFEStateMap
    spaces = (U,V,V_φ)
    assems = assem_U,assem_adjoint,assem_φ
    state_map = AffineFEStateMap(a,l,res,dΩ,spaces,assems,ls,adjoint_ls)

    PDEConstrainedFunctionals(J_spiwm,C_spiwm,dJ,dC,state_map)
end

function evaluate(pcf,φ)
    u = asm(φ)
    evaluate(pcf,φ,u)
end

function evaluate(pcf,φ,u)
    [J(u,φ), map(Ci(u,φ),1:length(Ci))]
end

function evaluate_derivatives(pcf,φ)
    u = asm(φ)
    evaluate_derivatives(pcf,φ,u)
end

function evaluate_derivatives(pcf,φ,u)
    dJ = rrule(J,u,φ)
    dC = rrule(C,u,φ)
end

function evaluate_both(pcf,φ)
    u = asm(φ)
    evaluate(pcf,φ,u)
    evaluate_derivatives(pcf,φ,u)
end

####################################################################
## Shape derivative for AbstractFunctional
function AffineFEStateMap(J::T,C::Vector{T},args...;ls,adjoint_ls) where T<:AbstractFunctional
    @abstractmethod
end

function compute_shape_derivative!(φh,state_map::AffineFEStateMap)
    @abstractmethod
end

## Shape derivative for Functional{2}
"""
    Pass objective functional and array of constraint functions and return 
    AffineFEStateMap for objective and array of AffineFEStateMap for each constraint.

    Additional args for setup
""" 
function AffineFEStateMap(J::T,C::Vector{T},args...;ls=LUSolver(),
        adjoint_ls=ls) where T<:Functional{2}
    U,V,V_φ,a,l,res = args
    
    J_ssfunc = SingleStateFunctional(J,U,V,V_φ);
    J_smap = AffineFEStateMap(J_ssfunc,a,l,res,ls,adjoint_ls);
    function setup_from_cache(Ci::Functional{2})
        # dFh = J_ssfunc.dFh 
        spaces = J_ssfunc.spaces
        assems = J_ssfunc.assemblers
        ssfunc_cache = J_ssfunc.caches
        caches = J_smap.caches
        adjoint_caches = J_smap.adjoint_caches
        Ci_ssfunc = SingleStateFunctional(Ci,zero(V_φ),spaces,assems,ssfunc_cache);
        Ci_smap = AffineFEStateMap(Ci_ssfunc,a,l,res,caches,adjoint_caches)
        return Ci_smap
    end
    C_smap = map(setup_from_cache,C)

    return J_smap,C_smap
end

function compute_shape_derivative!(_φh::GridapDistributed.DistributedCellField,
        state_map::AffineFEStateMap{S}) where S<:SingleStateFunctional
    dFh = _compute_shape_derivative!(_φh,state_map)
    consistent!(get_free_dof_values(dFh)) |> fetch
    dFh
end

function compute_shape_derivative!(_φh::FEFunction,state_map::AffineFEStateMap{S}) where S<:SingleStateFunctional
    _compute_shape_derivative!(_φh,state_map)
end

function compute_shape_derivative!(_φh,state_maps::Vector{AffineFEStateMap})
    map(state_map -> compute_shape_derivative!(_φh,state_map),state_maps)
end

function _compute_shape_derivative!(_φh,state_map::AffineFEStateMap{S}) where S<:SingleStateFunctional
    ssfunc,smap = state_map.F,state_map
    _,_,V_φ = ssfunc.spaces
    _,aux_assem = ssfunc.assemblers
    dFh = ssfunc.dFh
    
    if isnothing(ssfunc.F.DF)
        # It doesn't seem resonable that we have to compute the fwd problem for each
        #   shape derivative. I think we can compute new u and objective outside and then
        #   run this without getting u and j below.
        #
        # E.g., we can call ssfunc(state...) to compute j and smap(φh) to update uh.

        u, u_pullback = rrule(smap,_φh); # Compute fwd problem
        j, j_pullback = rrule(ssfunc,u,_φh); # Compute functional
        _, du, dφ₍ⱼ₎  = j_pullback(1); # Compute derivatives of J wrt to u and φ
        _, dφ₍ᵤ₎      = u_pullback(du); # Compute adjoint for derivatives of φ wrt to u 
        dφ            = dφ₍ᵤ₎ + dφ₍ⱼ₎
        copy!(get_free_dof_values(dFh),dφ)
    else
        meas = ssfunc.F.dΩ
        state = ssfunc.F.state
        _dF = (q) -> ssfunc.F.DF(q,state...,meas...)
        dF_vec = get_free_dof_values(dFh);
        assemble_vector!(_dF,dF_vec,aux_assem,V_φ)
    end
    return dFh
end