using ChainRulesCore

abstract type AbstractFunctional end

"""
    Functional

    A functional over a measure dΩ that enables derivatives in each state.

    - F: (states...,[measures...]) -> ∫(f(states...))dΩ1 + ∫(g(states...))dΩ2 ...
    - dΩ: A measure, can be a lower dimensional measure as well (e.g., boundary measure dΓ).
    - state: Tuple of length N of states. These must be of type FEFunction or DistributedCellField.

    Optional:
    - DF: a function defining the shape derivative of F. This function takes a test function q plus the 
        same arguments as F. If DF is nothing, automatic differentation is used to determine the shape derivative.

        E.g., for `F = (u,φ,dΩ) -> ∫(H ∘ φ)dΩ`, DF = (q,u,φ,dΩ) -> ∫(1*q*(DH ∘ φh)*(norm ∘ ∇(φh)))dΩ
        where `DH(φh)*|∇(φh)|dΩ ≈ dΓ`, `H` is the smoothed Heaviside function, and `dH` is the
        smoothed delta function.        
"""
struct Functional{N} <: AbstractFunctional
    F
    DF
    dΩ::Vector
    state::Tuple
    function Functional(F,dΩ::Vector,args...;DF=nothing)
        N = length(args)
        new{N}(F,DF,dΩ,args)
    end
end

Functional(F,dΩ::Measure,args...;DF=nothing) = Functional(F,[dΩ],args...;DF=DF)
Functional(F,dΩ::GridapDistributed.DistributedMeasure,args...;DF=nothing) = Functional(F,[dΩ],args...;DF=DF)

(F::Functional)() = sum(F.F(F.state...,F.dΩ...))
(F::Functional)(args...) = sum(F.F(args...,F.dΩ...))

function Gridap.gradient(F::Functional{N},uh::FEFunction,K::Int) where N
    @assert 0<K<=N
    fields = map(i->i==K ? uh : F.state[i],1:N)
    _f = u -> F.F(fields[1:K-1]...,u,fields[K+1:end]...,F.dΩ...)
    return Gridap.Fields.gradient(_f,fields[K])
end

function Gridap.gradient(F::Functional,uh::GridapDistributed.DistributedCellField,K::Int)
    _gradient(F,uh,K)
end

function Gridap.gradient(F::Functional,uh::GridapDistributed.DistributedMultiFieldFEFunction,K::Int)
    _gradient(F,uh,K)
end

function _gradient(F::Functional{N},uh,K::Int) where N
    @assert 0<K<=N
    fields = map(i->i==K ? uh : F.state[i],1:N)
    local_fields = map(local_views,fields) |> GridapDistributed.to_parray_of_arrays
    local_measures = map(local_views,F.dΩ) |> GridapDistributed.to_parray_of_arrays
    contribs = map(local_measures,local_fields) do dΩ,lf
        _f = u -> F.F(lf[1:K-1]...,u,lf[K+1:end]...,dΩ...)
        return Gridap.Fields.gradient(_f,lf[K]) # <- A lot of allocations from Gridap AD
    end
    return GridapDistributed.DistributedDomainContribution(contribs)
end

####################################################################
abstract type AbstractStateFunctional end

"""
    SingleStateFunctional

    Assume that we have a Functional of the following form:
        F: (u,φ,[dΩ,dΓ]) ↦ ∫_Ω f(u(φ),φ) dΩ + ∫_Γ g(u(φ),φ) dΓ,
    where u is a state field and φ is auxilary. The signature MUST match the weak form.

    The number of states refers to number of solutions to PDEs (e.g., u above).
    We assume that there is only one additional auxilary field φ for purpose of AD.
"""
struct SingleStateFunctional{A,B<:Tuple,C<:Tuple,D<:Tuple} <: AbstractStateFunctional
    F           ::Functional{2}
    dFh         ::A
    spaces      ::B
    assemblers  ::C
    caches      ::D
end

function SingleStateFunctional(F::Functional{2},
        U,
        V,
        V_φ)
    Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv=Vector{PetscScalar}
    trial_assem=SparseMatrixAssembler(Tm,Tv,U,V)
    aux_assem=SparseMatrixAssembler(Tm,Tv,V_φ,V_φ)
    dF_vec = zero(V_φ)
    djdu_vec = zero_free_values(U)
    djdφ_vec = zero_free_values(V_φ)
    dφdu_vec = zero_free_values(V_φ)
    assemblers = (trial_assem,aux_assem)
    spaces = (U,V,V_φ)
    caches = (djdu_vec,djdφ_vec,dφdu_vec)
    SingleStateFunctional(F,dF_vec,spaces,assemblers,caches)
end

(u_to_j::SingleStateFunctional)(_uh,_φh) = u_to_j.F(_uh,_φh)

function ChainRulesCore.rrule(u_to_j::SingleStateFunctional,_uh,_φh)
    F=u_to_j.F
    U,_,V_φ = u_to_j.spaces
    trial_assem,aux_assem = u_to_j.assemblers
    djdu_vec,djdφ_vec,_ = u_to_j.caches
    function u_to_j_pullback(dj)
        djdu = ∇(F,_uh,1)
        djdu_vecdata = collect_cell_vector(U,djdu)
        assemble_vector!(djdu_vec,trial_assem,djdu_vecdata)
        djdφ = ∇(F,_φh,2)
        djdφ_vecdata = collect_cell_vector(V_φ,djdφ)
        assemble_vector!(djdφ_vec,aux_assem,djdφ_vecdata)
        djdu_vec .*= dj
        djdφ_vec .*= dj
        (  NoTangent(), djdu_vec, djdφ_vec )
    end
    u_to_j(_uh,_φh), u_to_j_pullback
end

####################################################################
struct AffineFEStateMap{A<:AbstractStateFunctional,B,C,D,E<:Tuple,F<:Tuple}
    F               ::A
    a               ::B
    l               ::C
    res             ::D
    caches          ::E
    adjoint_caches  ::F
end

function AffineFEStateMap(
        F::A,
        a::B,
        l::C,
        res::D,
        ls::Gridap.Algebra.LinearSolver,
        adjoint_ls::Gridap.Algebra.LinearSolver) where {A<:SingleStateFunctional,B,C,D}
    U,V,_ = F.spaces
    trial_assem,_ = F.assemblers
    
    ## K,b,x
    _,_φh = F.F.state;
    meas = F.F.dΩ
    op = AffineFEOperator((u,v) -> a(u,v,_φh,meas...),v -> l(v,_φh,meas...),U,V,trial_assem)
    K = get_matrix(op); b = get_vector(op); 
    x = get_free_dof_values(zero(U))
    
    ## Adjoint K,b,x
    Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv=Vector{PetscScalar}
    adjoint_assem=SparseMatrixAssembler(Tm,Tv,V,U)
    dv = get_fe_basis(V)
    du = get_trial_fe_basis(U)
    data = collect_cell_matrix_and_vector(V,U,a(du,dv,_φh,meas...),l(dv,_φh,meas...))
    adjoint_K, adjoint_b = assemble_matrix_and_vector(adjoint_assem,data)
    adjoint_x = get_free_dof_values(zero(V))
    
    ## Numerical setups
    ns = numerical_setup(symbolic_setup(ls,K),K)
    adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
    
    ## Caches and adjoint caches
    caches = (ns,K,b,x)
    adjoint_caches = (adjoint_ns,adjoint_K,adjoint_b,adjoint_x,adjoint_assem)
    return AffineFEStateMap(F,a,l,res,caches,adjoint_caches)
end

get_states(m::AffineFEStateMap) = m.F.F.state

evaluate_functional(m::AffineFEStateMap) = m.F.F()

function (φ_to_u::AffineFEStateMap{S} where S<:SingleStateFunctional)(_φh::GridapDistributed.DistributedCellField)
    uh = _eval_φ_to_u(φ_to_u,_φh)
    consistent!(get_free_dof_values(uh)) |> fetch
    uh
end

function (φ_to_u::AffineFEStateMap{S} where S<:SingleStateFunctional)(_φh::FEFunction)
    _eval_φ_to_u(φ_to_u,_φh)
end

function _eval_φ_to_u(φ_to_u::AffineFEStateMap{S},_φh) where S<:SingleStateFunctional
    a=φ_to_u.a
    l=φ_to_u.l
    U,V,_ = φ_to_u.F.spaces
    ns,K,b,x = φ_to_u.caches
    meas = φ_to_u.F.F.dΩ
    trial_assem,_ = φ_to_u.F.assemblers
    uh,_ = φ_to_u.F.F.state
    
    ## Reassemble and solve
    dv = get_fe_basis(V)
    du = get_trial_fe_basis(U)
    data = collect_cell_matrix_and_vector(U,V,a(du,dv,_φh,meas...),l(dv,_φh,meas...))
    assemble_matrix_and_vector!(K,b,trial_assem,data)
    numerical_setup!(ns,K)
    solve!(x,ns,b)
    copy!(get_free_dof_values(uh),x)
    uh
end

function ChainRulesCore.rrule(φ_to_u::AffineFEStateMap{S} where S<:SingleStateFunctional,_φh)
    a=φ_to_u.a
    res=φ_to_u.res
    U,V,_ = φ_to_u.F.spaces
    meas = φ_to_u.F.F.dΩ
    
    ## Forward problem
    _uh = φ_to_u(_φh)
    
    ## Adjoint operator
    _,adjoint_K,adjoint_b,_,adjoint_assem = φ_to_u.adjoint_caches
    dv = get_fe_basis(V)
    du = get_trial_fe_basis(U)
    adjoint_data = collect_cell_matrix(V,U,a(dv,du,_φh,meas...))
    assemble_matrix!(adjoint_K,adjoint_assem,adjoint_data)
    adjoint_op = AffineFEOperator(V,U,adjoint_K,adjoint_b)
    function φ_to_u_pullback(du)
        dφ = Adjoint(_φh,_uh,du,adjoint_op,res,φ_to_u)     
        ( NoTangent(),dφ)
    end
    # get_free_dof_values(_uh), φ_to_u_pullback
    _uh, φ_to_u_pullback
end

function Adjoint(_φh,_uh,du,adjoint_op,res,F::AffineFEStateMap{S} where S<:SingleStateFunctional)
    V = adjoint_op.trial
    _,_,V_φ = F.F.spaces
    adjoint_ns,_,λ,_ = F.adjoint_caches
    
    ## Adjoint Solve
    Aᵀ = Gridap.jacobian(adjoint_op,_uh)
    numerical_setup!(adjoint_ns,Aᵀ)
    solve!(λ,adjoint_ns,du)
    λh = FEFunction(V,λ)
    
    ## Compute grad
    _,_,dφdu_vec = F.F.caches
    _,aux_assem = F.F.assemblers
    res_functional = Functional(res,F.F.F.dΩ,_uh,λh,_φh)
    dφ = ∇(res_functional,_φh,3)
    dφdu_vecdata = collect_cell_vector(V_φ,dφ)
    assemble_vector!(dφdu_vec,aux_assem,dφdu_vecdata)
    dφdu_vec .*= -1;
    dφdu_vec
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

# ## Helpers
# function φ_to_φₕ(φ::AbstractArray,Q)
# 	φ = FEFunction(Q,φ)
# end
# function φ_to_φₕ(φ::FEFunction,Q)
# 	φ
# end
# function φ_to_φₕ(φ::CellField,Q)
# 	φ
# end
# function φ_to_φₕ(φ::GridapDistributed.DistributedCellField,Q)
# 	φ
# end