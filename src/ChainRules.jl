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

        E.g., for `F = (u,ϕ,dΩ) -> ∫(H ∘ ϕ)dΩ`, DF = (q,u,ϕ,dΩ) -> ∫(1*q*(DH ∘ φh)*(norm ∘ ∇(φh)))dΩ
        where `DH(φh)*|∇(φh)|dΩ ≈ dΓ`, `H` is the smoothed Heaviside function, and `dH` is the
        smoothed delta function.        
"""
struct Functional{N} <: AbstractFunctional
    F
    DF
    dΩ::Vector
    state::NTuple{N} # <- can we remove this and define states somewhere else?
    function Functional(F,dΩ::Vector,args...;DF=nothing)
        N = length(args)
        new{N}(F,DF,dΩ,args)
    end
end

Functional(F,dΩ::Measure,args...;DF=nothing) = Functional(F,[dΩ],args...;DF=DF)
Functional(F,dΩ::GridapDistributed.DistributedMeasure,args...;DF=nothing) = Functional(F,[dΩ],args...;DF=DF)

(F::Functional)() = sum(F.F(F.state...,F.dΩ...))
(F::Functional)(args...) = sum(F.F(args...,F.dΩ...))

function Gridap.gradient(F::Functional{N},uh::GridapDistributed.DistributedCellField,K::Int) where N
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

function Gridap.gradient(F::Functional{N},uh::FEFunction,K::Int) where N
    @assert 0<K<=N
    fields = map(i->i==K ? uh : F.state[i],1:N)
    _f = u -> F.F(fields[1:K-1]...,u,fields[K+1:end]...,F.dΩ...)
    return Gridap.Fields.gradient(_f,fields[K])
end

####################################################################
abstract type AbstractStateFunctional end

"""
    SingleStateFunctional

    Assume that we have a Functional of the following form:
        F: (u,ϕ,[dΩ,dΓ]) ↦ ∫_Ω f(u(ϕ),ϕ) dΩ + ∫_Γ g(u(ϕ),ϕ) dΓ,
    where u is a state field and ϕ is auxilary. The signature MUST match the weak form.

    The number of states refers to number of solutions to PDEs (e.g., u above).
    We assume that there is only one additional auxilary field ϕ for purpose of AD.
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
        V_ϕ)
    Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv=Vector{PetscScalar}
    trial_assem=SparseMatrixAssembler(Tm,Tv,U,V)
    aux_assem=SparseMatrixAssembler(Tm,Tv,V_ϕ,V_ϕ)
    dF_vec = zero(V_ϕ)
    djdu_vec = zero_free_values(U)
    djdϕ_vec = zero_free_values(V_ϕ)
    dϕdu_vec = zero_free_values(V_ϕ)
    assemblers = (trial_assem,aux_assem)
    spaces = (U,V,V_ϕ)
    caches = (djdu_vec,djdϕ_vec,dϕdu_vec)
    SingleStateFunctional(F,dF_vec,spaces,assemblers,caches)
end

(u_to_j::SingleStateFunctional)(_uh,_ϕh) = u_to_j.F(_uh,_ϕh)

function ChainRulesCore.rrule(u_to_j::SingleStateFunctional,_uh,_ϕh)
    F=u_to_j.F
    U,_,V_ϕ = u_to_j.spaces
    trial_assem,aux_assem = u_to_j.assemblers
    djdu_vec,djdϕ_vec,_ = u_to_j.caches
    function u_to_j_pullback(dj)
        djdu = ∇(F,_uh,1)
        djdu_vecdata = collect_cell_vector(U,djdu)
        assemble_vector!(djdu_vec,trial_assem,djdu_vecdata)
        djdϕ = ∇(F,_ϕh,2)
        djdϕ_vecdata = collect_cell_vector(V_ϕ,djdϕ)
        assemble_vector!(djdϕ_vec,aux_assem,djdϕ_vecdata)
        djdu_vec .*= dj
        djdϕ_vec .*= dj
        (  NoTangent(), djdu_vec, djdϕ_vec )
    end
    u_to_j(_uh,_ϕh), u_to_j_pullback
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
    _,_ϕh = F.F.state;
    meas = F.F.dΩ
    op = AffineFEOperator(a(_ϕh,meas...),l(_ϕh,meas...),U,V,trial_assem)
    K = get_matrix(op); b = get_vector(op); 
    x = get_free_dof_values(zero(U))
    
    ## Adjoint K,b,x
    Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv=Vector{PetscScalar}
    adjoint_assem=SparseMatrixAssembler(Tm,Tv,V,U)
    dv = get_fe_basis(V)
    du = get_trial_fe_basis(U)
    data = collect_cell_matrix_and_vector(V,U,a(du,dv,_ϕh,meas...),l(dv,_ϕh,meas...))
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

function (ϕ_to_u::AffineFEStateMap{S} where S<:SingleStateFunctional)(_ϕh)
    a=ϕ_to_u.a
    l=ϕ_to_u.l
    U,V,_ = ϕ_to_u.F.spaces
    ns,K,b,x = ϕ_to_u.caches
    meas = ϕ_to_u.F.F.dΩ
    trial_assem,_ = ϕ_to_u.F.assemblers
    uh,_ = ϕ_to_u.F.F.state
    
    ## Reassemble and solve
    dv = get_fe_basis(V)
    du = get_trial_fe_basis(U)
    data = collect_cell_matrix_and_vector(U,V,a(du,dv,_ϕh,meas...),l(dv,_ϕh,meas...))
    assemble_matrix_and_vector!(K,b,trial_assem,data)
    numerical_setup!(ns,K)
    solve!(x,ns,b)
    copy!(get_free_dof_values(uh),x)
    consistent!(get_free_dof_values(uh)) |> fetch
    uh
end

function ChainRulesCore.rrule(ϕ_to_u::AffineFEStateMap{S} where S<:SingleStateFunctional,_ϕh)
    a=ϕ_to_u.a
    res=ϕ_to_u.res
    U,V,_ = ϕ_to_u.F.spaces
    meas = ϕ_to_u.F.F.dΩ
    
    ## Forward problem
    _uh = ϕ_to_u(_ϕh)
    
    ## Adjoint operator
    _,adjoint_K,adjoint_b,_,adjoint_assem = ϕ_to_u.adjoint_caches
    dv = get_fe_basis(V)
    du = get_trial_fe_basis(U)
    adjoint_data = collect_cell_matrix(V,U,a(dv,du,_ϕh,meas...))
    assemble_matrix!(adjoint_K,adjoint_assem,adjoint_data)
    adjoint_op = AffineFEOperator(V,U,adjoint_K,adjoint_b)
    function ϕ_to_u_pullback(du)
        dϕ = Adjoint(_ϕh,_uh,du,adjoint_op,res,ϕ_to_u)     
        ( NoTangent(),dϕ)
    end
    # get_free_dof_values(_uh), ϕ_to_u_pullback
    _uh, ϕ_to_u_pullback
end

function Adjoint(_ϕh,_uh,du,adjoint_op,res,F::AffineFEStateMap{S} where S<:SingleStateFunctional)
    V = adjoint_op.trial
    _,_,V_ϕ = F.F.spaces
    adjoint_ns,_,λ,_ = F.adjoint_caches
    
    ## Adjoint Solve
    Aᵀ = Gridap.jacobian(adjoint_op,_uh)
    numerical_setup!(adjoint_ns,Aᵀ)
    solve!(λ,adjoint_ns,du)
    λh = FEFunction(V,λ)
    
    ## Compute grad
    _,_,dϕdu_vec = F.F.caches
    _,aux_assem = F.F.assemblers
    res_functional = Functional(res,F.F.F.dΩ,_uh,λh,_ϕh)
    dϕ = ∇(res_functional,_ϕh,3)
    dϕdu_vecdata = collect_cell_vector(V_ϕ,dϕ)
    assemble_vector!(dϕdu_vec,aux_assem,dϕdu_vecdata)
    dϕdu_vec .*= -1;
    dϕdu_vec
end

####################################################################
## Shape derivative for AbstractFunctional
function AffineFEStateMap(J::T,C::Vector{T},args...;ls,adjoint_ls) where T<:AbstractFunctional
    @abstractmethod
end

function compute_shape_derivative!(ϕh,state_map::AffineFEStateMap)
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

function compute_shape_derivative!(_ϕh,state_map::AffineFEStateMap{S}) where S<:SingleStateFunctional
    ssfunc,smap = state_map.F,state_map
    _,_,V_φ = ssfunc.spaces
    _,aux_assem = ssfunc.assemblers
    dFh = ssfunc.dFh
    
    if isnothing(ssfunc.F.DF)
        # It doesn't seem resonable that we have to compute the fwd problem for each
        #   shape derivative. I think we can compute new u and objective outside and then
        #   run this without getting u and j below.
        #
        # E.g., we can call ssfunc(state...) to compute j and smap(ϕh) to update uh.

        u, u_pullback = rrule(smap,_ϕh); # Compute fwd problem
        j, j_pullback = rrule(ssfunc,u,_ϕh); # Compute functional
        _, du, dϕ₍ⱼ₎  = j_pullback(1); # Compute derivatives of J wrt to u and φ
        _, dϕ₍ᵤ₎      = u_pullback(du); # Compute adjoint for derivatives of ϕ wrt to u 
        dϕ            = dϕ₍ᵤ₎ + dϕ₍ⱼ₎
        copy!(get_free_dof_values(dFh),dϕ)
        consistent!(get_free_dof_values(dFh)) |> fetch
    else
        meas = ssfunc.F.dΩ
        state = ssfunc.F.state
        _dF = (q) -> ssfunc.F.DF(q,state...,meas...)
        dF_vec = get_free_dof_values(dFh);
        assemble_vector!(_dF,dF_vec,aux_assem,V_φ)
        consistent!(dF_vec) |> fetch
    end
    return dFh
end

## Helpers
function φ_to_φₕ(φ::AbstractArray,Q)
	φ = FEFunction(Q,φ)
end
function φ_to_φₕ(φ::FEFunction,Q)
	φ
end
function φ_to_φₕ(φ::CellField,Q)
	φ
end
function φ_to_φₕ(φ::GridapDistributed.DistributedCellField,Q)
	φ
end