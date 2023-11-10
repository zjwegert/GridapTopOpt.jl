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
        return Gridap.Fields.gradient(_f,lf[K]) # <- A lot of allocations from Gridap AD
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

DFESpace = GridapDistributed.DistributedFESpace

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
    aux_space::Vϕ
    trial_assem::Assembler
    aux_assem::Assembler
    djdu_vec
    djdϕ_vec
    dϕdu_vec
    function SingleStateFunctional(F::Functional{2},trial_space::U,test_space::V,aux_space::Vϕ) where {
            U<:DFESpace,V<:DFESpace,Vϕ<:DFESpace}
        Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
        Tv=Vector{PetscScalar}
        trial_assem=SparseMatrixAssembler(Tm,Tv,trial_space,test_space)
        aux_assem=SparseMatrixAssembler(Tm,Tv,aux_space,aux_space)
        djdu_vec = pfill(PetscScalar(0.0),partition(trial_space.gids));
        djdϕ_vec = pfill(PetscScalar(0.0),partition(aux_space.gids));
        dϕdu_vec = pfill(PetscScalar(0.0),partition(aux_space.gids));
        new{U,V,Vϕ}(F,trial_space,test_space,aux_space,trial_assem,aux_assem,djdu_vec,djdϕ_vec,dϕdu_vec)
    end
    function SingleStateFunctional(F::Functional{2},trial_space::U,test_space::V,aux_space::Vϕ) where {
            U<:FESpace,V<:FESpace,Vϕ<:FESpace}
        Tm=SparseMatrixCSR{0,PetscScalar,Int64}
        Tv=Vector{PetscScalar}
        trial_assem=SparseMatrixAssembler(Tm,Tv,trial_space,test_space)
        aux_assem=SparseMatrixAssembler(Tm,Tv,aux_space,aux_space)
        djdu_vec = zero_free_values(trial_space);
        djdϕ_vec = zero_free_values(aux_space);
        dϕdu_vec = zero_free_values(aux_space);
        new{U,V,Vϕ}(F,trial_space,test_space,aux_space,trial_assem,aux_assem,djdu_vec,djdϕ_vec,dϕdu_vec)
    end
end

function (u_to_j::SingleStateFunctional)(_uh,_ϕh)
    F=u_to_j.F
    sum(F.F(_uh,_ϕh,F.dΩ...))
end

function ChainRulesCore.rrule(u_to_j::SingleStateFunctional,_uh,_ϕh)
    F=u_to_j.F
    U=u_to_j.trial_space
    V_ϕ=u_to_j.aux_space
    trial_assem = u_to_j.trial_assem
    aux_assem = u_to_j.aux_assem
    djdu_vec = u_to_j.djdu_vec
    djdϕ_vec = u_to_j.djdϕ_vec
    function u_to_j_pullback(dj)
        djdu = FunctionalGradient(F,1)(_uh)
        djdu_vecdata = collect_cell_vector(U,djdu)
        assemble_vector!(djdu_vec,trial_assem,djdu_vecdata)
        djdϕ = FunctionalGradient(F,2)(_ϕh)
        djdϕ_vecdata = collect_cell_vector(V_ϕ,djdϕ)
        assemble_vector!(djdϕ_vec,aux_assem,djdϕ_vecdata)
        djdu_vec .*= dj
        djdϕ_vec .*= dj
        (  NoTangent(), djdu_vec, djdϕ_vec )
    end
    u_to_j(_uh,_ϕh), u_to_j_pullback
end

struct AffineFEStateMap{S}
    a::Function
    l::Function
    res::Function
    F::S
    ns
    adjoint_ns
    adjoint_assem
    K
    b
    x
    adjoint_K
    adjoint_b
    adjoint_x
    function AffineFEStateMap(ϕh,a::Function,l::Function,res::Function,F::S;
            ls = PETScLinearSolver(),adjoint_ls = PETScLinearSolver()) where {
            S<:SingleStateFunctional{U} where U<:DFESpace}
        ## This should only happen during setup!!
        # K,b,x
        meas = F.F.dΩ
        U = F.trial_space; V = F.test_space;
        assem = F.trial_assem
        op = AffineFEOperator(a(ϕh,meas...),l(ϕh,meas...),U,V,assem)
        K = get_matrix(op); b = get_vector(op); 
        x = pfill(PetscScalar(0.0),partition(axes(K,2)))
        # Adjoint K,b,x
        Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
        Tv=Vector{PetscScalar}
        adjoint_assem=SparseMatrixAssembler(Tm,Tv,V,U)
        adjoint_op = AffineFEOperator(a(ϕh,meas...),l(ϕh,meas...),V,U,adjoint_assem)
        adjoint_K = get_matrix(adjoint_op); adjoint_b = get_vector(adjoint_op); 
        adjoint_x = pfill(PetscScalar(0.0),partition(axes(adjoint_K,2)))
        # NSs
        ns = numerical_setup(symbolic_setup(ls,K),K)
        adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
        new{S}(a,l,res,F,ns,adjoint_ns,adjoint_assem,K,b,x,adjoint_K,adjoint_b,adjoint_x)
    end
    function AffineFEStateMap(ϕh,a::Function,l::Function,res::Function,F::S;
            ls = PETScLinearSolver(),adjoint_ls = PETScLinearSolver()) where {
            S<:SingleStateFunctional{U} where U<:FESpace}
        ## This should only happen during setup!!
        # K,b,x
        meas = F.F.dΩ
        U = F.trial_space; V = F.test_space;
        assem = F.trial_assem
        op = AffineFEOperator(a(ϕh,meas...),l(ϕh,meas...),U,V,assem)
        K = get_matrix(op); b = get_vector(op); 
        x = zero(b)
        # Adjoint K,b,x
        Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
        Tv=Vector{PetscScalar}
        adjoint_assem=SparseMatrixAssembler(Tm,Tv,V,U)
        dv = get_fe_basis(V)
        du = get_trial_fe_basis(U)
        data = collect_cell_matrix_and_vector(V,U,a(du,dv,ϕh,meas...),l(dv,ϕh,meas...))
        adjoint_K, adjoint_b = assemble_matrix_and_vector(adjoint_assem,data)
        adjoint_x = zero(b)
        # NSs
        ns = numerical_setup(symbolic_setup(ls,K),K)
        adjoint_ns = numerical_setup(symbolic_setup(adjoint_ls,adjoint_K),adjoint_K)
        new{S}(a,l,res,F,ns,adjoint_ns,adjoint_assem,K,b,x,adjoint_K,adjoint_b,adjoint_x)
    end
end

function (ϕ_to_u::AffineFEStateMap{S} where S<:SingleStateFunctional)(_ϕh)
    # Forms
    a=ϕ_to_u.a
    l=ϕ_to_u.l
    res=ϕ_to_u.res
    # Spaces and solver
    U=ϕ_to_u.F.trial_space
    V=ϕ_to_u.F.test_space
    ns = ϕ_to_u.ns
    # Measures and assembler
    meas = ϕ_to_u.F.F.dΩ
    trial_assem = ϕ_to_u.F.trial_assem
    # Cache
    K = ϕ_to_u.K
    b = ϕ_to_u.b
    x = ϕ_to_u.x    
    dv = get_fe_basis(V)
    du = get_trial_fe_basis(U)
    data = collect_cell_matrix_and_vector(U,V,a(du,dv,_ϕh,meas...),l(dv,_ϕh,meas...))
    assemble_matrix_and_vector!(K,b,trial_assem,data)
    numerical_setup!(ns,K)
    solve!(x,ns,b)
    FEFunction(U,x)
end

function ChainRulesCore.rrule(ϕ_to_u::AffineFEStateMap{S} where S<:SingleStateFunctional,_ϕh)
    a=ϕ_to_u.a
    l=ϕ_to_u.l
    res=ϕ_to_u.res
    V_ϕ=ϕ_to_u.F.aux_space
    U=ϕ_to_u.F.trial_space
    V=ϕ_to_u.F.test_space
    meas = ϕ_to_u.F.F.dΩ
    # Forward problem
    _uh = ϕ_to_u(_ϕh)
    # Adjoint operator
    adjoint_assem = ϕ_to_u.adjoint_assem
    adjoint_K = ϕ_to_u.adjoint_K
    adjoint_b = ϕ_to_u.adjoint_b # <- This can always be a zero vector? Only need to form AffineFEOperator
    dv = get_fe_basis(V)
    du = get_trial_fe_basis(U)
    adjoint_data = collect_cell_matrix(V,U,a(dv,du,_ϕh,meas...))
    assemble_matrix!(adjoint_K,adjoint_assem,adjoint_data)
    # numerical_setup!(ϕ_to_u.adjoint_ns,adjoint_K) # <- This happens in Adjoint
    adjoint_op = AffineFEOperator(V,U,adjoint_K,adjoint_b)
    function ϕ_to_u_pullback(du)
        dϕ = Adjoint(_ϕh,_uh,du,adjoint_op,res,ϕ_to_u)     
        ( NoTangent(),dϕ)
    end
    get_free_dof_values(_uh), ϕ_to_u_pullback
end

function Adjoint(_ϕh,_uh,du,adjoint_op,res,F::AffineFEStateMap{S} where S<:SingleStateFunctional)
    V_ϕ = F.F.aux_space
    Aᵀ = Gridap.jacobian(adjoint_op,_uh)
    adjoint_ns = F.adjoint_ns
    numerical_setup!(adjoint_ns,Aᵀ)
    V = adjoint_op.trial
    # Adjoint solve
    λ = F.adjoint_x
    solve!(λ,adjoint_ns,du)
    λh = FEFunction(V,λ)
    # Compute grad
    dϕdu_vec = F.F.dϕdu_vec;
    aux_assem = F.F.aux_assem
    res_functional = Functional(res,F.F.F.dΩ,_uh,λh,_ϕh)
    dϕ() = FunctionalGradient(res_functional,3)(_ϕh)
    dϕdu_vecdata = collect_cell_vector(V_ϕ,dϕ())
    assemble_vector!(dϕdu_vec,aux_assem,dϕdu_vecdata)
    dϕdu_vec .*= -1;
    dϕdu_vec
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