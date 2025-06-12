using Test

using GridapTopOpt
using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.ODEs
using GridapEmbedded

function update_reuse!(state,reuse_new,op;zero_tF=false)
  U, (tF, stateF, state0, uF, odecache) = state
  odeslvrcache, odeopcache = odecache
  _, ui_pre, slopes, J, r, sysslvrcaches = odeslvrcache

  data = allocate_odecache(ode_solver,ODEOpFromTFEOp(op),tF,stateF)
  odeslvrcache_new = (reuse_new, ui_pre, slopes, J, r, data[1][end])
  odecache_new = odeslvrcache_new, odeopcache
  _tF = zero_tF ? 0.0 : tF
  return U, (_tF, stateF, state0, uF, odecache_new)
end

order = 1
n = 50
_model = CartesianDiscreteModel((0,1,0,1),(n,n))
cd = Gridap.Geometry.get_cartesian_descriptor(_model)
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = ref_model.model
h = maximum(cd.sizes)
steps = 10

reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe_scalar)
φh = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)+0.25,V_φ)
velh = interpolate(x->-1,V_φ)

## Trians

Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)
Γg = SkeletonTriangulation(Ω)
dΓg = Measure(Γg,2order)
n_Γg = get_normal_vector(Γg)

## ODE Solver
ode_ls = LUSolver()
ode_nl = NLSolver(ode_ls, show_trace=false, method=:newton, iterations=10)
ode_solver = GridapTopOpt.MutableRungeKutta(ode_nl, ode_ls, 0.1*h, :DIRK_CrankNicolson_2_2)

## ODE Op
v_norm = maximum(abs,get_free_dof_values(velh))
β(vh,∇φ) = vh/(1e-20 + v_norm) * ∇φ/(1e-20 + norm(∇φ))
stiffness(t,u,v) = ∫(((β ∘ (velh,∇(φh))) ⋅ ∇(u)) * v)dΩ + ∫(0.1*h^2*jump(∇(u) ⋅ n_Γg)*jump(∇(v) ⋅ n_Γg))dΓg
mass(t, ∂ₜu, v) = ∫(∂ₜu * v)dΩ
forcing(t,v) = ∫(0v)dΩ + ∫(0*jump(∇(v) ⋅ n_Γg))dΓg
Ut_φ = TransientTrialFESpace(V_φ)

# Both forms constant
op = TransientLinearFEOperator((stiffness,mass),forcing,Ut_φ,V_φ;
  constant_forms=(true,true))
ode_sol = ODEs.solve(ode_solver,op,0.0,ode_solver.dt*steps,φh)
data0, state0 = Base.iterate(ode_sol)
data1, state1 = Base.iterate(ode_sol,state0)

# # Update φh and velh
# φh_new = FEFunction(V_φ,copy(get_free_dof_values(data1[2])))
# velh_new = interpolate(x->-2,V_φ)
# v_norm_new = maximum(abs,get_free_dof_values(velh_new))
# β_new(vh,∇φ) = vh/(1e-20 + v_norm_new) * ∇φ/(1e-20 + norm(∇φ))
# stiffness_new(t,u,v) = ∫(((β_new ∘ (velh,∇(φh))) ⋅ ∇(u)) * v)dΩ + ∫(0.1*h^2*jump(∇(u) ⋅ n_Γg)*jump(∇(v) ⋅ n_Γg))dΓg
# op = TransientLinearFEOperator((stiffness_new,mass),forcing,Ut_φ,V_φ;
#   constant_forms=(true,true))
# ode_sol_new = ODEs.solve(ode_solver,op,0.0,ode_solver.dt*steps,φh_new)
# data2, state2 = Base.iterate(ode_sol)
# data3, state3 = Base.iterate(ode_sol,state2)

# first form non-constant
nc_op = TransientLinearFEOperator((stiffness,mass),forcing,Ut_φ,V_φ;
  constant_forms=(false,true))
nc_ode_sol = ODEs.solve(ode_solver,nc_op,0.0,ode_solver.dt*steps,φh)
nc_data0, nc_state0 = Base.iterate(nc_ode_sol)
nc_data1, nc_state1 = Base.iterate(nc_ode_sol,nc_state0)

data1[1] == nc_data1[1]
norm(get_free_dof_values(data1[2]) - get_free_dof_values(nc_data1[2]),Inf)

# first form non-constant then switch to constant
s_op = TransientLinearFEOperator((stiffness,mass),forcing,Ut_φ,V_φ;
  constant_forms=(false,true))
s_ode_sol = ODEs.solve(ode_solver,s_op,0.0,ode_solver.dt*steps,φh)
s_data0, s_state0 = Base.iterate(s_ode_sol)
s_state0_new = update_reuse!(s_state0,true)
s_data1, s_state1 = Base.iterate(s_ode_sol,s_state0_new)

data1[1] == s_data1[1]
norm(get_free_dof_values(data1[2]) - get_free_dof_values(s_data1[2]),Inf)

########################################################
using Gridap.FESpaces, Gridap.Polynomials
using LinearAlgebra
# TrianientLinearFEOpFromWeakForm2
struct TransientLinearFEOpFromWeakForm2 <: TransientFEOperator{LinearODE}
  forms::Tuple{Vararg{Function}}
  res::Function
  jacs::Tuple{Vararg{Function}}
  constant_forms::BitVector
  assembler::Assembler
  trial::FESpace
  test::FESpace
  order::Integer
end

# Constructor with manual jacobians
function TransientLinearFEOperator2(
  forms::Tuple{Vararg{Function}}, res::Function, jacs::Tuple{Vararg{Function}},
  trial, test;
  constant_forms::BitVector=falses(length(forms)),
  assembler=SparseMatrixAssembler(trial, test)
)
  order = length(jacs) - 1
  TransientLinearFEOpFromWeakForm2(
    forms, res, jacs, constant_forms,
    assembler, trial, test, order
  )
end

# No constructor with flat arguments: would clash with the constructors
# below with flat forms and automatic jacobians, which are more useful

# Constructor with automatic jacobians
function TransientLinearFEOperator2(
  forms::Tuple{Vararg{Function}}, res::Function,
  trial, test;
  constant_forms::BitVector=falses(length(forms)),
  assembler=SparseMatrixAssembler(trial, test)
)
  # When the operator is linear, the jacobians are the forms themselves
  order = length(forms) - 1
  jacs = ntuple(k -> ((t, u, duk, v) -> forms[k](t, duk, v)), order + 1)

  TransientLinearFEOperator2(
    forms, res, jacs, trial, test;
    constant_forms, assembler
  )
end

# Constructor with flat forms and automatic jacobians (orders 0, 1, 2)
function TransientLinearFEOperator2(
  mass::Function, res::Function,
  trial, test;
  constant_forms::BitVector=falses(1),
  assembler=SparseMatrixAssembler(trial, test)
)
  TransientLinearFEOperator2(
    (mass,), res, trial, test;
    constant_forms, assembler
  )
end

function TransientLinearFEOperator2(
  stiffness::Function, mass::Function, res::Function,
  trial, test;
  constant_forms::BitVector=falses(2),
  assembler=SparseMatrixAssembler(trial, test)
)
  TransientLinearFEOperator2(
    (stiffness, mass), res, trial, test;
    constant_forms, assembler
  )
end

function TransientLinearFEOperator2(
  stiffness::Function, damping::Function, mass::Function, res::Function,
  trial, test;
  constant_forms::BitVector=falses(3),
  assembler=SparseMatrixAssembler(trial, test)
)
  TransientLinearFEOpFromWeakForm2(
    (stiffness, damping, mass), res, trial, test;
    constant_forms, assembler
  )
end

# TransientFEOperator interface
FESpaces.get_test(tfeop::TransientLinearFEOpFromWeakForm2) = tfeop.test

FESpaces.get_trial(tfeop::TransientLinearFEOpFromWeakForm2) = tfeop.trial

Polynomials.get_order(tfeop::TransientLinearFEOpFromWeakForm2) = tfeop.order

ODEs.get_res(tfeop::TransientLinearFEOpFromWeakForm2) = (t, u, v) -> tfeop.res(t, v)

ODEs.get_jacs(tfeop::TransientLinearFEOpFromWeakForm2) = tfeop.jacs

ODEs.get_forms(tfeop::TransientLinearFEOpFromWeakForm2) = tfeop.forms

function ODEs.is_form_constant(tfeop::TransientLinearFEOpFromWeakForm2, k::Integer)
  tfeop.constant_forms[k+1]
end

ODEs.get_assembler(tfeop::TransientLinearFEOpFromWeakForm2) = tfeop.assembler

#######
ss_op = TransientLinearFEOperator2((stiffness,mass),forcing,Ut_φ,V_φ;
  constant_forms=BitVector((false,true)))
ss_ode_sol = ODEs.solve(ode_solver,ss_op,0.0,ode_solver.dt*steps,φh)
ss_data0, ss_state0 = Base.iterate(ss_ode_sol)

_t = ss_data0[1]
stiff_const_form = copy(ss_state0[2][5][1][4]) # <- cache this guy

#...

us = (get_free_dof_values(ss_data0[2]),get_free_dof_values(ss_data0[2]))
order = get_order(ss_op)
Ut = get_trial(ss_op)
# U = allocate_space(Ut)
# Uts = (Ut,)
# Us = (U,)
# for k in 1:order
#   Uts = (Uts..., ∂t(Uts[k]))
#   Us = (Us..., allocate_space(Uts[k+1]))
# end

_odeopcache = ss_state0[end][end][end]
Us = ()
for k in 0:get_order(ss_op)
  Us = (Us..., evaluate!(_odeopcache.Us[k+1], _odeopcache.Uts[k+1], _t))
end

uh = ODEs._make_uh_from_us(ss_op, us, Us)
du = get_trial_fe_basis(Ut_φ)
V = get_test(ss_op)
v = get_fe_basis(V)

jacs = ODEs.get_jacs(ss_op)
jac = jacs[1]
dc = jac(_t, uh, du, v)
matdata = collect_cell_matrix(Ut, V, dc)
LinearAlgebra.fillstored!(stiff_const_form, zero(eltype(stiff_const_form)))
assemble_matrix_add!(stiff_const_form, get_assembler(ss_op), matdata)

## Update
new_const_forms = (stiff_const_form,ss_state0[end][end][end].const_forms[2])
ss_state0[end][end][end].const_forms = new_const_forms

ss_state0_new = update_reuse!(ss_state0,true)
ss_op.constant_forms[1] = true
ss_data1, ss_state1 = Base.iterate(ss_ode_sol,ss_state0_new)

data1[1] == ss_data1[1]
norm(get_free_dof_values(data1[2]) - get_free_dof_values(ss_data1[2]),Inf)

