# ==========================================================
# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
# ==========================================================

function Adjoint(ϕ,u,du,op,res,Q)#,solver)
  A = Gridap.jacobian(op,u) # = dr/du
  Aᵀ = adjoint(A) 
  V = op.test
  λₕ = FEFunction(V,Aᵀ\du)#,solver)
  ϕh = FEFunction(Q,ϕ)
  dϕ() = Gridap.FESpaces.gradient(ϕh -> res(u,λₕ,ϕh,Q),ϕh)
  dϕ = -assemble_vector(dϕ(),Q)
end

function AdjointAffine(ϕ,u,du,op,a,Q)#,solver)
  A = Gridap.jacobian(op,u) # = dr/du
  Aᵀ = adjoint(A) 
  V = op.test
  λₕ = FEFunction(V,Aᵀ\du)#,solver)
  ϕh = FEFunction(Q,ϕ)
  dϕ() = Gridap.FESpaces.gradient(ϕh -> res(u,λₕ,ϕh,Q),ϕh)
  dϕ = -assemble_vector(dϕ(),Q)
end

# ================
# ================
# CHAIN RULES MAIN 
# ================
# ================

# ============================================================
# u -> J : Struct representing the scalar objective evaluation
# ============================================================

struct LossFunction{P,U}
  loss::Function
  param_sp::P
  state_sp::U
  # assem::Assembler
end

function (u_to_j::LossFunction)(u,ϕ)
  loss=u_to_j.loss
  U=u_to_j.state_sp
  Q=u_to_j.param_sp
  uₕ=FEFunction(U,u)
  ϕₕ=FEFunction(Q,ϕ)
  sum(loss(uₕ,ϕₕ))
end

function ChainRulesCore.rrule(u_to_j::LossFunction,u,ϕ)
  loss=u_to_j.loss
  U=u_to_j.state_sp
  Q=u_to_j.param_sp
  uₕ=FEFunction(U,u)
  ϕₕ=FEFunction(Q,ϕ)
  jp=sum(loss(uₕ,ϕₕ))
  function u_to_j_pullback(dj)
    djdu = ∇(uₕ->loss(uₕ,ϕₕ))(uₕ)
    djdu_vec = assemble_vector(djdu,U)
    djdϕ = ∇(ϕₕ->loss(uₕ,ϕₕ))(ϕₕ)
    djdϕ_vec = assemble_vector(djdϕ,Q)
    (  NoTangent(), dj*djdu_vec, dj*djdϕ_vec )
  end
  jp, u_to_j_pullback
end
# ==================================================================================================================================
# ϕ -> u : Struct that represents the FE solution of a parameterised Affine PDE. It is a map from the parameter space P to a FESpace
# ==================================================================================================================================

struct _AffineFEStateMap{P,U <: FESpace, V <: FESpace}
	a::Function
	ℓ::Function
  res::Function
	param_sp::P # params (CellData)
	trial::U
	test::V
	# assem::Assembler
end

function (ϕ_to_u::_AffineFEStateMap)(ϕ)
  a=ϕ_to_u.a
  l=ϕ_to_u.ℓ
  res=ϕ_to_u.res
  Q=ϕ_to_u.param_sp
  U=ϕ_to_u.trial
  V=ϕ_to_u.test
  op = AffineFEOperator(a(ϕ),l(ϕ),U,V)
  get_free_dof_values(Gridap.solve(op))
end

function ChainRulesCore.rrule(ϕ_to_u::_AffineFEStateMap,ϕ)
  a=ϕ_to_u.a
  l=ϕ_to_u.ℓ
  res=ϕ_to_u.res
  Q=ϕ_to_u.param_sp
  U=ϕ_to_u.trial
  V=ϕ_to_u.test
  op = AffineFEOperator(a(ϕ),l(ϕ),U,V)
  uₕ = Gridap.solve(op)
  function ϕ_to_u_pullback(du)
      dϕ = Adjoint(ϕ,uₕ,du,op,res,Q)     
      ( NoTangent(),dϕ)
  end
  get_free_dof_values(uₕ), ϕ_to_u_pullback
end

# =========================================================
#  ϕn1 -> ϕb4 : Adding the volume constraint enforcing bias
# ========================================================= 

struct VolumeConstraintMap{P}
  ϕ_b_to_ϕc::Function
  b::Function
  ϕ_b_to_Vol::Function
  param_sp::P
  Vₘₐₓ::Float64
  problem#::ProblemType
end

function (ϕₛ₃_to_ϕ::VolumeConstraintMap)(ϕₛ₃)
  ϕ_b_to_ϕc=ϕₛ₃_to_ϕ.ϕ_b_to_ϕc
  b = ϕₛ₃_to_ϕ.b
  Vbg = ϕₛ₃_to_ϕ.param_sp
  Vₘₐₓ = ϕₛ₃_to_ϕ.Vₘₐₓ 
  problem = ϕₛ₃_to_ϕ.problem

  bp  = b(ϕₛ₃,Vbg,Vₘₐₓ,problem)
  ϕp  = ϕ_b_to_ϕc(ϕₛ₃,bp,problem)
end

function ChainRulesCore.rrule(ϕₛ₃_to_ϕ::VolumeConstraintMap,ϕₛ₃)
  ϕ_b_to_ϕc=ϕₛ₃_to_ϕ.ϕ_b_to_ϕc
  b = ϕₛ₃_to_ϕ.b
  ϕ_b_to_Vol = ϕₛ₃_to_ϕ.ϕ_b_to_Vol
  Vbg = ϕₛ₃_to_ϕ.param_sp
  Vₘₐₓ = ϕₛ₃_to_ϕ.Vₘₐₓ 
  problem = ϕₛ₃_to_ϕ.problem

  bp  = b(ϕₛ₃,Vbg,Vₘₐₓ,problem)
  ϕp  = ϕ_b_to_ϕc(ϕₛ₃,bp,problem)

  ab_r = AD.ReverseDiffBackend()
  pb_ϕₛ₃ = AD.pullback_function(ab_r, ϕₛ₃ -> ϕ_b_to_ϕc(ϕₛ₃,bp,problem) , ϕₛ₃)
  pb_b = AD.pullback_function(ab_r, bp -> ϕ_b_to_ϕc(ϕₛ₃,bp,problem) , bp)

  function ϕᵤ_to_ϕ_pullback(dϕ)
    dVoldϕₛ₃ = ReverseDiff.gradient( ϕ -> ϕ_b_to_Vol(ϕ,bp,Vbg,problem), reshape(ϕₛ₃,(length(ϕₛ₃),1)))  # N X 1 
    dVoldb = ForwardDiff.derivative( b -> ϕ_b_to_Vol(ϕₛ₃,b,Vbg,problem), bp)  # N X 1 
    dϕdb_dϕ  = pb_b(dϕ)[1][1]   #sum(dϕ) # 1 X 1 = ( 1 X N ) * ( N X 1 )
    dϕₛ₃₍ₘ₎ = pb_ϕₛ₃(dϕ)[1] - dVoldϕₛ₃*(dVoldb)^-1 * dϕdb_dϕ  # N X Q X Q X 1 + (N X 1) X (1 X 1) X [(1 X 1)] where the last 1X1 is from the above loop 
    dϕₛ₃ = collect1d(dϕₛ₃₍ₘ₎)
    (NoTangent(), dϕₛ₃)
  end
  ϕp,ϕᵤ_to_ϕ_pullback
end

#=

function (ϕₛ₃_to_ϕ::VolumeConstraintMap)(ϕₛ₃,method::MethodType{:constrained})
  ϕₛ₃
end

function ChainRulesCore.rrule(ϕₛ₃_to_ϕ::VolumeConstraintMap,ϕₛ₃,method::MethodType{:constrained})
  function ϕᵤ_to_ϕ_pullback(dϕ)
    (NoTangent(), dϕ, NoTangent())
  end
  ϕₛ₃,ϕᵤ_to_ϕ_pullback
end

function (ϕₛ₃_to_ϕ::VolumeConstraintMap)(ϕₛ₃,method::MethodType{:unconstrained})
  ϕₛ₃_to_ϕ(ϕₛ₃)
end

function ChainRulesCore.rrule(ϕₛ₃_to_ϕ::VolumeConstraintMap,ϕₛ₃,method::MethodType{:unconstrained})
  ϕₛ₃,ϕᵤ_to_ϕ_pullback =rrule(ϕₛ₃_to_ϕ,ϕₛ₃)
  function ϕᵤ_to_ϕ_pullback_with_extra_NoTangent(dϕ)
    (ϕᵤ_to_ϕ_pullback(dϕ)...,NoTangent())
  end
  ϕₛ₃, ϕᵤ_to_ϕ_pullback_with_extra_NoTangent
end

=#

# =============================================================================================================================================================================================
# ϕᵧ₂ -> ϕₛ₃ : Applying the signed distance reinitialisation. This is another FEStateMap which, as apposed to the Affine version, solves the nonlinear residual using a specified initial point
# =============================================================================================================================================================================================

struct InitialisableFEStateMap{P,U <: FESpace, V <: FESpace}
	res::Function
  jac::Function
	get_geo_params::Function
	param_sp::P # params (CellData)
	trial::U
	test::V
	# assem::Assembler
end

function (ϕᵧ₂_to_ϕₛ₃::InitialisableFEStateMap)(ϕ)
  res=ϕᵧ₂_to_ϕₛ₃.res
  jac=ϕᵧ₂_to_ϕₛ₃.jac
  #get_geo_params_=ϕᵧ₂_to_ϕₛ₃.get_geo_params
  Q=ϕᵧ₂_to_ϕₛ₃.param_sp
  U=ϕᵧ₂_to_ϕₛ₃.trial
  V=ϕᵧ₂_to_ϕₛ₃.test
  #fϕ,_=get_geo_params(ϕ,Q)
  op = FEOperator(res(ϕ),jac(ϕ),U,V)
  ls = LUSolver()
  nls = NLSolver(
          show_trace=true, method=:newton, linesearch=BackTracking(), ftol=1e-12, iterations= 50 )
  solver = FESolver(nls)
  ϕh = FEFunction(V,copy(ϕ))
  ϕₛ₃,_ = Gridap.solve!(ϕh,nls,op)
  ϕₛ₃.free_values
end

function ChainRulesCore.rrule(ϕᵧ₂_to_ϕₛ₃::InitialisableFEStateMap,ϕ)
  res=ϕᵧ₂_to_ϕₛ₃.res
  jac=ϕᵧ₂_to_ϕₛ₃.jac
  #get_geo_params_=ϕᵧ₂_to_ϕₛ₃.get_geo_params
  Q=ϕᵧ₂_to_ϕₛ₃.param_sp
  U=ϕᵧ₂_to_ϕₛ₃.trial
  V=ϕᵧ₂_to_ϕₛ₃.test
  #fϕ,_=get_geo_params(ϕ,Q)
  op = FEOperator(res(ϕ),jac(ϕ),U,V)
  ls = LUSolver()
  nls = NLSolver(
          show_trace=true, method=:newton, linesearch=BackTracking(), ftol=1e-12, iterations= 50 )
  solver = FESolver(nls)
  ϕh = FEFunction(V,copy(ϕ))
  ϕₛ₃,_ = Gridap.solve!(ϕh,nls,op)
  function ϕᵧ₂_to_ϕₛ₃_pullback(dϕₛ₃)
    dϕᵧ₂ = Adjoint(ϕ,ϕₛ₃,dϕₛ₃,op,res,Q)
    (NoTangent(),dϕᵧ₂)
  end
  ϕₛ₃.free_values, ϕᵧ₂_to_ϕₛ₃_pullback
end

#=

function (ϕᵧ₂_to_ϕₛ₃::InitialisableFEStateMap)(ϕᵧ₂,method::MethodType{:unconstrained},problem::ProblemType{:heat_simp})
  ϕᵧ₂
end

function ChainRulesCore.rrule(ϕᵧ₂_to_ϕₛ₃::InitialisableFEStateMap,ϕᵧ₂,method::MethodType{:unconstrained},problem::ProblemType{:heat_simp})
  function ϕᵧ₂_to_ϕₛ₃_pullback(dϕₛ₃)
    ( NoTangent(),dϕₛ₃, NoTangent(), NoTangent())
  end
  ϕᵧ₂, ϕᵧ₂_to_ϕₛ₃_pullback
end

function (ϕᵧ₂_to_ϕₛ₃::InitialisableFEStateMap)(ϕᵧ₂,method::MethodType{:constrained},problem::ProblemType{:heat_simp})
  ϕᵧ₂
end

function ChainRulesCore.rrule(ϕᵧ₂_to_ϕₛ₃::InitialisableFEStateMap,ϕᵧ₂,method::MethodType{:constrained},problem::ProblemType{:heat_simp})
  function ϕᵧ₂_to_ϕₛ₃_pullback(dϕₛ₃)
    ( NoTangent(),dϕₛ₃, NoTangent(), NoTangent())
  end
  ϕᵧ₂, ϕᵧ₂_to_ϕₛ₃_pullback
end

function (ϕᵧ₂_to_ϕₛ₃::InitialisableFEStateMap)(ϕᵧ₂,method::MethodType{:constrained},problem)
  ϕᵧ₂
end

function ChainRulesCore.rrule(ϕᵧ₂_to_ϕₛ₃::InitialisableFEStateMap,ϕᵧ₂,method::MethodType{:constrained},problem)
  function ϕᵧ₂_to_ϕₛ₃_pullback(dϕₛ₃)
    ( NoTangent(),dϕₛ₃, NoTangent(), NoTangent())
  end
  ϕᵧ₂, ϕᵧ₂_to_ϕₛ₃_pullback
end

function (ϕᵧ₂_to_ϕₛ₃::InitialisableFEStateMap)(ϕᵧ₂,method::MethodType{:unconstrained},problem)
  ϕᵧ₂_to_ϕₛ₃(ϕᵧ₂)
end

function ChainRulesCore.rrule(ϕᵧ₂_to_ϕₛ₃::InitialisableFEStateMap,ϕᵧ₂,method::MethodType{:unconstrained},problem)
  ϕₛ₃,ϕᵧ₂_to_ϕₛ₃_pullback = rrule(ϕᵧ₂_to_ϕₛ₃,ϕᵧ₂)
  function ϕᵧ₂_to_ϕₛ₃_pullback_with_extra_NoTangent(dϕₛ₃)
    (ϕᵧ₂_to_ϕₛ₃_pullback(dϕₛ₃)...,NoTangent(), NoTangent())
  end
  ϕₛ₃, ϕᵧ₂_to_ϕₛ₃_pullback_with_extra_NoTangent
end

=#


# =============================================================================================================================================================================================
# An extra Affine InitialisableAffineFEStateMap
# =============================================================================================================================================================================================

struct InitialisableAffineFEStateMap{P,U <: FESpace, V <: FESpace}
	a::Function
  l::Function
  res::Function
	param_sp::P # params (CellData)
	trial::U
	test::V
	# assem::Assembler
end

function (ϕᵧ₂_to_ϕₛ₃::InitialisableAffineFEStateMap)(ϕ)
  a = ϕᵧ₂_to_ϕₛ₃.a
  l = ϕᵧ₂_to_ϕₛ₃.l
  res=ϕᵧ₂_to_ϕₛ₃.res
  Q=ϕᵧ₂_to_ϕₛ₃.param_sp
  U=ϕᵧ₂_to_ϕₛ₃.trial
  V=ϕᵧ₂_to_ϕₛ₃.test
  op = AffineFEOperator(a(ϕ),l(ϕ),U,V)
  ls = LUSolver()
  solver = FESolver(nls)
  ϕh = FEFunction(V,copy(ϕ))
  ϕₛ₃,_ = Gridap.solve!(ϕh,ls,op)
  ϕₛ₃.free_values
end

function ChainRulesCore.rrule(ϕᵧ₂_to_ϕₛ₃::InitialisableAffineFEStateMap,ϕ)
  res=ϕᵧ₂_to_ϕₛ₃.res
  jac=ϕᵧ₂_to_ϕₛ₃.jac
  #get_geo_params_=ϕᵧ₂_to_ϕₛ₃.get_geo_params
  Q=ϕᵧ₂_to_ϕₛ₃.param_sp
  U=ϕᵧ₂_to_ϕₛ₃.trial
  V=ϕᵧ₂_to_ϕₛ₃.test
  #fϕ,_=get_geo_params(ϕ,Q)
  op = AffineFEOperator(a(ϕ),l(ϕ),U,V)
  ls = LUSolver()
  solver = FESolver(nls)
  ϕh = FEFunction(V,copy(ϕ))
  ϕₛ₃,_ = Gridap.solve!(ϕh,ls,op)
  function ϕᵧ₂_to_ϕₛ₃_pullback(dϕₛ₃)
    dϕᵧ₂ = Adjoint(ϕ,ϕₛ₃,dϕₛ₃,op,res,Q)
    (NoTangent(),dϕᵧ₂)
  end
  ϕₛ₃.free_values, ϕᵧ₂_to_ϕₛ₃_pullback
end


# =============================================================================================================================================================================================
# An extra Affine InitialisableFEStateMap that can utilise autodiff but has to use a hack for the part of res not dep on u for the moment
# =============================================================================================================================================================================================

struct InitialisableAutoFEStateMap{P,U <: FESpace, V <: FESpace}
  res::Function
  res_ϕ::Function
	param_sp::P # params (CellData)
	trial::U
	test::V
  u0::Array{Float64}
	# assem::Assembler
end

function (ϕᵧ₂_to_ϕₛ₃::InitialisableAutoFEStateMap)(ϕ)
  res=ϕᵧ₂_to_ϕₛ₃.res
  res_ϕ=ϕᵧ₂_to_ϕₛ₃.res_ϕ
  Q=ϕᵧ₂_to_ϕₛ₃.param_sp
  U=ϕᵧ₂_to_ϕₛ₃.trial
  V=ϕᵧ₂_to_ϕₛ₃.test
  u0 = ϕᵧ₂_to_ϕₛ₃.u0
  op = FEOperator(res(ϕ),U,V)
  #ls = LUSolver()
  nls = NLSolver(
          show_trace=true, method=:newton, linesearch=BackTracking(), ftol=1e-9, iterations= 50 )
  solver = FESolver(nls)
  ϕₛ₃0h = FEFunction(V,copy(u0))
  ϕₛ₃,_ = Gridap.solve!(ϕₛ₃0h,nls,op)
  ϕₛ₃.free_values
end

function ChainRulesCore.rrule(ϕᵧ₂_to_ϕₛ₃::InitialisableAutoFEStateMap,ϕ)
  res=ϕᵧ₂_to_ϕₛ₃.res
  res_ϕ=ϕᵧ₂_to_ϕₛ₃.res_ϕ
  Q=ϕᵧ₂_to_ϕₛ₃.param_sp
  U=ϕᵧ₂_to_ϕₛ₃.trial
  V=ϕᵧ₂_to_ϕₛ₃.test
  u0 = ϕᵧ₂_to_ϕₛ₃.u0
  op = FEOperator(res(ϕ),U,V)
  #ls = LUSolver()
  nls = NLSolver(
          show_trace=true, method=:newton, linesearch=BackTracking(), ftol=1e-9, iterations= 50 )
  solver = FESolver(nls)
  ϕh = FEFunction(V,copy(u0))
  ϕₛ₃,_ = Gridap.solve!(ϕh,nls,op)
  function ϕᵧ₂_to_ϕₛ₃_pullback(dϕₛ₃)
    dϕᵧ₂ = Adjoint(ϕ,ϕₛ₃,dϕₛ₃,op,res_ϕ,Q)
    (NoTangent(),dϕᵧ₂)
  end
  ϕₛ₃.free_values, ϕᵧ₂_to_ϕₛ₃_pullback
end

# =========================================================
# ϕₙ₁ -> ϕᵧ₂ : Smoothing the level set with a linear filter 
# =========================================================

struct LinearFilter
  bgmodel::CartesianDiscreteModel
  filter_weights::Matrix{Float64}
  pullback_jacobian::Function   # This is a constant matrix : we keep it as a pullback to avoid assembling 
end

function (ϕₙ₁_to_ϕᵧ₂::LinearFilter)(ϕₙ₁)
  bgmodel=ϕₙ₁_to_ϕᵧ₂.bgmodel
  filter_weights = ϕₙ₁_to_ϕᵧ₂.filter_weights
  ϕᵧ₂ =  apply_filter(filter_weights,bgmodel,ϕₙ₁)
end

function ChainRulesCore.rrule(ϕₙ₁_to_ϕᵧ₂::LinearFilter,ϕₙ₁)
  bgmodel=ϕₙ₁_to_ϕᵧ₂.bgmodel
  filter_weights = ϕₙ₁_to_ϕᵧ₂.filter_weights
  pb_f = ϕₙ₁_to_ϕᵧ₂.pullback_jacobian
  ϕᵧ₂ =  apply_filter(filter_weights,bgmodel,ϕₙ₁)
  function ϕₙ₁_to_ϕᵧ₂_pullback(dϕᵧ₂)
    dϕₙ₁M = pb_f(dϕᵧ₂)[1]
    cd = Gridap.Geometry.get_cartesian_descriptor(bgmodel)
    cells = cd.partition
    dϕₙ₁ = collect(reshape(dϕₙ₁M,(cells[1]+1)*(cells[2]+1)))
    (NoTangent(),dϕₙ₁)
  end
  ϕᵧ₂,ϕₙ₁_to_ϕᵧ₂_pullback
end

function apply_filter(filter_weights,bgmodel,ϕₙ₁)
  cd = Gridap.Geometry.get_cartesian_descriptor(bgmodel)
  cells = cd.partition
  ϕₙ₁M=reshape(ϕₙ₁,(cells[1]+1,cells[2]+1))
  w = centered(filter_weights)
  ϕᵧ₂M = imfilter(ϕₙ₁M,w,"replicate")  
  ϕᵧ₂ = reshape(ϕᵧ₂M,(cells[1]+1)*(cells[2]+1))
end

# ===============================================================================================================================================================================================
#  p -> ϕn1 : A map from the initial parameterisation to level set values. The initial parameterisation may be a neural network or simply pixel based (in which case this map is the identity).
# =============================================================================================================================================================================================== 

# We also include here a tape for mapping p to L where L is the sub-objective for matching a level set to a desired level set. This is used to train a neural network to provide a specified initial guess (defined by a set of level set values). 
struct NeuralObjective
  N0::Vector{Float64}
end

function (ϕₙ₁_to_L::NeuralObjective)(ϕₙ₁)
  N0 = ϕₙ₁_to_L.N0
  sum( (N0.-ϕₙ₁).^2 ) / length(N0)
end

function ChainRulesCore.rrule(ϕₙ₁_to_L::NeuralObjective,ϕₙ₁)
  function ϕₙ₁_to_L_pullback(dL)
    dϕₙ₁ = ForwardDiff.gradient(ϕₙ₁_to_L,ϕₙ₁) * dL
    ( NoTangent(), dϕₙ₁)
  end
  ϕₙ₁_to_L(ϕₙ₁), ϕₙ₁_to_L_pullback
end

struct NeuralGeometry
  N::Function
  pP::Vector{Float64}
end

function (p_to_ϕₙ₁::NeuralGeometry)(p)
  N = p_to_ϕₙ₁.N
  pP = p_to_ϕₙ₁.pP
  Nh =  N(p) # nucleation_promotion(N(p),pP) #  
  Np = collect(Iterators.flatten(Nh))
end

function ChainRulesCore.rrule(p_to_ϕₙ₁::NeuralGeometry,p)
  N = p_to_ϕₙ₁.N
  pP = p_to_ϕₙ₁.pP
  Nh, dNdp_vjp =  Zygote.pullback(N,p)  # Zygote.pullback(p->nucleation_promotion(N(p),pP),p) # 
  Np = collect(Iterators.flatten(Nh))
  function p_to_ϕₙ₁_pullback(ds)
    dp = dNdp_vjp(Float32.(ds))[1]
    ( NoTangent(),dp )
  end
  Np, p_to_ϕₙ₁_pullback
end

#=

function (p_to_ϕₙ₁::NeuralGeometry)(p,prior::PriorType{:pixel})
  p
end

function ChainRulesCore.rrule(p_to_ϕₙ₁::NeuralGeometry,p,prior::PriorType{:pixel})
  function p_to_ϕₙ₁_pullback(dϕₙ₁)
    dp=dϕₙ₁
    ( NoTangent(),dp, NoTangent())
  end
  p_to_ϕₙ₁(p,prior), p_to_ϕₙ₁_pullback
end

function (p_to_ϕₙ₁::NeuralGeometry)(p,prior::PriorType{:neural})
  p_to_ϕₙ₁(p)
end

function ChainRulesCore.rrule(p_to_ϕₙ₁::NeuralGeometry,p,prior::PriorType{:neural})
  ϕₙ₁, p_to_ϕₙ₁_pullback = rrule(p_to_ϕₙ₁,p)
  function p_to_ϕₙ₁_pullbackrrule_with_extra_NoTangent(dϕₙ₁)
    (p_to_ϕₙ₁_pullback(dϕₙ₁)..., NoTangent())
  end
  ϕₙ₁, p_to_ϕₙ₁_pullbackrrule_with_extra_NoTangent
end

function p_to_L(p,bg_params) #prior)
    ϕₙ₁   = p_to_ϕₙ₁(p,bg_params) #prior)
    L   = ϕₙ₁_to_L(ϕₙ₁,bg_params)     # unfiltered unconstrained nodal values (s) to filtered unconstrained values (ϕᵤ)
    L
end

p_to_L(prior) = p -> p_to_L(p,prior)

=#

# ==========================================================
# ==========================================================
# CHAIN RULES CONSTRAINTS ( FOR USE WITH MMA )
# ==========================================================
# ==========================================================

struct VolumeMap{V<:FESpace}
  ϕc_to_Vol::Function
  param_space::V
  problem#::ProblemType
end

function (ϕ_to_Vol::VolumeMap)(ϕ)
  ϕc_to_Vol=ϕ_to_Vol.ϕc_to_Vol
  Vbg=ϕ_to_Vol.param_space
  problem = ϕ_to_Vol.problem

  ϕc_to_Vol(ϕ,Vbg,problem)
end

function ChainRulesCore.rrule(ϕ_to_Vol::VolumeMap,ϕ)
  ϕc_to_Vol=ϕ_to_Vol.ϕc_to_Vol
  Vbg=ϕ_to_Vol.param_space
  problem = ϕ_to_Vol.problem

  Vol0=ϕc_to_Vol(ϕ,Vbg,problem)
  
  function ϕ_to_Vol_pullback(dVol)
    dVoldϕ = ReverseDiff.gradient(ϕ -> ϕc_to_Vol(ϕ,Vbg,problem),reshape(ϕ,(length(ϕ),1)))
    dϕₘ = dVol*dVoldϕ
    dϕ = collect1d(dϕₘ)
    ( NoTangent(),dϕ )
  end
  Vol0, ϕ_to_Vol_pullback
end

struct UnstructuredVolumeMap{V<:FESpace}
  unstructured_ϕc_to_Vol::Function
  param_space::V
end

function (ϕ_to_Vol::UnstructuredVolumeMap)(ϕ)
  unstructured_ϕc_to_Vol=ϕ_to_Vol.unstructured_ϕc_to_Vol
  Qf=ϕ_to_Vol.param_space

  ϕh = FEFunction(Qf,ϕ)

  sum(unstructured_ϕc_to_Vol(ϕh))
end

function ChainRulesCore.rrule(ϕ_to_Vol::UnstructuredVolumeMap,ϕ)
  unstructured_ϕc_to_Vol=ϕ_to_Vol.unstructured_ϕc_to_Vol
  Qf=ϕ_to_Vol.param_space

  ϕh = FEFunction(Qf,ϕ)

  Vol0=sum(unstructured_ϕc_to_Vol(ϕh))
  
  function ϕ_to_Vol_pullback(dVol)
    dVoldϕ() = Gridap.FESpaces.gradient(ϕh -> unstructured_ϕc_to_Vol(ϕh),ϕh)  #ReverseDiff.gradient(ϕ -> ϕc_to_Vol(ϕ,Vbg,problem),reshape(ϕ,(length(ϕ),1)))
    dVoldϕ = assemble_vector(dVoldϕ(),Qf)
    dϕ = dVol*dVoldϕ
    ( NoTangent(),dϕ )
  end
  Vol0, ϕ_to_Vol_pullback
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