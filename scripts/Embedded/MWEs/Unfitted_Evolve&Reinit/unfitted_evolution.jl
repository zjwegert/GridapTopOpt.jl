struct UnfittedFEEvolution <: GridapTopOpt.LevelSetEvolution
  spaces
  params
  ode_solver
  reinit_nls
  function UnfittedFEEvolution(Ut_φ, V_φ, dΩ, h;
      ode_ls = LUSolver(),
      ode_nl = NLSolver(ode_ls, show_trace=true, method=:newton, iterations=10),
      ode_solver = RungeKutta(ode_nl, ode_ls, 0.01, :DIRK_CrankNicolson_2_2),
      NT = 10,
      c = 0.1,
      reinit_nls = NLSolver(ftol=1e-14, iterations = 50, show_trace=true)
    )
    spaces = (Ut_φ,V_φ)
    params = (;NT,h,c,dΩ)
    return new(spaces,params,ode_solver,reinit_nls)
  end
end

function GridapTopOpt.evolve!(s::UnfittedFEEvolution,φ::AbstractArray,args...)
  _, V_φ = s.spaces
  evolve!(s,FEFunction(V_φ,φ),args...)
end

# Based on:
# @article{
#   Burman_Elfverson_Hansbo_Larson_Larsson_2018,
#   title={Shape optimization using the cut finite element method},
#   volume={328},
#   ISSN={00457825},
#   DOI={10.1016/j.cma.2017.09.005},
#   journal={Computer Methods in Applied Mechanics and Engineering},
#   author={Burman, Erik and Elfverson, Daniel and Hansbo, Peter and Larson, Mats G. and Larsson, Karl},
#   year={2018},
#   month=jan,
#   pages={242–261},
#   language={en}
# }
function GridapTopOpt.evolve!(s::UnfittedFEEvolution,φh,vel,γ)
  Ut_φ, V_φ = s.spaces
  params = s.params
  NT, h, c, dΩ = params.NT, params.h, params.c, params.dΩ
  # ode_solver = s.ode_solver
  Tf = γ*NT
  velh = FEFunction(V_φ,vel)

  # This is temp as can't update γ in ode_solver yet
  ode_ls = LUSolver()
  ode_nl = NLSolver(ode_ls, show_trace=true, method=:newton, iterations=10)
  ode_solver = RungeKutta(ode_nl, ode_ls, γ, :DIRK_CrankNicolson_2_2)

  ϵ = 1e-20

  geo = DiscreteGeometry(φh,model)
  F = EmbeddedFacetDiscretization(LevelSetCutter(),model,geo)
  FΓ = SkeletonTriangulation(F)
  dFΓ = Measure(FΓ,2*order)
  n = get_normal_vector(FΓ)

  d1(∇u) = 1/(ϵ + norm(∇u))
  _n(∇u) = ∇u/(ϵ + norm(∇u))
  β = velh*∇(φh)/(ϵ + norm ∘ ∇(φh))
  stiffness(t,u,v) = ∫((β ⋅ ∇(u)) * v)dΩ + ∫(c*h^2*jump(∇(u) ⋅ n)*jump(∇(v) ⋅ n))dFΓ
  mass(t, ∂ₜu, v) = ∫(∂ₜu * v)dΩ
  forcing(t,v) = ∫(0v)dΩ

  op = TransientLinearFEOperator((stiffness,mass),forcing,Ut_φ,V_φ,constant_forms=(true,true))
  uht = solve(ode_solver,op,0.0,Tf,φh)
  for (t,uh) in uht
    if t ≈ Tf
      copyto!(get_free_dof_values(φh),get_free_dof_values(uh))
    end
  end
end

function GridapTopOpt.reinit!(s::UnfittedFEEvolution,φ::AbstractArray,args...)
  _, V_φ = s.spaces
  reinit!(s,FEFunction(V_φ,φ),args...)
end

# Based on:
# @article{
#   Mallon_Thornton_Hill_Badia,
#   title={NEURAL LEVEL SET TOPOLOGY OPTIMIZATION USING UNFITTED FINITE ELEMENTS},
#   author={Mallon, Connor N and Thornton, Aaron W and Hill, Matthew R and Badia, Santiago},
#   language={en}
# }
function GridapTopOpt.reinit!(s::UnfittedFEEvolution,φh,args...)
  Ut_φ, V_φ = s.spaces
  params = s.params
  NT, h, c, dΩ = params.NT, params.h, params.c, params.dΩ
  reinit_nls = s.reinit_nls

  γd = 20
  cₐ = 0.5 # <- 3 in connor's paper
  ϵ = 1e-20

  # Tmp
  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)
  Γ = EmbeddedBoundary(cutgeo)
  dΓ = Measure(Γ,2*order)

  sgn(ϕ₀) = sign ∘ ϕ₀
  d1(∇u) = 1 / ( ϵ + norm(∇u) )
  W(u) =  sgn(u) * ∇(u) * (d1 ∘ (∇(u)))
  νₐ(w) = cₐ*h*(sqrt∘( w ⋅ w ))
  a_ν(w,u,v) = ∫((γd/h)*v*u)dΓ + ∫(νₐ(W(w))*∇(u)⋅∇(v) + v*W(w)⋅∇(u))dΩ
  b_ν(w,v) = ∫( sgn(w)*v )dΩ
  res(u,v)    = a_ν(u,u,v)  - b_ν(u,v)
  jac(u,du,v)  = a_ν(u,du,v)

  op = FEOperator(res,jac,V_φ,V_φ)
  Gridap.solve!(φh,reinit_nls,op)
end

function GridapTopOpt.get_dof_Δ(s::UnfittedFEEvolution)
  s.params.h
end