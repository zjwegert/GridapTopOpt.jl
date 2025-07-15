"""
    struct HeatReinitialiser <: Reinitialiser end

A heat method for reinitialisation of the level-set function as an approximate
signed distance function. This is based on Feng and Crane (2024) [doi: 10.1145/3658220].

This works using in three steps:
- Step 1: Solve (I - tΔ)X = {Γ_n on Γ, 0 otherwise} for small t on M
- Step 2: Get Y = X/|X| (L2 projection to ensure we have an FEFunction)
- Step 3: Solve ΔΦ = ∇ ⋅ Y on M with n ⋅ ∇Φ = ∂M_n ⋅ Y on ∂M.
The resulting Φ is an approximation of the signed distance function. In the
above M is our background domain and Γ is the interface defined by the zero
isosurface of the level-set function.

Note, in Step 3 we add a penalty term to ensure that the resulting
isosurface Φ=0 lies close to φ=0. This results in a multifield tangent
space for yφ_to_sdf.

The implementation is in terms of `AffineFEStateMap`s to enable AD through the
maps using Zygote. The differentiable map `φ_to_sdf` can be obtained using the
function `get_lsf_to_sdf_map`.

# Parameters
- `φ_to_x`: AffineFEStateMap mapping from φ to the vector field x (Step 1)
- `x_to_y`: AffineFEStateMap mapping from the vector field x to the vector field y (Step 2)
- `yφ_to_sdf`: AffineFEStateMap mapping from the vector field y and φ to the sdf (Step 3)
- `Ωs`: An EmbeddedCollection for holding information regarding the geometry.
"""
struct HeatReinitialiser <: Reinitialiser
  φ_to_x   :: AffineFEStateMap
  x_to_y   :: AffineFEStateMap
  yφ_to_sdf :: AffineFEStateMap
  Ωs       :: EmbeddedCollection

  @doc """
      HeatReinitialiser(V_φ,model;
        t  = minimum(get_element_diameters(model))^2,
        γd = 10.0,
        boundary_tags = "boundary",
        V_xy = build_V_xy(model,V_φ),
        M_xyφ = MultiFieldFESpace([Vxy,V_φ]),
        V_sdf = build_V_sdf(model,V_φ),
        Ωs = build_Ωs(model,boundary_tags,V_φ)
      )

  Create an instance of `HeatReinitialiser`.

  Optional parameters:
  - `t`: Time step for the heat equation, default is the square of the minimum element diameter.
  - `γd`: Penalty parameter to preserve level sets, default is 10.
  - `boundary_tags`: Tag/s for the boundary, default is "boundary" (e.g., for CartesianDiscreteModel).
    For your own mesh (e.g., unstructured) covering M, you should provide the tags that consistute ∂M.

  Advanced optional parameters:
  - `V_xy`: TestFESpace for the vector field x, default is built using `build_V_xy`.
  - `M_xyφ`: MultiFieldFESpace for the vector field x and φ, default is built using `MultiFieldFESpace([V_xy,V_φ])`.
    This is used in the third step to solve for the signed distance function.
  - `V_sdf`: TestFESpace for the signed distance function, default is built using `build_V_sdf`.
  - `Ωs`: An EmbeddedCollection for holding information regarding the geometry, default is built using `build_Ωs`
  - `φ_to_x_ls`: Solver for the φ_to_x map, default is `LUSolver()`.
  - `x_to_y_ls`: Solver for the x_to_y map, default is `LUSolver()`.
  - `yφ_to_sdf_ls`: Solver for the yφ_to_sdf map, default is `LUSolver()`.
  - `φ_to_x_adjoint_ls`: Adjoint solver for the φ_to_x map, default is `φ_to_x_ls`.
  - `x_to_y_adjoint_ls`: Adjoint solver for the x_to_y map, default is `x_to_y_ls`.
  - `yφ_to_sdf_adjoint_ls`: Adjoint solver for the yφ_to_sdf map, default is `yφ_to_sdf_ls`.

  !!! note
      When using an unstructured mesh with significant changes in mesh sizes, you should consider
      testing different values of `t`, e.g., minimum, maximum, or mean element diameter. It is
      also possible to set `t` to be the square of the element size field from `get_element_diameter_field`:
      E.g.,
      ```
      hsq(h) = h^2
      t = hsq ∘ get_element_diameter_field(model)
      ```
      We have found that the latter yields better results for some problems.
  """
  function HeatReinitialiser(V_φ,model;
    t  = minimum(get_element_diameters(model))^2,
    γd = 10.0,
    boundary_tags = "boundary",
    V_xy = build_V_xy(model,V_φ),
    M_xyφ = MultiFieldFESpace([V_xy,V_φ]),
    V_sdf = build_V_sdf(model,V_φ),
    Ωs = build_Ωs(model,boundary_tags,V_φ),
    φ_to_x_ls = LUSolver(),
    x_to_y_ls = LUSolver(),
    yφ_to_sdf_ls = LUSolver(),
    φ_to_x_adjoint_ls = φ_to_x_ls,
    x_to_y_adjoint_ls = x_to_y_ls,
    yφ_to_sdf_adjoint_ls = yφ_to_sdf_ls
  )
    φ_to_x_solvers = (;ls=φ_to_x_ls,adjoint_ls=φ_to_x_adjoint_ls)
    x_to_y_solvers = (;ls=x_to_y_ls,adjoint_ls=x_to_y_adjoint_ls)
    yφ_to_sdf_solvers = (;ls=yφ_to_sdf_ls,adjoint_ls=yφ_to_sdf_adjoint_ls)


    φ_to_x, x_to_y, yφ_to_sdf = build_state_maps(Ωs,V_φ,V_xy,M_xyφ,V_sdf,t,γd,
      φ_to_x_solvers, x_to_y_solvers, yφ_to_sdf_solvers)

    new(φ_to_x,x_to_y,yφ_to_sdf,Ωs)
  end
end

function reinit!(m::HeatReinitialiser,φh::FEFunction)
  update_collection!(m.Ωs,φh)
  φ   = get_free_dof_values(φh)
  x   = m.φ_to_x(φ)
  y   = m.x_to_y(x)
  yφ  = combine_fields(get_deriv_space(m.yφ_to_sdf),y,φ)
  sdf = m.yφ_to_sdf(yφ)
  copyto!(φ,sdf)
  φ,nothing
end

function reinit!(m::HeatReinitialiser,φh::DistributedCellField)
  update_collection!(m.Ωs,φh)
  φ   = get_free_dof_values(φh)
  x   = m.φ_to_x(φ)
  y   = m.x_to_y(x)
  yφ  = combine_fields(get_deriv_space(m.yφ_to_sdf),y,φ)
  sdf = m.yφ_to_sdf(yφ)
  copyto!(φ,sdf)
  consistent!(φ) |> fetch
  φ,nothing
end

function reinit!(m::HeatReinitialiser,φ::AbstractVector)
  V_φ = get_deriv_space(m.φ_to_x)
  reinit!(m,FEFunction(V_φ,φ))
end
function reinit!(m::HeatReinitialiser,φ::AbstractVector,cache)
  V_φ = get_deriv_space(m.φ_to_x)
  reinit!(m,FEFunction(V_φ,φ))
end
reinit!(m::HeatReinitialiser,φh,cache) = reinit!(m,φh)

function get_lsf_to_sdf_map(m::HeatReinitialiser)
  V_φ = get_deriv_space(m.φ_to_x)
  function φ_to_sdf(φ)
    Zygote.ignore_derivatives() do
      update_collection!(m.Ωs,FEFunction(V_φ,φ))
    end
    x   = m.φ_to_x(φ)
    y   = m.x_to_y(x)
    yφ  = combine_fields(get_deriv_space(m.yφ_to_sdf),y,φ)
    sdf = m.yφ_to_sdf(yφ)
  end
end

# Utils
function build_V_xy(model,V_φ)
  D = _num_dims(V_φ)
  order = get_order(V_φ)
  T = eltype(get_vector_type(V_φ))
  TestFESpace(model,ReferenceFE(lagrangian,VectorValue{D,T},order))
end

function build_V_sdf(model,V_φ)
  order = get_order(V_φ)
  T = eltype(get_vector_type(V_φ))
  TestFESpace(model,ReferenceFE(lagrangian,T,order))
end

function build_Ωs(model,boundary_tags,V_φ)
  order = get_order(V_φ)
  Ω = Triangulation(model)
  dM = Measure(Ω,2order)
  ∂M = BoundaryTriangulation(model,tags=boundary_tags)
  d∂M = Measure(∂M,2order)
  n = get_normal_vector(∂M)
  hₕ = get_element_diameter_field(model)

  Ωs = EmbeddedCollection(model) do cutgeo,_,_
    Γ = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
    dΓ = Measure(Γ,2order)
    (;
      :dM   => dM,
      :d∂M  => d∂M,
      :∂M_n => n,
      :Γ    => Γ,
      :dΓ   => dΓ,
      :hₕ    => hₕ
    )
  end
end

function build_state_maps(Ωs,V_φ,Vxy,M_xyφ,V_sdf,t,γd,
    φ_to_x_solvers,x_to_y_solvers,yφ_to_sdf_solvers)
  # Step 1 -- Solve (I - tΔ)X = {Γ_n on Γ, 0 otherwise} for small t on M
  A1(u,v,φ) = ∫(u ⋅ v + t*(∇(u) ⊙ ∇(v)))Ωs.dM
  function L1(v,φ)
    Γ_n = get_normal_vector(Ωs.Γ) # <- to enable AD
    return ∫(Γ_n ⋅ v)Ωs.dΓ
  end
  φ_to_x = AffineFEStateMap(A1,L1,Vxy,Vxy,V_φ;φ_to_x_solvers...)

  # Step 2 -- Get Y = X/|X| (we use an L2 projection here)
  _y(x) = -x/norm(x)
  A2(u,v,x) = ∫(u⋅v)Ωs.dM
  L2(v,x) = ∫(v⋅(_y∘x))Ωs.dM
  x_to_y = AffineFEStateMap(A2,L2,Vxy,Vxy,Vxy;x_to_y_solvers...)

  # Step 3 -- Solve ΔΦ = ∇ ⋅ Y on M with n ⋅ ∇Φ = ∂M_n ⋅ Y on ∂M
  # Note, we add the penalty term ∫((γd/h)*v*u)Ωs.dΓ to ensure the resulting sdf
  # isosurface Φ=0 lies close to φ=0.
  @check V_φ !== V_sdf "These must be different objects (programmatically) for the purpose of differentiation"
  A3(u,v,(y,φ)) = ∫(∇(u)⋅∇(v))Ωs.dM + ∫((γd/Ωs.hₕ)*v*u)Ωs.dΓ
  L3(v,(y,φ)) = ∫(v*(∇ ⋅ y))Ωs.dM + ∫(-(Ωs.∂M_n ⋅ y)*v)Ωs.d∂M
  # NOTE: The latter integral in A3 is on Γ(φ), so we have to treat this problem
  #       as multifield with dependence on phi
  yφ_to_sdf = AffineFEStateMap(A3,L3,V_sdf,V_sdf,M_xyφ;yφ_to_sdf_solvers...)

  return φ_to_x,x_to_y,yφ_to_sdf
end