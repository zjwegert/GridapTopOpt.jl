module HeatReintialiserDifferentiability

using Gridap, Gridap.Adaptivity, Gridap.Geometry, Gridap.Helpers
using GridapEmbedded, GridapEmbedded.LevelSetCutters
using GridapTopOpt, GridapSolvers
using FiniteDiff
using GridapDistributed
using Test

function main(;vtk=false)
  _model = CartesianDiscreteModel((0,1,0,1),(11,11))
  base_model = UnstructuredDiscreteModel(_model)
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = get_model(ref_model)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,2)

  order = 1
  reffe_scalar = ReferenceFE(lagrangian,Float64,order)
  V_φ = TestFESpace(model,reffe_scalar)

  ## Level-set function
  f1((x,y)) = (x-0.5)^2+(y-0.5)^2-0.35^2
  f2((x,y)) = (x-0.5)^2/1.5^2+(y-0.5)^2-0.2^2
  φh = interpolate(x->max(f1(x),-f2(x)),V_φ)
  φh_old = FEFunction(V_φ,copy(get_free_dof_values(φh)))

  reinit_method = HeatReinitialiser(V_φ,model)
  reinit!(reinit_method,φh);
  vtk && writevtk(Ω,"results/crane_sdf",cellfields=["φh_old"=>φh_old,"φh"=>φh,"|∇(φh)|"=>(norm ∘ ∇(φh))])

  # This is defined purely for E(d_Ω)
  _Ωs = EmbeddedCollection(model,φh) do cutgeo,_,_
    Ωin = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL),V_φ)
    (;
      :Ωin  => Ωin,
      :dΩin => Measure(Ωin,2),
    )
  end

  # E(d_Ω) Based on Section 9.2.2 of Allaire et al. (2016) [doi: 10.1007/s00158-016-1453-y]
  # I.e., we want to differentiate through E(d_Ω(φ),φ) = ∫( d_Ω^2*max(d_Ω+dmin/2,0)^2 )dΩin(φ)
  # where d_Ω is the signed distance function to the boundary of the domain defined by φ
  # which is also possibly a signed distance function.

  # Notes:
  # - We need to be careful as the sdf is defined on V_sdf, but φ is defined on V_φ. These
  #   are mathematically identical spaces, but programmatically they MUST be different for the
  #   purpose of AD. This because dΩin depends on φ, not d_Ω.
  # - Technically, we are using a signed distance function assuming that the boundary of
  #   the background domain is not part of the design domain Ω. This could be considered
  #   in future.
  V_sdf = GridapTopOpt.get_trial_space(reinit_method.yφ_to_sdf)
  dmin = 0.1
  _op(sdf) = -sdf^2*max(sdf+dmin/2,0)^2
  _E(sdf,φ) = ∫(_op ∘ sdf)_Ωs.dΩin
  E = GridapTopOpt.StateParamMap(_E,V_sdf,V_φ,SparseMatrixAssembler(V_φ,V_φ),SparseMatrixAssembler(V_sdf,V_sdf))

  φ_to_sdf = get_lsf_to_sdf_map(reinit_method)

  function φ_to_E(φ)
    sdf = φ_to_sdf(φ)
    E(sdf,φ)/1e-8
  end

  val, grad = GridapTopOpt.val_and_jacobian(φ_to_E,get_free_dof_values(φh))

  function φ_to_E_fd(φ)
    update_collection!(reinit_method.Ωs,FEFunction(V_φ,φ))
    update_collection!(_Ωs,FEFunction(V_φ,φ))
    sdf = φ_to_sdf(φ)
    E(sdf,φ)/1e-8
  end

  fdm_grad = FiniteDiff.finite_difference_gradient(φ_to_E_fd, get_free_dof_values(φh))
  rel_error = norm(grad[1][1] - fdm_grad, Inf)/norm(fdm_grad,Inf)
  @test rel_error < 1e-6
  vtk && writevtk(Ω,"results/crane_sdf",cellfields=["φh_old"=>φh_old,"φh"=>φh,"|∇(φh)|"=>(norm ∘ ∇(φh)),
    "grad"=>FEFunction(V_φ,grad[1][1]),"fdm_grad"=>FEFunction(V_φ,fdm_grad)])
end

main(;vtk=false)

end