function diffusion_isolated_volumes_mask(mesh_size,cutgeo,tags,in_or_out::Int,dΩ,dΓg,n_Γg;
  h_ψ::Number = 0.01,
  ψ_inf::Number = 1.0,
  α_GP::Number = 0.05,
  k::Number = 1.0,
  kw::Number = 1000.0,
  kt::Number = 0.99,
  reffe = ReferenceFE(lagrangian,Float64,1))

  Ξ = TestFESpace(Triangulation(cutgeo,in_or_out),reffe,dirichlet_tags=tags)
  Ψ = TrialFESpace(Ξ)

  # Weak form
  J(c) = k*∇(c)
  r_Ω(ψ,ξ) = ∫(∇(ξ)⋅J(ψ) - ξ*h_ψ*ψ)dΩ
  γ_GP(h) = α_GP*h
  r_GP(ψ,ξ,h::Number) = ∫(mean(γ_GP(h))*jump(∇(ξ)⋅n_Γg)*jump(J(ψ)⋅n_Γg))dΓg
  r_GP(ψ,ξ,h::CellField) = ∫(mean(γ_GP ∘ h)*jump(∇(ξ)⋅n_Γg)*jump(J(ψ)⋅n_Γg))dΓg

  A(ψ,ξ) = r_Ω(ψ,ξ) + r_GP(ψ,ξ,mesh_size)
  B(ξ) = ∫(-ξ*h_ψ*ψ_inf)dΩ

  op = AffineFEOperator(A,B,Ξ,Ψ)
  ψh = solve(op)

  ψbar(ψ) = 1/2 + 1/2*tanh(kw*(ψ-kt*ψ_inf))

  return ψbar ∘ ψh
end

function diffusion_isolated_volumes_mask(h,cutgeo,in_or_out,tags;order=1,kwargs...)
  in_or_out_phys = (GridapEmbedded.Interfaces.CutInOrOut(in_or_out),in_or_out)
  Ω = Triangulation(cutgeo,in_or_out_phys)
  dΩ = Measure(Ω,2order)
  Γg = GhostSkeleton(cutgeo)
  n_Γg = get_normal_vector(Γg)
  dΓg = Measure(Γg,2order)

  diffusion_isolated_volumes_mask(h,cutgeo,tags,in_or_out,dΩ,dΓg,n_Γg;kwargs...)
end