module EmbeddedDifferentiationTests
using Test

using GridapTopOpt
using Gridap, Gridap.Geometry, Gridap.Adaptivity
using GridapEmbedded, GridapEmbedded.LevelSetCutters

using Gridap.Arrays: Operation
using GridapTopOpt: get_conormal_vector,get_subfacet_normal_vector,get_ghost_normal_vector

function generate_model(D,n)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  base_model = UnstructuredDiscreteModel((CartesianDiscreteModel(domain,cell_partition)))
  ref_model = refine(base_model, refinement_method = "barycentric")
  model = ref_model.model
  return model
end

function level_set(shape::Symbol;N=4)
  if shape == :square
    x -> max(abs(x[1]-0.5),abs(x[2]-0.5))-0.25 # Square
  elseif shape == :corner_2d
    x -> ((x[1]-0.5)^N+(x[2]-0.5)^N)^(1/N)-0.25 # Curved corner
  elseif shape == :diamond
    x -> abs(x[1]-0.5)+abs(x[2]-0.5)-0.25-0/n/10 # Diamond
  elseif shape == :circle
    x -> sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-0.5223 # Circle
  elseif shape == :square_prism
    x -> max(abs(x[1]-0.5),abs(x[2]-0.5),abs(x[3]-0.5))-0.25 # Square prism
  elseif shape == :corner_3d
    x -> ((x[1]-0.5)^N+(x[2]-0.5)^N+(x[3]-0.5)^N)^(1/N)-0.25 # Curved corner
  elseif shape == :diamond_prism
    x -> abs(x[1]-0.5)+abs(x[2]-0.5)+abs(x[3]-0.5)-0.25-0/n/10 # Diamond prism
  elseif shape == :sphere
    x -> sqrt((x[1]-0.5)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-0.53 # Sphere
  elseif shape == :regular_2d
    x -> cos(2π*x[1])*cos(2π*x[2])-0.11 # "Regular" LSF
  elseif shape == :regular_3d
    x -> cos(2π*x[1])*cos(2π*x[2])*cos(2π*x[3])-0.11 # "Regular" LSF
  end
end

function main(
  model,φ::Function,f::Function;
  vtk=false,
  name="embedded"
)
  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  V_φ = TestFESpace(model,reffe)

  φh = interpolate(φ,V_φ)
  fh = interpolate(f,V_φ)

  # Correction if level set is on top of a node
  x_φ = get_free_dof_values(φh)
  idx = findall(isapprox(0.0;atol=10^-10),x_φ)
  !isempty(idx) && @info "Correcting level values!"
  x_φ[idx] .+= 10*eps(eltype(x_φ))

  geo = DiscreteGeometry(φh,model)
  cutgeo = cut(model,geo)

  # A) Volume integral

  Ω = Triangulation(cutgeo,PHYSICAL_IN)
  Ω_AD = DifferentiableTriangulation(Ω)
  dΩ = Measure(Ω_AD,2*order)

  Γ = EmbeddedBoundary(cutgeo)
  dΓ = Measure(Γ,2*order)

  J_bulk(φ) = ∫(fh)dΩ
  dJ_bulk_AD = gradient(J_bulk,φh)
  dJ_bulk_AD_vec = assemble_vector(dJ_bulk_AD,V_φ)

  dJ_bulk_exact(q) = ∫(-fh*q/(norm ∘ (∇(φh))))dΓ
  dJ_bulk_exact_vec = assemble_vector(dJ_bulk_exact,V_φ)

  @test norm(dJ_bulk_AD_vec - dJ_bulk_exact_vec) < 1e-10

  # B) Facet integral

  Γ = EmbeddedBoundary(cutgeo)
  Γ_AD = DifferentiableTriangulation(Γ)
  Λ = Skeleton(Γ)
  Σ = Boundary(Γ)

  dΓ = Measure(Γ,2*order)
  dΛ = Measure(Λ,2*order)
  dΣ = Measure(Σ,2*order)

  n_Γ = get_normal_vector(Γ)

  n_S_Λ = get_normal_vector(Λ)
  m_k_Λ = get_conormal_vector(Λ)
  ∇ˢφ_Λ = Operation(abs)(n_S_Λ ⋅ ∇(φh).plus)

  n_S_Σ = get_normal_vector(Σ)
  m_k_Σ = get_conormal_vector(Σ)
  ∇ˢφ_Σ = Operation(abs)(n_S_Σ ⋅ ∇(φh))

  dΓ_AD = Measure(Γ_AD,2*order)
  J_int(φ) = ∫(fh)dΓ_AD
  dJ_int_AD = gradient(J_int,φh)
  dJ_int_AD_vec = assemble_vector(dJ_int_AD,V_φ)

  dJ_int_exact(w) = ∫((-n_Γ⋅∇(fh))*w/(norm ∘ (∇(φh))))dΓ + 
                    ∫(-n_S_Λ ⋅ (jump(fh*m_k_Λ) * mean(w) / ∇ˢφ_Λ))dΛ +
                    ∫(-n_S_Σ ⋅ (fh*m_k_Σ * w / ∇ˢφ_Σ))dΣ
  dJ_int_exact_vec = assemble_vector(dJ_int_exact,V_φ)

  @test norm(dJ_int_AD_vec - dJ_int_exact_vec) < 1e-10

  if vtk
    path = "results/$(name)"
    Ω_bg = Triangulation(model)
    writevtk(
      Ω_bg,"$(path)_results",
      cellfields = [
        "φh" => φh,"∇φh" => ∇(φh),
        "dJ_bulk_AD" => FEFunction(V_φ,dJ_bulk_AD_vec),
        "dJ_bulk_exact" => FEFunction(V_φ,dJ_bulk_exact_vec),
        "dJ_int_AD" => FEFunction(V_φ,dJ_int_AD_vec),
        "dJ_int_exact" => FEFunction(V_φ,dJ_int_exact_vec)
      ],
      celldata = [
        "inoutcut" => GridapEmbedded.Interfaces.compute_bgcell_to_inoutcut(model,geo)
      ];
      append = false
    )
  
    writevtk(
      Ω, "$(path)_omega"; append = false
    )
    writevtk(
      Γ, "$(path)_gamma"; append = false
    )

    n_∂Ω_Λ = get_subfacet_normal_vector(Λ)
    n_k_Λ  = get_ghost_normal_vector(Λ)
    writevtk(
      Λ, "$(path)_lambda",
      cellfields = [
        "n_∂Ω.plus" => n_∂Ω_Λ.plus,"n_∂Ω.minus" => n_∂Ω_Λ.minus,
        "n_k.plus" => n_k_Λ.plus,"n_k.minus" => n_k_Λ.minus,
        "n_S" => n_S_Λ,
        "m_k.plus" => m_k_Λ.plus,"m_k.minus" => m_k_Λ.minus,
        "∇ˢφ" => ∇ˢφ_Λ,
        "∇φh_Γs_plus" => ∇(φh).plus,"∇φh_Γs_minus" => ∇(φh).minus,
        "jump(fh*m_k)" => jump(fh*m_k_Λ)
      ];
      append = false
    )

    if num_cells(Σ) > 0
      n_∂Ω_Σ = get_subfacet_normal_vector(Σ)
      n_k_Σ  = get_ghost_normal_vector(Σ)
      writevtk(
        Σ, "$(path)_sigma",
        cellfields = [
          "n_∂Ω" => n_∂Ω_Σ, "n_k" => n_k_Σ,
          "n_S" => n_S_Σ, "m_k" => m_k_Σ,
          "∇ˢφ" => ∇ˢφ_Σ, "∇φh_Γs" => ∇(φh),
        ];
        append = false
      )
    end
  end
end

#######################

D = 2
n = 10
model = generate_model(D,n)
φ = level_set(:circle)
f = x -> 1.0
main(model,φ,f;vtk=true)

end