using Gridap,GridapTopOpt, GridapSolvers
using Gridap.Adaptivity, Gridap.Geometry
using GridapEmbedded, GridapEmbedded.LevelSetCutters

using GridapTopOpt: StateParamIntegrandWithMeasure

using DataStructures

const CUT = 0

# TODO: Can be optimized CartesianModels
function generate_neighbor_graph(model::DiscreteModel{Dc}) where Dc
  topo = get_grid_topology(model)
  cell_to_node = Geometry.get_faces(topo, Dc, 0)
  node_to_cell = Geometry.get_faces(topo, 0, Dc)
  cell_to_nbors = map(1:num_cells(model)) do cell
    unique(sort(vcat(map(n -> view(node_to_cell,n), view(cell_to_node,cell))...)))
  end
  return cell_to_nbors
end

"""
  Given an initial interface cell, enqueue all the CUT cells in the same interface
  inside the queue `q_cut` and mark them as touched in the `touched` array.
"""
function enqueue_interface!(q_cut,cell_to_nbors,cell_to_inoutcut,touched,cell)
  q = Queue{Int}(); enqueue!(q,cell)
  enqueue!(q_cut,cell)
  touched[cell] = true
  while !isempty(q)
    cell = dequeue!(q)
    nbors = cell_to_nbors[cell]
    for nbor in nbors
      if !touched[nbor] && (cell_to_inoutcut[nbor] == CUT)
        touched[nbor] = true
        enqueue!(q_cut,nbor)
        enqueue!(q,nbor)
      end
    end
  end
end

function tag_isolated_volumes(
  model::DiscreteModel{Dc}, cell_to_inoutcut::Vector{<:Integer}
) where Dc

  n_cells = num_cells(model)
  cell_to_nbors = generate_neighbor_graph(model)

  n_color = 0
  cell_color = zeros(Int16, n_cells)
  color_to_inout = Int8[]
  touched  = falses(n_cells)
  q, q_cut = Queue{Int}(), Queue{Int}()

  # First pass: Color IN/OUT cells
  #   - We assume that every IN/OUT transition can be bridged by a CUT cell
  first_cell = findfirst(state -> state != CUT, cell_to_inoutcut)
  enqueue!(q,first_cell); touched[first_cell] = true; # Queue first cell
  while !isempty(q)
    cell  = dequeue!(q)
    nbors = cell_to_nbors[cell]
    state = cell_to_inoutcut[cell]

    # Mark with color
    if state != CUT
      i = findfirst(!iszero,view(cell_color,nbors))
      if isnothing(i) # New patch
        n_color += 1
        cell_color[cell] = n_color
        push!(color_to_inout, state)
      else # Existing patch
        color = cell_color[nbors[i]]
        cell_color[cell] = color
      end
    end

    # Queue and touch unseen neighbors
    # We touch neighbors here to avoid enqueuing the same cell multiple times
    for nbor in nbors
      if !touched[nbor]
        touched[nbor] = true
        enqueue!(q,nbor)
        if cell_to_inoutcut[nbor] == CUT
          enqueue_interface!(q_cut,cell_to_nbors,cell_to_inoutcut,touched,nbor)
        end
      end
    end
  end

  # Second pass: Color CUT cells
  #   - We assume that every CUT cell has an IN neighbor
  #   - We assume all IN neighbors have the same color
  # Then we assign the same color to the CUT cell
  while !isempty(q_cut)
    cell  = dequeue!(q_cut)
    nbors = cell_to_nbors[cell]
    @assert cell_to_inoutcut[cell] == CUT

    i = findfirst(n -> cell_to_inoutcut[n] == IN, nbors)
    @assert !isnothing(i)
    cell_color[cell] = cell_color[nbors[i]]
  end

  return cell_color, color_to_inout
end

path="./results/FCM_thermal_compliance_ALM_with_islands/"
rm(path,force=true,recursive=true)
mkpath(path)
n = 50
order = 1
γ = 0.1
max_steps = floor(Int,order*n/5)
vf = 0.4
α_coeff = 4max_steps*γ
iter_mod = 1

_model = CartesianDiscreteModel((0,1,0,1),(n,n))
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = ref_model.model
el_Δ = get_el_Δ(_model)
h = maximum(el_Δ)
h_refine = maximum(el_Δ)/2
f_Γ_D(x) = (x[1] ≈ 0.0 && (x[2] <= 0.2 + eps() || x[2] >= 0.8 - eps()))
f_Γ_N(x) = (x[1] ≈ 1 && 0.4 - eps() <= x[2] <= 0.6 + eps())
update_labels!(1,model,f_Γ_D,"Gamma_D")
update_labels!(2,model,f_Γ_N,"Gamma_N")

## Triangulations and measures
Ω = Triangulation(model)
Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
dΩ = Measure(Ω,2*order)
dΓ_N = Measure(Γ_N,2*order)
vol_D = sum(∫(1)dΩ)

## Levet-set function space and derivative regularisation space
reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_reg = TestFESpace(model,reffe_scalar;dirichlet_tags=["Gamma_N"])
U_reg = TrialFESpace(V_reg,0)
V_φ = TestFESpace(model,reffe_scalar)

## Levet-set function
φh = interpolate(x->-cos(4π*x[1])*cos(4π*x[2])-0.4,V_φ)
Ωs = EmbeddedCollection(model,φh) do cutgeo,_
  Ωin = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_IN),V_φ)
  Ωout = DifferentiableTriangulation(Triangulation(cutgeo,PHYSICAL_OUT),V_φ)
  Γ = DifferentiableTriangulation(EmbeddedBoundary(cutgeo),V_φ)
  Γg = GhostSkeleton(cutgeo)
  Ωact = Triangulation(cutgeo,ACTIVE)
  (;
    :Ωin  => Ωin,
    :dΩin => Measure(Ωin,2*order),
    :Ωout  => Ωout,
    :dΩout => Measure(Ωout,2*order),
    :Γg   => Γg,
    :dΓg  => Measure(Γg,2*order),
    :n_Γg => get_normal_vector(Γg),
    :Γ    => Γ,
    :dΓ   => Measure(Γ,2*order),
    :Ωact => Ωact
  )
end

## Weak form
const ϵ = 1e-3
a(u,v,φ) = ∫(∇(v)⋅∇(u))Ωs.dΩin + ∫(ϵ*∇(v)⋅∇(u))Ωs.dΩout
l(v,φ) = ∫(v)dΓ_N

## Optimisation functionals
J(u,φ) = a(u,u,φ)
Vol(u,φ) = ∫(1/vol_D)Ωs.dΩin - ∫(vf/vol_D)dΩ
dVol(q,u,φ) = ∫(-1/vol_D*q/(norm ∘ (∇(φ))))Ωs.dΓ

## Setup solver and FE operators
V = TestFESpace(Ω,reffe_scalar;dirichlet_tags=["Gamma_D"])
U = TrialFESpace(V,0.0)
state_map = AffineFEStateMap(a,l,U,V,V_φ,U_reg,φh)
pcfs = PDEConstrainedFunctionals(J,[Vol],state_map;analytic_dC=(dVol,))

## Evolution Method
evo = CutFEMEvolve(V_φ,Ωs,dΩ,h;max_steps)
reinit = StabilisedReinit(V_φ,Ωs,dΩ,h;stabilisation_method=ArtificialViscosity(3.0))
ls_evo = UnfittedFEEvolution(evo,reinit)
reinit!(ls_evo,φh)

## Hilbertian extension-regularisation problems
α = α_coeff*(h_refine/order)^2
a_hilb(p,q) =∫(α*∇(p)⋅∇(q) + p*q)dΩ;
vel_ext = VelocityExtension(a_hilb,U_reg,V_reg)

## Optimiser
converged(m) = GridapTopOpt.default_al_converged(
  m;
  L_tol = 0.01*h_refine,
  C_tol = 0.01
)
optimiser = AugmentedLagrangian(pcfs,ls_evo,vel_ext,φh;debug=true,
  γ,verbose=true,constraint_names=[:Vol],converged)
for (it,uh,φh,state) in optimiser
  x_φ = get_free_dof_values(φh)
  idx = findall(isapprox(0.0;atol=10^-10),x_φ)
  !isempty(idx) && @warn "Boundary intersects nodes!"
  if iszero(it % iter_mod)
    geo = DiscreteGeometry(φh,model)
    bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo)
    colors, color_to_inout = tag_isolated_volumes(model,bgcell_to_inoutcut)

    writevtk(Ω,path*"Omega$it",
      cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh,"velh"=>FEFunction(V_φ,state.vel)],
      celldata=["inoutcut"=>bgcell_to_inoutcut,"volumes"=>colors];
      append=false)
    writevtk(Ωs.Ωin,path*"Omega_in$it",cellfields=["uh"=>uh])
  end
  write_history(path*"/history.txt",optimiser.history)
end
it = get_history(optimiser).niter; uh = get_state(pcfs)
geo = DiscreteGeometry(φh,model)
bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo)
colors, color_to_inout = tag_isolated_volumes(model,bgcell_to_inoutcut)
writevtk(Ω,path*"Omega$it",
  cellfields=["φ"=>φh,"|∇(φ)|"=>(norm ∘ ∇(φh)),"uh"=>uh],
  celldata=["inoutcut"=>bgcell_to_inoutcut,"volumes"=>colors];
      append=false)
writevtk(Ωs.Ωin,path*"Omega_in$it",cellfields=["uh"=>uh])