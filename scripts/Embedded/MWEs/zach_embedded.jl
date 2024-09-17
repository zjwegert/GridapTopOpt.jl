using GridapTopOpt

using Gridap

using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData
import Gridap.Geometry: get_node_coordinates, collect1d

include("../differentiable_trians.jl")

order = 1
n = 101
N = 16
a = 1
begin
_model = CartesianDiscreteModel((0,1,0,1),(n,n))
model = simplexify(_model)
Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)

reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe_scalar)

# φh = interpolate(x->max(abs(x[1]-0.5),abs(x[2]-0.5))-0.25,V_φ) # Square
# φh = interpolate(x->((x[1]-0.5)^N+(x[2]-0.5)^N)^(1/N)-0.25,V_φ) # Curved corner example
# φh = interpolate(x->abs(x[1]-0.5)+abs(x[2]-0.5)-0.25-0/n/10,V_φ) # Diamond
# φh = interpolate(x->(1+0.25)abs(x[1]-0.5)+0abs(x[2]-0.5)-0.25,V_φ) # Straight lines with scaling
# φh = interpolate(x->abs(x[1]-0.5)+0abs(x[2]-0.5)-0.25/(1+0.25),V_φ) # Straight lines without scaling
# φh = interpolate(x->tan(-pi/3)*(x[1]-0.5)+(x[2]-0.5),V_φ) # Angled interface
# φh = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-0.25,V_φ) # Circle
φh = interpolate(x->a*(cos(2π*x[1])*cos(2π*x[2])-0.11),V_φ) # "Regular" LSF
x_φ = get_free_dof_values(φh)

if any(isapprox(0.0;atol=10^-10),x_φ)
  println("Issue with lvl set alignment")
  idx = findall(isapprox(0.0;atol=10^-10),x_φ)
  x_φ[idx] .+= 10eps()
end
any(x -> x < 0,x_φ)
any(x -> x > 0,x_φ)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

Ωin = Triangulation(cutgeo,PHYSICAL_IN).a
Ωout = Triangulation(cutgeo,PHYSICAL_OUT).a

diff_Ωin = DifferentiableTriangulation(Ωin)
diff_Ωout = DifferentiableTriangulation(Ωout)

fh = interpolate(x->x[1]+x[2],V_φ)

dΩin = Measure(diff_Ωin,2*order)
j_in(φ) = ∫(fh)dΩin
dj_in = gradient(j_in,φh)
dj_vec_in = assemble_vector(dj_in,V_φ)
norm(dj_vec_in)

dΩout = Measure(diff_Ωout,2*order)
j_out(φ) = ∫(fh)dΩout
dj_out = gradient(j_out,φh)
dj_vec_out = -assemble_vector(dj_out,V_φ)
norm(dj_vec_out)

Γ = EmbeddedBoundary(cutgeo)
dΓ = Measure(Γ,2*order)
dj_expected(q) = ∫(-(fh)*q/(norm ∘ (∇(φh))))dΓ
dj_exp_vec = assemble_vector(dj_expected,V_φ)
norm(dj_exp_vec)

@show dj_vec_in ≈ dj_exp_vec
@show dj_vec_out ≈ dj_exp_vec
bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo)

writevtk(
  Ω,"results/test",
  cellfields=["φh"=>φh,"∇φh"=>∇(φh),"dj_in"=>FEFunction(V_φ,dj_vec_in),"dj_expected"=>FEFunction(V_φ,dj_exp_vec),"dj_out"=>FEFunction(V_φ,dj_vec_out)],
  celldata=["inoutcut"=>bgcell_to_inoutcut]
)

writevtk(
  Triangulation(cutgeo,PHYSICAL_IN),"results/trian_in"
)
writevtk(
  Triangulation(cutgeo,PHYSICAL_OUT),"results/trian_out"
)
end

order = 1
n = 101
N = 16
a = 1
begin
model = CartesianDiscreteModel((0,1,0,1,0,1),(n,n,n))
model = simplexify(model)
Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)

reffe_scalar = ReferenceFE(lagrangian,Float64,order)
V_φ = TestFESpace(model,reffe_scalar)

# φh = interpolate(x->max(abs(x[1]-0.5),abs(x[2]-0.5),abs(x[3]-0.5))-0.25,V_φ) # Square prism
# φh = interpolate(x->((x[1]-0.5)^N+(x[2]-0.5)^N+(x[3]-0.5)^N)^(1/N)-0.25,V_φ) # Curved corner example
# φh = interpolate(x->abs(x[1]-0.5)+abs(x[2]-0.5)+abs(x[3]-0.5)-0.25-0/n/10,V_φ) # Diamond prism
# φh = interpolate(x->(1+0.25)abs(x[1]-0.5)+0abs(x[2]-0.5)+0abs(x[3]-0.5)-0.25,V_φ) # Straight lines with scaling
# φh = interpolate(x->abs(x[1]-0.5)+0abs(x[2]-0.5)+0abs(x[3]-0.5)-0.25/(1+0.25),V_φ) # Straight lines without scaling
# φh = interpolate(x->tan(-pi/3)*(x[1]-0.5)+(x[2]-0.5),V_φ) # Angled interface
# φh = interpolate(x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-0.25,V_φ) # Sphere
φh = interpolate(x->a*(cos(2π*x[1])*cos(2π*x[2])*cos(2π*x[3])-0.11),V_φ) # "Regular" LSF
# φh = interpolate(x->0.3(cos(2π*(x[1]-0.5))*cos(2π*(x[2]-0.5))*cos(2π*(x[3]-0.5)))+0.2(cos(2π*(x[1]-0.5))+cos(2π*(x[2]-0.5))+cos(2π*(x[3]-0.5)))+0.1(cos(2π*2*(x[1]-0.5))*cos(2π*2*(x[2]-0.5))*cos(2π*2*(x[3]-0.5)))+0.1(cos(2π*2*(x[1]-0.5))+cos(2π*2*(x[2]-0.5))+cos(2π*2*(x[3]-0.5)))+0.05(cos(2π*3*(x[1]-0.5))+cos(2π*3*(x[2]-0.5))+cos(2π*3*(x[3]-0.5)))+0.1(cos(2π*(x[1]-0.5))*cos(2π*(x[2]-0.5))+cos(2π*(x[2]-0.5))*cos(2π*(x[3]-0.5))+cos(2π*(x[3]-0.5))*cos(2π*(x[1]-0.5))),V_φ)
x_φ = get_free_dof_values(φh)

if any(isapprox(0.0;atol=10^-10),x_φ)
  println("Issue with lvl set alignment")
  idx = findall(isapprox(0.0;atol=10^-10),x_φ)
  x_φ[idx] .+= 10eps()
end
any(x -> x < 0,x_φ)
any(x -> x > 0,x_φ)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

Ωin = Triangulation(cutgeo,PHYSICAL_IN).a
Ωout = Triangulation(cutgeo,PHYSICAL_OUT).a

diff_Ωin = DifferentiableTriangulation(Ωin)
diff_Ωout = DifferentiableTriangulation(Ωout)

fh = interpolate(x->x[1]+x[2]+x[3],V_φ)

dΩin = Measure(diff_Ωin,2*order)
j_in(φ) = ∫(fh)dΩin
dj_in = gradient(j_in,φh)
dj_vec_in = assemble_vector(dj_in,V_φ)
norm(dj_vec_in)

dΩout = Measure(diff_Ωout,2*order)
j_out(φ) = ∫(fh)dΩout
dj_out = gradient(j_out,φh)
dj_vec_out = -assemble_vector(dj_out,V_φ)
norm(dj_vec_out)

Γ = EmbeddedBoundary(cutgeo)
dΓ = Measure(Γ,2*order)
dj_expected(q) = ∫(-fh*q/(norm ∘ (∇(φh))))dΓ
dj_exp_vec = assemble_vector(dj_expected,V_φ)
norm(dj_exp_vec)

@show dj_vec_in ≈ dj_exp_vec
@show dj_vec_out ≈ dj_exp_vec
bgcell_to_inoutcut = compute_bgcell_to_inoutcut(model,geo)

writevtk(
  Ω,"results/test3d",
  cellfields=["φh"=>φh,"∇φh"=>∇(φh),"dj_in"=>FEFunction(V_φ,dj_vec_in),"dj_expected"=>FEFunction(V_φ,dj_exp_vec),"dj_out"=>FEFunction(V_φ,dj_vec_out)],
  celldata=["inoutcut"=>bgcell_to_inoutcut]
)

writevtk(
  Triangulation(cutgeo,PHYSICAL_IN),"results/trian_in3d"
)
writevtk(
  Triangulation(cutgeo,PHYSICAL_OUT),"results/trian_out3d"
)
end

## Surface functionals
# We should have something like the following for J(φ)=∫_∂Ω(φ) f ds:
#   dJ(φ)(w) = lim_{t->0} (J(φₜ)-J(φ))/t = -∫_∂Ω (n⋅∇f)w/|∂ₙφ| dS - ∑(...)
# The first term is easy, the second term not so much...

# Need [[fm]]=f₁m₁ + f₂m₂, where mₖ is the co-normal for τ = 0 with
#                   mₖ = tₖˢ×n_{∂Ω(φ(0))∩S}
# where tₖˢ is the tangent vector along the edge ∂Ω(φ(0))∩S and fₖ is
# the limit of f on S defined by fₖ(x)=lim_{ϵ->0⁺} f(x-ϵmₖ) for x ∈ ∂Ω∩S.