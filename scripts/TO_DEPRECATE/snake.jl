using Gridap
using Gridap.ReferenceFEs, Gridap.Geometry
using Gridap.ReferenceFEs: get_graph, isactive
using Gridap.Adaptivity

using GridapEmbedded, GridapEmbedded.Interfaces, GridapEmbedded.LevelSetCutters

using GridapDistributed, PartitionedArrays

using GridapTopOpt

function snake(X)
  x, y = X[1], X[2]
  A = (0.1 < x < 0.61) && (0.61 < y < 0.89)
  B = (-0.1 < x < 0.61) && (0.11 < y < 0.39)
  C = (0.6 < x < 0.8) && (0.11 < y < 0.89)
  inside = A || B || C
  ifelse(inside,-1.0,1.0)
end

np = (2,2)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end

n = 10
_model = CartesianDiscreteModel(ranks,np,(0,1,0,1),(n,n))
base_model = UnstructuredDiscreteModel(_model)
ref_model = refine(base_model, refinement_method = "barycentric")
model = Adaptivity.get_model(ref_model)

reffe = ReferenceFE(lagrangian,Float64,1)
V = TestFESpace(model,reffe)
φh = interpolate(snake,V)

geo = DiscreteGeometry(φh,model)
cutgeo = cut(model,geo)

Ωbg = Triangulation(model) 
Ωp_in = Triangulation(cutgeo,PHYSICAL)
Ωp_out = Triangulation(cutgeo,PHYSICAL_OUT)
Ωa_in = Triangulation(cutgeo,ACTIVE)
Ωa_out = Triangulation(cutgeo,ACTIVE_OUT)
Ωc_in = Triangulation(cutgeo,CUT_IN)
Ωc_out = Triangulation(cutgeo,CUT_OUT)
Γ  = EmbeddedBoundary(cutgeo)

cell_values = map(get_cell_dof_values,local_views(φh));
pmodel, subcell_to_inout, subcell_to_cell = GridapTopOpt.cut_conforming(model,cell_values);

subcell_to_lcolor, lcolor_to_group, color_gids = GridapTopOpt.tag_disconnected_volumes(pmodel,subcell_to_inout;groups=(IN,OUT))
subcell_to_color = map((c2l,l2g) -> collect(l2g[c2l]),subcell_to_lcolor,local_to_global(color_gids))

cf_IN, cf_OUT = GridapTopOpt.get_isolated_volumes_mask_polytopal(
  model,cell_values,["boundary"]
)

path = "results/"
writevtk(Ωp_in,path*"snake_Ωp_in";append=false)
writevtk(Ωp_out,path*"snake_Ωp_out";append=false)
writevtk(Ωa_in,path*"snake_Ωa_in";append=false)
writevtk(Ωa_out,path*"snake_Ωa_out";append=false)
writevtk(Ωc_in,path*"snake_Ωc_in";append=false)
writevtk(Ωc_out,path*"snake_Ωc_out";append=false)
writevtk(Γ,path*"snake_Gamma";append=false)

writevtk(pmodel,path*"snake_pmodel";append=false)
writevtk(Ωbg,path*"snake_Ωbg";append=false, 
  cellfields=[
    "φh"=>φh,
    "cf_IN"=>cf_IN,
    "cf_OUT"=>cf_OUT,
  ],
)

cgids = get_cell_gids(pmodel)
own_cells = own_to_local(cgids)
trians = map(local_views(pmodel),own_cells) do pmodel, cells 
  mgrid = Grid(ReferenceFE{2},pmodel)
  tgrid = Geometry.restrict(mgrid,cells)
  Geometry.BodyFittedTriangulation(pmodel,tgrid,cells)
end
Ωpoly = GridapDistributed.DistributedTriangulation(trians,pmodel)
tcell_to_lcolor = map((t,x) -> x[t],own_cells,subcell_to_lcolor)
tcell_to_color = map((t,x) -> x[t],own_cells,subcell_to_color)

#wait(consistent!(PVector(subcell_to_lcolor,partition(cgids))))
n_lcolors = scan(+,map(maximum,tcell_to_lcolor);init=0,type=:exclusive)
lcolor = map((n,x) -> x .+ n, n_lcolors,tcell_to_lcolor)

writevtk(Ωpoly,path*"snake_Ωpoly";append=false,
  celldata = [
    "lvols" => lcolor,
    "vols" => tcell_to_color
  ]
)

ranks_model = CartesianDiscreteModel((0,1,0,1),(2,2))
writevtk(Triangulation(ReferenceFE{1},ranks_model),path*"snake_ranks_model";append=false)
