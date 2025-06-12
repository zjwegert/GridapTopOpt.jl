
using Gridap
using Gridap.Geometry, Gridap.FESpaces
using GridapGmsh

using GridapEmbedded
using GridapEmbedded.Interfaces, GridapEmbedded.LevelSetCutters

using GridapDistributed, PartitionedArrays

using GridapTopOpt

np = (2,1)
ranks = DebugArray([1,2])

model = UnstructuredDiscreteModel(GmshDiscreteModel(ranks,(@__DIR__)*"/mesh_low_res_3d.msh"))

reffe_m = ReferenceFE(lagrangian,Float64,1)
M = FESpace(model,reffe_m)

L = 4.0
H = 1.0
x0 = 2
l = 1.0
w = 0.05
a = 0.7
b = 0.05
cw = 0.1
vol_D = L*H

hₕ = get_element_diameter_field(model)
hmin = minimum(get_element_diameters(model))

_e = 1/3*hmin
f0((x,y,z),a,b) = max(2/a*abs(x-x0),1/(b/2+1)*abs(y-b/2+1),2/(H-2cw)*abs(z-H/2))-1
f1((x,y,z),q,r) = - cos(q*π*x)*cos(q*π*y)*cos(q*π*z)/q - r/q
fin(x) = f0(x,l*(1+_e),a*(1+_e))
fsolid(x) = min(f0(x,l*(1+_e),b*(1+_e)),f0(x,w*(1+_e),a*(1+_e)))
fholes((x,y,z),q,r) = max(f1((x,y,z),q,r),f1((x-1/q,y,z),q,r))
lsf(x) = fin(x) # Test low res geometry for shape opt
# lsf(x) = min(max(fin(x),fholes(x,5,0.5)),fsolid(x))
mh = interpolate(lsf,M)

geo = DiscreteGeometry(mh,model)
cutgeo = cut(model,geo)

Ωin  = Triangulation(cutgeo,PHYSICAL)
Ωout = Triangulation(cutgeo,PHYSICAL_OUT)

Ωain = Triangulation(cutgeo,ACTIVE)
Ωaout = Triangulation(cutgeo,ACTIVE_OUT)


