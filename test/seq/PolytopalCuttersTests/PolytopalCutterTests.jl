module PolytopalCutterTests

using Gridap, Gridap.Geometry, Gridap.Adaptivity, Gridap.ReferenceFEs, Gridap.CellData, Gridap.Fields, Gridap.Arrays, Gridap.Helpers
using GridapEmbedded
using GridapEmbedded.LevelSetCutters
using Test
using ForwardDiff
using Random

using GridapTopOpt

function generate_model(D,n,simplex_bgmodel)
  domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
  cell_partition = (D==2) ? (n,n) : (n,n,n)
  base_model = UnstructuredDiscreteModel((CartesianDiscreteModel(domain,cell_partition)))
  if simplex_bgmodel
    ref_model = refine(base_model, refinement_method = "barycentric")
    model = ref_model.model
    return model
  else
    return base_model
  end
end

function run_test_comparison(;simplex_bgmodel=false,write_vtk=false)
  for _ls in (2,3)
    for _dim in (2,3)
      model = generate_model(_dim,21,simplex_bgmodel)
      order = 1
      reffe = ReferenceFE(lagrangian,Float64,order)
      V_φs = MultiFieldFESpace([TestFESpace(model,reffe) for _ in 1:_ls])

      # LevelSetCutter

      if _ls == 3
        _r = sqrt(97)/40
        if _dim == 2
          f1 = x->sqrt((x[1]-0.4)^2+(x[2]-0.5)^2)-_r
          f2 = x->sqrt((x[1]-0.6)^2+(x[2]-0.5)^2)-_r
          f3 = x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2)-(0.725-0.5)
        elseif _dim == 3
          f1 = x->sqrt((x[1]-0.4)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-_r
          f2 = x->sqrt((x[1]-0.6)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-_r
          f3 = x->sqrt((x[1]-0.5)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-(0.725-0.5)
        else
          error()
        end
        φsh = interpolate([f1,f2,f3],V_φs)
        φh1, φh2, φh3 = φsh
        geo1 = DiscreteGeometry(φh1,model,name="φ1")
        geo2 = DiscreteGeometry(φh2,model,name="φ2")
        geo3 = DiscreteGeometry(φh3,model,name="φ3")
        _all = union(union(geo1,geo2),geo3)
        cutgeo = cut(model,_all)
        cutgeo_facets = cut_facets(model,_all)
      elseif _ls == 2
        if _dim == 2
          f1 = x->sqrt((x[1]-0.4)^2+(x[2]-0.5)^2)-0.205+10^-5
          f2 = x->sqrt((x[1]-0.6)^2+(x[2]-0.5)^2)-0.23+10^-5
        elseif _dim == 3
          f1 = x->sqrt((x[1]-0.4)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-0.205+10^-5
          f2 = x->sqrt((x[1]-0.6)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)-0.23+10^-5
        else
          error()
        end
        φsh = interpolate([f1,f2],V_φs)
        φh1, φh2 = φsh
        geo1 = DiscreteGeometry(φh1,model,name="φ1")
        geo2 = DiscreteGeometry(φh2,model,name="φ2")
        _all = union(geo1,geo2)
        cutgeo = cut(model,_all)
        cutgeo_facets = cut_facets(model,_all)
      else
        error()
      end

      if write_vtk
        writevtk(Triangulation(model),"bgsimplex$(simplex_bgmodel)_dim$(_dim)_ls$(_ls)_background",cellfields=["φh$i"=>φsh[i]  for i in 1:length(V_φs)],
          celldata=["phi$(i)_bh_inoutcut"=>cutgeo.ls_to_bgcell_to_inoutcut[i] for i in 1:length(V_φs)])
        writevtk(cutgeo.subcells,"bgsimplex$(simplex_bgmodel)_dim$(_dim)_ls$(_ls)_subcells",
          celldata=["phi$(i)_inoutcut"=>cutgeo.ls_to_subcell_to_inout[i] for i in 1:length(V_φs)])
        writevtk(cutgeo.subfacets,"bgsimplex$(simplex_bgmodel)_dim$(_dim)_ls$(_ls)_subfacets",
          celldata=["phi$(i)_inoutcut"=>cutgeo.ls_to_subfacet_to_inout[i] for i in 1:length(V_φs)])

        writevtk(BoundaryTriangulation(model,ones(Bool,num_faces(model,_dim-1))),"bgsimplex$(simplex_bgmodel)_dim$(_dim)_ls$(_ls)_background_facets",
          celldata=["phi$(i)_bh_inoutcut"=>cutgeo_facets.ls_to_facet_to_inoutcut[i] for i in 1:length(V_φs)])
        writevtk(cutgeo_facets.subfacets,"bgsimplex$(simplex_bgmodel)_dim$(_dim)_ls$(_ls)_facet_subcells",
          celldata=["phi$(i)_inoutcut"=>cutgeo_facets.ls_to_subfacet_to_inout[i] for i in 1:length(V_φs)])
      end

      # Polytopal cutter
      geo1_fe = DiscreteGeometryFromFEFunction(φh1,model,name="φ1")
      geo2_fe = DiscreteGeometryFromFEFunction(φh2,model,name="φ2")
      if _ls == 2
        _all_fe = union(geo1_fe,geo2_fe)
      elseif _ls == 3
        geo3_fe = DiscreteGeometryFromFEFunction(φh3,model,name="φ3")
        _all_fe = union(union(geo1_fe,geo2_fe),geo3_fe)
      else
        error()
      end

      GridapEmbedded.CSG.test_geometry(_all_fe)

      pcutgeo = cut(PolytopalLevelSetCutter(),model,_all_fe)
      pcutgeo_facets = cut_facets(PolytopalLevelSetCutter(),model,_all_fe)

      if write_vtk
        writevtk(Triangulation(model),"bgsimplex$(simplex_bgmodel)_dim$(_dim)_ls$(_ls)_polytopal_background",cellfields=["φh$i"=>φsh[i]  for i in 1:length(V_φs)],
          celldata=["phi$(i)_bh_inoutcut"=>pcutgeo.ls_to_bgcell_to_inoutcut[i] for i in 1:length(V_φs)])
        writevtk(pcutgeo.subcells,"bgsimplex$(simplex_bgmodel)_dim$(_dim)_ls$(_ls)_polytopal_subcells",
          celldata=["phi$(i)_inoutcut"=>pcutgeo.ls_to_subcell_to_inout[i] for i in 1:length(V_φs)])
        writevtk(pcutgeo.subfacets,"bgsimplex$(simplex_bgmodel)_dim$(_dim)_ls$(_ls)_polytopal_subfacets",
          celldata=["phi$(i)_inoutcut"=>pcutgeo.ls_to_subfacet_to_inout[i] for i in 1:length(V_φs)])

        writevtk(BoundaryTriangulation(model,ones(Bool,num_faces(model,_dim-1))),"bgsimplex$(simplex_bgmodel)_dim$(_dim)_ls$(_ls)_polytopal_background_facets",
          celldata=["phi$(i)_bh_inoutcut"=>pcutgeo_facets.ls_to_facet_to_inoutcut[i] for i in 1:length(V_φs)])
        writevtk(pcutgeo_facets.subfacets,"bgsimplex$(simplex_bgmodel)_dim$(_dim)_ls$(_ls)_polytopal_facet_subcells",
          celldata=["phi$(i)_inoutcut"=>pcutgeo_facets.ls_to_subfacet_to_inout[i] for i in 1:length(V_φs)])
      end

      @test compute_bgcell_to_inoutcut(LevelSetCutter(),model,geo1) == compute_bgcell_to_inoutcut(PolytopalLevelSetCutter(),model,geo1_fe)
      @test compute_bgfacet_to_inoutcut(LevelSetCutter(),model,geo1) == compute_bgfacet_to_inoutcut(PolytopalLevelSetCutter(),model,geo1_fe)

      @test compute_bgcell_to_inoutcut(LevelSetCutter(),model,geo2) == compute_bgcell_to_inoutcut(PolytopalLevelSetCutter(),model,geo2_fe)
      @test compute_bgfacet_to_inoutcut(LevelSetCutter(),model,geo2) == compute_bgfacet_to_inoutcut(PolytopalLevelSetCutter(),model,geo2_fe)

      if _ls == 3
        @test compute_bgcell_to_inoutcut(LevelSetCutter(),model,geo3) == compute_bgcell_to_inoutcut(PolytopalLevelSetCutter(),model,geo3_fe)
        @test compute_bgfacet_to_inoutcut(LevelSetCutter(),model,geo3) == compute_bgfacet_to_inoutcut(PolytopalLevelSetCutter(),model,geo3_fe)
      end

      @test compute_bgcell_to_inoutcut(LevelSetCutter(),model,_all) == compute_bgcell_to_inoutcut(PolytopalLevelSetCutter(),model,_all_fe)
      @test compute_bgfacet_to_inoutcut(LevelSetCutter(),model,_all) == compute_bgfacet_to_inoutcut(PolytopalLevelSetCutter(),model,_all_fe)

    end
  end
end

# Test vertex reorder
function run_test_vertex_reorder()
  function test_reorder_1(pts,perm,n,p0)
    @assert n[3] != 0
    f(x,y) = p0[3] - n[1]/n[3]*(x - p0[1]) - n[2]/n[3]*(y - p0[2])
    proj_pts = [VectorValue(p[1],p[2],f(p[1],p[2])) for p in pts]
    reordered_pts = GridapTopOpt.reorder_vertices!(deepcopy(proj_pts[perm]))

    correct_ordering = false
    for i in 1:length(proj_pts)
      circshift(proj_pts,i-1) == reordered_pts && (correct_ordering = true; break)
    end
    correct_ordering_reverse = false
    for i in 1:length(proj_pts)
      circshift(reverse(proj_pts),i-1) == reordered_pts && (correct_ordering_reverse = true; break)
    end
    @test correct_ordering || correct_ordering_reverse
  end
  function test_reorder_2(pts,perm,n,p0)
    @assert n[2] != 0
    f(x,z) = p0[2] - n[1]/n[2]*(x - p0[1]) - n[3]/n[2]*(z - p0[3])
    proj_pts = [VectorValue(p[1],f(p[1],p[2]),p[2]) for p in pts]
    reordered_pts = GridapTopOpt.reorder_vertices!(deepcopy(proj_pts[perm]))

    correct_ordering = false
    for i in 1:length(proj_pts)
      circshift(proj_pts,i-1) == reordered_pts && (correct_ordering = true; break)
    end
    correct_ordering_reverse = false
    for i in 1:length(proj_pts)
      circshift(reverse(proj_pts),i-1) == reordered_pts && (correct_ordering_reverse = true; break)
    end
    @test correct_ordering || correct_ordering_reverse
  end

  pts_quad = [
    VectorValue(0.0,0.0),
    VectorValue(1.0,0.0),
    VectorValue(0.5,1.0),
  ]
  p0 = VectorValue(1,1,1)
  perm = [1,3,2]
  for n in (VectorValue(0,0,1),VectorValue(0,1,1),VectorValue(1,1,1),rand(VectorValue{3,Float64}))
    test_reorder_1(pts_quad,perm,n,p0)
  end
  for n in (VectorValue(0,1,0),VectorValue(0,1,1),VectorValue(1,1,1),rand(VectorValue{3,Float64}))
    test_reorder_2(pts_quad,perm,n,p0)
  end

  pts_quad = [
    VectorValue(0.0,0.0),
    VectorValue(1.0,0.0),
    VectorValue(1.0,1.0),
    VectorValue(0.0,1.0)
  ]
  p0 = VectorValue(1,1,1)
  perm = [1,2,4,3]
  for n in (VectorValue(0,0,1),VectorValue(0,1,1),VectorValue(1,1,1),rand(VectorValue{3,Float64}))
    test_reorder_1(pts_quad,perm,n,p0)
  end
  for n in (VectorValue(0,1,0),VectorValue(0,1,1),VectorValue(1,1,1),rand(VectorValue{3,Float64}))
    test_reorder_2(pts_quad,perm,n,p0)
  end

  pts_arb = [
    VectorValue(0,0),
    VectorValue(1,0),
    VectorValue(1.5,1),
    VectorValue(1.5,2),
    VectorValue(1,3),
    VectorValue(0,2),
    VectorValue(-0.3,1),
    VectorValue(0.5,1)
  ]

  p0 = VectorValue(1,1,1)
  perm = [4, 7, 8, 5, 3, 1, 6, 2]
  for n in (VectorValue(0,0,1),VectorValue(0,1,1),VectorValue(1,1,1),rand(VectorValue{3,Float64}))
    test_reorder_1(pts_arb,perm,n,p0)
  end
  for n in (VectorValue(0,1,0),VectorValue(0,1,1),VectorValue(1,1,1),rand(VectorValue{3,Float64}))
    test_reorder_2(pts_arb,perm,n,p0)
  end
end

function test_embeddeddiscretization_2d()
  n = 51
  partition = (n,n)
  _R = 0.7
  dom = (-0.707,0.707,-0.707,0.707)
  model_quad = CartesianDiscreteModel(dom,partition)
  model_tet = simplexify(CartesianDiscreteModel(dom,partition))

  reffe = ReferenceFE(lagrangian,Float64,1)

  for model in (model_tet,) # model_quad
    V_φ = TestFESpace(model,reffe)
    φh = interpolate(x->x[1]^2+x[2]^2-_R^2,V_φ)
    geom = DiscreteGeometryFromFEFunction(φh,model)

    cutgeom = cut(PolytopalLevelSetCutter(),model,geom)

    Ωact_in = Triangulation(cutgeom,ACTIVE)
    @test isa(Ωact_in,Gridap.Geometry.BodyFittedTriangulation)
    Ωact_in = Triangulation(cutgeom,ACTIVE_IN)
    @test isa(Ωact_in,Gridap.Geometry.BodyFittedTriangulation)
    Ωact_out = Triangulation(cutgeom,ACTIVE_OUT)
    @test isa(Ωact_out,Gridap.Geometry.BodyFittedTriangulation)
    Ω = Triangulation(model)
    Ω_in = Triangulation(cutgeom,PHYSICAL)
    Ω_in = Triangulation(cutgeom,PHYSICAL_IN)
    Ω_out = Triangulation(cutgeom,PHYSICAL_OUT)
    Γ = EmbeddedBoundary(cutgeom)
    Λ_in = GhostSkeleton(cutgeom)
    Λ_in = GhostSkeleton(cutgeom,ACTIVE_IN)
    Λ_out = GhostSkeleton(cutgeom,ACTIVE_OUT)

    test_triangulation(Ωact_in)
    test_triangulation(Ωact_out)
    test_triangulation(Ω_in)
    test_triangulation(Γ)
    test_triangulation(Λ_in)
    test_triangulation(Λ_out)

    dΩ_in = Measure(Ω_in,2)
    dΓ = Measure(Γ,2)
    n_Γ = get_normal_vector(Γ)

    vol = sum( ∫(1)*dΩ_in )
    surf = sum( ∫(1)*dΓ )

    @test abs(pi*_R^2 - vol) < 1.0e-3
    @test abs(surf - 2*pi*_R) < 1.0e-3

    V_in = FESpace(Ωact_in,reffe,conformity=:H1)
    u(x) = x[1] + x[2]
    u_in = interpolate(u,V_in)
    v_in = FEFunction(V_in,rand(num_free_dofs(V_in)))

    # Check divergence theorem
    a = sum( ∫( ∇(v_in)⋅∇(u_in) )*dΩ_in )
    b = sum( ∫( v_in*n_Γ⋅∇(u_in) )*dΓ )
    @test abs(a-b) < 1.0e-9

    scell_val = (∫( ∇(v_in)⋅∇(u_in) )*dΩ_in)[Ω_in]
    @test isa(scell_val,AppendedArray)
    acell_val, Ωa = Gridap.Geometry.move_contributions(scell_val,Ω_in)
    @test isa(acell_val,AppendedArray)
    @test isa(Ωa,AppendedTriangulation)
    sface_val = (∫( v_in*n_Γ⋅∇(u_in) )*dΓ)[Γ]
    aface_val, Γa = Gridap.Geometry.move_contributions(sface_val,Γ)

    # Check divergence theorem (after moving contributions)
    a = sum( acell_val )
    b = sum( aface_val )
    @test abs(a-b) < 1.0e-9

    @test sum(sface_val) ≈ sum(aface_val)
    @test sum(scell_val) ≈ sum(acell_val)

    dv = get_fe_basis(V_in)
    du = get_trial_fe_basis(V_in)
    scell_val = (∫( ∇(v_in)⋅∇(u_in) )*dΩ_in)[Ω_in]
    acell_val, Ωa = Gridap.Geometry.move_contributions(scell_val,Ω_in)
    @test sum(scell_val) ≈ sum(acell_val)
  end
end

function test_embeddeddiscretization_3d()
  n = 21
  partition = (n,n,n)
  _R = 0.7
  dom = (-0.707,0.707,-0.707,0.707,-0.707,0.707)
  model_quad = CartesianDiscreteModel(dom,partition)
  model_tet = simplexify(CartesianDiscreteModel(dom,partition))

  reffe = ReferenceFE(lagrangian,Float64,1)

  for model in (model_tet,) # model_quad
    V_φ = TestFESpace(model,reffe)
    φh = interpolate(x->x[1]^2+x[2]^2+x[3]^2-_R^2,V_φ)
    geom = DiscreteGeometryFromFEFunction(φh,model)

    cutgeom = cut(PolytopalLevelSetCutter(),model,geom)

    Ωact_in = Triangulation(cutgeom,ACTIVE)
    @test isa(Ωact_in,Gridap.Geometry.BodyFittedTriangulation)
    Ωact_in = Triangulation(cutgeom,ACTIVE_IN)
    @test isa(Ωact_in,Gridap.Geometry.BodyFittedTriangulation)
    Ωact_out = Triangulation(cutgeom,ACTIVE_OUT)
    @test isa(Ωact_out,Gridap.Geometry.BodyFittedTriangulation)
    Ω = Triangulation(model)
    Ω_in = Triangulation(cutgeom,PHYSICAL)
    Ω_in = Triangulation(cutgeom,PHYSICAL_IN)
    Ω_out = Triangulation(cutgeom,PHYSICAL_OUT)
    Γ = EmbeddedBoundary(cutgeom)
    Λ_in = GhostSkeleton(cutgeom)
    Λ_in = GhostSkeleton(cutgeom,ACTIVE_IN)
    Λ_out = GhostSkeleton(cutgeom,ACTIVE_OUT)

    test_triangulation(Ωact_in)
    test_triangulation(Ωact_out)
    test_triangulation(Ω_in)
    test_triangulation(Γ)
    test_triangulation(Λ_in)
    test_triangulation(Λ_out)

    dΩ_in = Measure(Ω_in,2)
    dΓ = Measure(Γ,2)
    n_Γ = get_normal_vector(Γ)

    vol = sum( ∫(1)*dΩ_in )
    surf = sum( ∫(1)*dΓ )

    @test abs(4/3*pi*_R^3 - vol) < 1/n
    @test abs(surf - 4*pi*_R^2) < 1/n

    V_in = FESpace(Ωact_in,reffe,conformity=:H1)
    u(x) = x[1] + x[2]
    u_in = interpolate(u,V_in)
    v_in = FEFunction(V_in,rand(num_free_dofs(V_in)))

    # Check divergence theorem
    a = sum( ∫( ∇(v_in)⋅∇(u_in) )*dΩ_in )
    b = sum( ∫( v_in*n_Γ⋅∇(u_in) )*dΓ )
    @test abs(a-b) < 1.0e-9

    scell_val = (∫( ∇(v_in)⋅∇(u_in) )*dΩ_in)[Ω_in]
    @test isa(scell_val,AppendedArray)
    acell_val, Ωa = Gridap.Geometry.move_contributions(scell_val,Ω_in)
    @test isa(acell_val,AppendedArray)
    @test isa(Ωa,AppendedTriangulation)
    sface_val = (∫( v_in*n_Γ⋅∇(u_in) )*dΓ)[Γ]
    aface_val, Γa = Gridap.Geometry.move_contributions(sface_val,Γ)

    # Check divergence theorem (after moving contributions)
    a = sum( acell_val )
    b = sum( aface_val )
    @test abs(a-b) < 1.0e-9

    @test sum(sface_val) ≈ sum(aface_val)
    @test sum(scell_val) ≈ sum(acell_val)

    dv = get_fe_basis(V_in)
    du = get_trial_fe_basis(V_in)
    scell_val = (∫( ∇(v_in)⋅∇(u_in) )*dΩ_in)[Ω_in]
    acell_val, Ωa = Gridap.Geometry.move_contributions(scell_val,Ω_in)
    @test sum(scell_val) ≈ sum(acell_val)
  end
end

function test_embeddedfacetdiscretization_2d(;write_vtk=false)
  n = 10
  partition = (n,n)
  domain = (0,1,0,1)

  bgmodel = simplexify(CartesianDiscreteModel(domain,partition))
  reffe = ReferenceFE(lagrangian,Float64,1)

  _R = 0.72
  V_φ = TestFESpace(bgmodel,reffe)
  φh = interpolate(x->(x[1]-1)^2+(x[2]-1)^2-_R^2,V_φ)
  geom = DiscreteGeometryFromFEFunction(φh,bgmodel)

  cutgeo = cut(PolytopalLevelSetCutter(),bgmodel,geom)
  cutgeo_facets = cut_facets(PolytopalLevelSetCutter(),bgmodel,geom)

  Ωact = Triangulation(cutgeo,ACTIVE)

  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = FESpace(Ωact,reffe)

  Random.seed!(1234)
  v = FEFunction(V,rand(num_free_dofs(V)))
  _u = interpolate(x->x[1]+x[2],V)

  Ωbg = Triangulation(bgmodel)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γu = EmbeddedBoundary(cutgeo)
  Γf = BoundaryTriangulation(cutgeo_facets,PHYSICAL)
  Γ = lazy_append(Γu,Γf)
  Λ = SkeletonTriangulation(cutgeo_facets,PHYSICAL)

  face_model = get_active_model(Γu)
  Σb = BoundaryTriangulation(Γu)
  Σi = SkeletonTriangulation(Γu)

  test_triangulation(Ω)
  test_triangulation(Γ)
  test_triangulation(Λ)

  dΩ = Measure(Ω,2*order)
  dΓ = Measure(Γ,2*order)
  dΛ = Measure(Λ,2*order)

  n_Γ = get_normal_vector(Γ)
  n_Λ = get_normal_vector(Λ)

  # Check divergence theorem
  a = sum( ∫( ∇(v)⋅∇(_u) )*dΩ )
  b = sum( ∫( v*n_Γ⋅∇(_u) )*dΓ )
  @test abs(a-b) < 1.0e-9

  a = sum( ∫( jump(_u) )*dΛ )
  @test abs(a) < 1.0e-9

  a = sum( ∫( jump(v) )*dΛ )
  @test abs(a) < 1.0e-9

  D = num_cell_dims(bgmodel)
  celldata_Ω = ["bgcell"=>collect(Int,get_glue(Ω,Val(D)).tface_to_mface)]
  celldata_Γ = ["bgcell"=>collect(Int,get_glue(Γ,Val(D)).tface_to_mface)]
  cellfields_Ω = ["v"=>v,"u"=>_u]
  cellfields_Γ = ["normal"=>n_Γ,"v"=>v,"u"=>_u]
  celldata_Λ = [
  "bgcell_left"=>collect(Int,get_glue(Λ.⁺,Val(D)).tface_to_mface),
  "bgcell_right"=>collect(Int,get_glue(Λ.⁻,Val(D)).tface_to_mface)]
  cellfields_Λ = ["normal"=> n_Λ.⁺,"jump_v"=>jump(v),"jump_u"=>jump(_u)]

  if write_vtk
    writevtk(Ωbg,("trian"),append=false)
    writevtk(Ω,("trian_O"),celldata=celldata_Ω,cellfields=cellfields_Ω,append=false)
    writevtk(Γ,("trian_G"),celldata=celldata_Γ,cellfields=cellfields_Γ,append=false)
    writevtk(Λ,("trian_sO"),celldata=celldata_Λ,cellfields=cellfields_Λ,append=false)
    writevtk(Γf,("trian_Gf"),append=false)
  end
end

function test_embeddedfacetdiscretization_3d(;write_vtk=false)
  n = 10
  partition = (n,n,n)
  domain = (0,1,0,1,0,1)
  bgmodel = simplexify(CartesianDiscreteModel(domain,partition))
  reffe = ReferenceFE(lagrangian,Float64,1)

  _R = 0.49
  V_φ = TestFESpace(bgmodel,reffe)
  φh = interpolate(x->(x[1]-0.5)^2+(x[2]-0.5)^2+(x[3]-0.5)^2-_R^2,V_φ)
  geom = DiscreteGeometryFromFEFunction(φh,bgmodel)

  cutgeo = cut(PolytopalLevelSetCutter(),bgmodel,geom)
  cutgeo_facets = cut_facets(PolytopalLevelSetCutter(),bgmodel,geom)

  trian_s = SkeletonTriangulation(bgmodel)
  trian_sΩ = SkeletonTriangulation(trian_s,cutgeo_facets,PHYSICAL_IN,geom)
  trian_sΩo = SkeletonTriangulation(trian_s,cutgeo_facets,PHYSICAL_OUT,geom)

  Γu = EmbeddedBoundary(cutgeo)
  face_model = get_active_model(Γu)
  Σb = BoundaryTriangulation(Γu)
  Σi = SkeletonTriangulation(Γu)

  if write_vtk
    writevtk(trian_s,("trian_s"))
    writevtk(trian_sΩ,("trian_sO"))
    writevtk(trian_sΩo,("trian_sOo"))
  end
end

# run_test_comparison(;simplex_bgmodel=false) # Currently only support simplices
run_test_comparison(;simplex_bgmodel=true)
run_test_vertex_reorder()
test_embeddeddiscretization_2d()
test_embeddeddiscretization_3d()
test_embeddedfacetdiscretization_2d()
test_embeddedfacetdiscretization_3d()

end