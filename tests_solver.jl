using Gridap, Gridap.TensorValues, Gridap.Geometry
using GridapDistributed
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code
using PartitionedArrays
using SparseMatricesCSR

import GridapDistributed.DistributedCellField
import GridapDistributed.DistributedFESpace
import GridapDistributed.DistributedDiscreteModel
import GridapDistributed.DistributedMeasure
import GridapDistributed.DistributedTriangulation

include("Utilities.jl"); # <- For some useful functions 

function elastic_compliance(φh,g,C,solve_data,interp,t,solver)
    I = interp.I; dΩ=solve_data.dΩ; dΓ_N=solve_data.dΓ_N;
    ## Weak form
    a(u,v) = ∫((I ∘ φh)*(C ⊙ ε(u) ⊙ ε(v)))dΩ
    l(v) = ∫(v ⋅ g)dΓ_N
    ## Solve finite element problem
    tic!(t)
    op = AffineFEOperator(a,l,solve_data.U,solve_data.V,solve_data.assem)
    toc!(t,"Assembly time")
    K = op.op.matrix;
    ## Solve
    tic!(t)
    uh = solve(solver,op)
    toc!(t,"Solve time")
    u = correct_ghost_layout(uh,K.col_partition)
    ## Compute J and v_J
    J = dot(u,(K*u))
    tic!(t)
    v_J = interpolate(-C ⊙ ε(uh) ⊙ ε(uh),solve_data.V_L2)
    toc!(t,"Interpolate shape sensitivity")
    return J,v_J,uh
end

function fe_setup(ranks,mesh_partition,order::T,coord_max::NTuple{3,M},
        el_size::NTuple{3,T}) where {T<:Integer,M<:AbstractFloat}
    ## Model
    dom = (0,coord_max[1],0,coord_max[2],0,coord_max[3]);
    model = CartesianDiscreteModel(ranks,mesh_partition,dom,el_size);
    ## Define Γ_N and Γ_D
    xmax,ymax,zmax = coord_max
    prop_Γ_N = 0.4
    f_Γ_D(x) = (x[1] ≈ 0.0) ? true : false;
    f_Γ_N(x) = (x[1] ≈ xmax && ymax/2-ymax*prop_Γ_N/4 - eps() <= x[2] <= 
        ymax/2+ymax*prop_Γ_N/4 + eps() && zmax/2-zmax*prop_Γ_N/4 - eps() <= x[3]
        <= zmax/2+zmax*prop_Γ_N/4 + eps()) ? true : false;
    update_labels!(1,model,f_Γ_D,coord_max,"Gamma_D")
    update_labels!(2,model,f_Γ_N,coord_max,"Gamma_N")
    ## Triangulations and measures
    Ω = Triangulation(model)
    Γ_N = BoundaryTriangulation(model,tags="Gamma_N")
    dΩ = Measure(Ω,2order)
    dΓ_N = Measure(Γ_N,2order)
    ## Spaces
    reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
    V = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["Gamma_D"],
        dirichlet_masks=[(true,true,true)])
    U = TrialFESpace(V,[VectorValue(0.0,0.0,0.0)])
    ## Assembler
    Tm=SparseMatrixCSR{0,PetscScalar,PetscInt}
    Tv=Vector{PetscScalar}
    assem=SparseMatrixAssembler(Tm,Tv,U,V)
    # Space for shape sensitivities
    reffe_scalar = ReferenceFE(lagrangian,Float64,order)
    V_L2 = TestFESpace(model,reffe_scalar;conformity=:L2)
    # FE space for LSF 
    V_φ = TestFESpace(model,reffe_scalar;conformity=:H1)
    ## Collect data
    solve_data = (U=U,V=V,assem=assem,V_L2=V_L2,dΩ=dΩ,dΓ_N=dΓ_N)
    return model,Ω,V_φ,solve_data
end

function _isotropic_3d(E::M,nu::M) where M<:AbstractFloat
    λ = E*nu/((1+nu)*(1-2nu)); μ = E/(2*(1+nu))
    C =[λ+2μ   λ      λ      0      0      0
        λ     λ+2μ    λ      0      0      0
        λ      λ     λ+2μ    0      0      0
        0      0      0      μ      0      0
        0      0      0      0      μ      0
        0      0      0      0      0      μ];
    return SymFourthOrderTensorValue(
        C[1,1], C[6,1], C[5,1], C[2,1], C[4,1], C[3,1],
        C[1,6], C[6,6], C[5,6], C[2,6], C[4,6], C[3,6],
        C[1,5], C[6,5], C[5,5], C[2,5], C[4,5], C[3,5],
        C[1,2], C[6,2], C[5,2], C[2,2], C[4,2], C[3,2],
        C[1,4], C[6,4], C[5,4], C[2,4], C[4,4], C[3,4],
        C[1,3], C[6,3], C[5,3], C[2,3], C[4,3], C[3,3])
end

############################################################################################

function ksp_setup(ksp,dim,nloc,coords)
    rtol = PetscScalar(1.e-12)
    atol = GridapPETSc.PETSC.PETSC_DEFAULT
    dtol = GridapPETSc.PETSC.PETSC_DEFAULT
    maxits = GridapPETSc.PETSC.PETSC_DEFAULT
  
    @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
    @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPCG)
    @check_error_code GridapPETSc.PETSC.KSPSetTolerances(ksp[], rtol, atol, dtol, maxits)
  
    pc = Ref{GridapPETSc.PETSC.PC}()
    @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
    @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCGAMG)
    @check_error_code GridapPETSc.PETSC.PCSetCoordinates(pc[],dim,nloc,coords)
end

function get_coords(trian::DistributedTriangulation{Dc}) where Dc
    coords = map(local_views(trian)) do trian
        node_coords = Gridap.Geometry.get_node_coordinates(trian)
        coords = Vector{PetscScalar}(undef,Dc*length(node_coords))
        k = 1
        for p in node_coords
          for d in 1:Dc
            coords[k] = p[d]
            k += 1
          end
        end
        return coords
    end
    coords = PartitionedArrays.getany(coords)
    return Dc, length(coords)÷Dc, coords
end

############################################################################################

function main(mesh_partition,distribute)
    ranks  = distribute(LinearIndices((prod(mesh_partition),)))
    options = "-ksp_error_if_not_converged true -ksp_converged_reason"
    el_size = (20,20,20)
    coord_max = (2.,1.,1.)
    order = 1;
    Δ = coord_max./el_size;
    path = (@__DIR__)*"/Results/LinearElastic3DSolverTesting";
    t = PTimer(ranks)
    GridapPETSc.with(args=split(options)) do
        model,Ω,V_φ,solve_data = fe_setup(ranks,mesh_partition,order,coord_max,el_size)
        interp = SmoothErsatzMaterialInterpolation(η = 2*maximum(Δ))
        C = _isotropic_3d(1.,0.3)
        g = VectorValue(0.,0.,-1.0)
        φh = interpolate(x->-sqrt((x[1]-0.5)^2+(x[2]-0.5)^2+(x[3]-0.5)^2)+0.25,V_φ)

        # Solver
        dim, nloc, coords = get_coords(Ω)
        _ksp_setup(ksp) = ksp_setup(ksp,dim,nloc,coords)
        solver = PETScLinearSolver(_ksp_setup)

        J,v_J,uh = elastic_compliance(φh,g,C,solve_data,interp,t,solver)
        display("Objective = $J")
        writevtk(Ω,path,cellfields=["phi"=>φh,"H(phi)"=>(interp.H ∘ φh),"|nabla(phi))|"=>(norm ∘ ∇(φh)),"uh"=>uh])
    end
    t
end

t = with_debug() do distribute
    main((1,1,1),distribute)
end;
display(t)
