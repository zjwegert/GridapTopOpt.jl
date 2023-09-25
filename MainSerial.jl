using MPI
if !MPI.Initialized()
    MPI.Init()
end
const comm = MPI.COMM_WORLD;
const root = 0;

using SparseMatricesCSR
using Gridap, Gridap.TensorValues, Gridap.FESpaces
using GridapDistributed
using GridapPETSc
using GridapPETSc: PetscScalar, PetscInt, PETSC,  @check_error_code
using PartitionedArrays
using DelimitedFiles
using LinearAlgebra
using Printf

import GridapDistributed.DistributedCellField
import GridapDistributed.DistributedFESpace
import GridapDistributed.DistributedDiscreteModel
import GridapDistributed.DistributedMeasure

include("Utilities.jl");
include("LevelSet.jl");
include("Setup.jl");
include("ProjectionMethod.jl");
# Currently, to switch between augmented Lagrangian method and 
#   Hilbertian projection method for constrained optimisation
#   replace "OptimiseALM.jl" with "OptimiseHPM.jl". This will
#   change later.
include("OptimiseALM.jl");

const mesh_partition = length(ARGS) == 3 ? parse.(Int,(ARGS[2],ARGS[3])) : 
            length(ARGS) == 4 ? parse.(Int,(ARGS[2],ARGS[3],ARGS[4])) : 
            error("mesh_partition should be of length 2 or 3");

function main(mesh_partition,distribute)
    ranks  = distribute(LinearIndices((prod(mesh_partition),)))
    # Set PETSc options
    options = "-pc_type gamg -ksp_type cg -ksp_error_if_not_converged true 
        -ksp_converged_reason -ksp_rtol 1.0e-12"
    # Get setup function and check exists
    setup = getfield(Main, Symbol(ARGS[1]))
    @assert @isdefined(setup) && isa(setup,Function) "Setup not defined"
    # Create results path
    ptn_str = replace(string(mesh_partition),"("=>"",")"=>"",", "=>"x")
    path = "$(pwd())/Results_serial_p$(ptn_str)_$setup"
    if MPI.Comm_rank(comm) == root
        !isdir(path) ? mkdir(path) : rm(path,recursive=true);
        !isdir(path) ? mkdir(path) : 0;
    end
    # Run and time
    t = PTimer(ranks)
    tic!(t)
    GridapPETSc.with(args=split(options)) do
        main(ranks,mesh_partition,setup,path)
    end
    toc!(t,"Run time")
    t
end

t = with_debug() do distribute
    main(mesh_partition,distribute)
end
display(t)