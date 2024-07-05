
function main_poisson(;nprocs,                  # Number of processors
                      ncells=(20,20),           # Number of cells
                      options=OPTIONS_CG_JACOBI # PETSc solver options
                      )
  with_mpi() do distribute
    main_poisson(distribute,nprocs,ncells,options)
  end
end

function main_poisson(distribute,nprocs,ncells,options)
  ranks = distribute(LinearIndices((prod(nprocs),)))

  GridapPETSc.with(args=split(options)) do
    domain = (0,1,0,1)
    model = CartesianDiscreteModel(ranks,nprocs,domain,ncells)

    order = 2
    u(x) = (x[1]+x[2])^order
    f(x) = -Δ(u,x)
    reffe = ReferenceFE(lagrangian,Float64,order)
    V = TestFESpace(model,reffe,dirichlet_tags="boundary")
    U = TrialFESpace(V,u)
    Ω = Triangulation(model)
    dΩ = Measure(Ω,2*order)
    a(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
    l(v) = ∫( v*f )dΩ
    op = AffineFEOperator(a,l,U,V)

    solver = PETScLinearSolver()
    uh = solve(solver,op)

    output_file = datadir("poisson")
    writevtk(Ω,output_file,cellfields=["uh"=>uh,"grad_uh"=>∇(uh)])
  end
end
