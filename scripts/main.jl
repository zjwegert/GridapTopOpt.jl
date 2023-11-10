



function main()

    # Problem setup, defining the blocks

    φh
    caches = setup!(optimizer,...) # <- must include first iteration
    for k in 1:nsteps
        J_new,C_new,φh_new = step!(optimizer,...)
        # Logging using results
        # Check convergence criteria, using results
    end

end