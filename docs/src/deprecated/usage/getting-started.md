# Getting started



## Known issues
- PETSc's GAMG preconditioner breaks for split Dirichlet DoFs (e.g., x constrained while y free for a single node). There is no simple fix for this. We recommend instead using MUMPS or another preconditioner for this case.
- There is a memory leak in Julia (ver>1.9) IO that affects `write_vtk`. This is a known problem and for now we occasionally use `gc()`. This will hopefully be fixed in an upcoming release. 
