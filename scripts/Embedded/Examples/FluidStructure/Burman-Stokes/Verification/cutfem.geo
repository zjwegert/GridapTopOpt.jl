// Gmsh project created on Mon Jan 20 21:49:56 2025
SetFactory("OpenCASCADE");
Mesh.SaveAll=1;

mesh_size = {{:mesh_size}};

Rectangle(1) = {0, 0, 0, 1, 0.5, 0};
MeshSize {1, 2, 3, 4} = mesh_size;

Physical Point("Gamma_f_D", 5) = {1, 4};
Physical Curve("Gamma_f_D", 6) = {4};
Physical Point("Gamma_NoSlipTop", 7) = {3};
Physical Curve("Gamma_NoSlipTop", 8) = {3};
Physical Point("Gamma_NoSlipBottom", 9) = {2};
Physical Curve("Gamma_NoSlipBottom", 10) = {1};
Physical Curve("Gamma_f_N", 11) = {2};