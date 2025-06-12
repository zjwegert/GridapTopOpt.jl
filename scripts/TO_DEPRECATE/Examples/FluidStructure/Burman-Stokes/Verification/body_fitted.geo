// Gmsh project created on Mon Jan 20 21:49:56 2025
SetFactory("OpenCASCADE");
Mesh.SaveAll=1;

mesh_size = {{:mesh_size}};

Rectangle(1) = {0, 0, 0, 1, 0.5, 0};
Disk(2) = {0.5, 0.2, -0, 0.1, 0.1};
BooleanDifference{ Surface{1}; Delete; }{ Surface{2}; Delete; }
MeshSize {8, 6, 7, 9, 5} = mesh_size;
//+
Physical Point("Gamma_f_D", 10) = {8, 6};
Physical Curve("Gamma_f_D", 11) = {7};
Physical Point("Gamma_NoSlipTop", 13) = {9};
Physical Curve("Gamma_NoSlipTop", 12) = {9};
Physical Point("Gamma_NoSlipBottom", 14) = {7};
Physical Curve("Gamma_NoSlipBottom", 15) = {6};
Physical Point("Gamma_fsi", 17) = {5};
Physical Curve("Gamma_fsi", 16) = {5};
Physical Curve("Gamma_f_N", 18) = {8};
