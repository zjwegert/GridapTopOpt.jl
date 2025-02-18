SetFactory("OpenCASCADE");
Mesh.SaveAll=1;

Cone(1) = {0, 0, -0.1, 0, 0, -0.4, 0.1, 1.0, 2*Pi};
Cone(2) = {0, 0, 0.1, 0, 0, 0.4, 0.1, 1.0, 2*Pi};
Cylinder(3) = {0, 0, -0.1, 0, 0, 0.2, 0.1, 2*Pi};
Cylinder(4) = {0, 0, -0.5, 0, 0, 1, 1, 2*Pi};

BooleanDifference{ Volume{4}; Delete; }{ Volume{2}; Volume{1}; Volume{3}; Delete; }

Physical Surface("Gamma_N", 11) = {4};
Physical Curve("Gamma_N", 12) = {9, 7, 6, 10};
Physical Point("Gamma_N", 13) = {6, 5, 4};
Physical Point("Gamma_D", 14) = {2, 3, 1};
Physical Curve("Gamma_D", 15) = {4, 3, 1, 2};
Physical Surface("Gamma_D", 16) = {1};
//+
MeshSize {2, 1, 4, 3, 5, 6} = 0.02;
