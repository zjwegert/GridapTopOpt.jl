SetFactory("OpenCASCADE");
Mesh.SaveAll=1;

Cylinder(1) = {0, 0, -0.15, 0, 0, 0.3, 0.1, 2*Pi};
Cylinder(2) = {0, 0, -0.15, 0, 0, 0.3, 1, 2*Pi};

BooleanDifference{ Volume{2}; Delete; }{ Volume{1}; Delete; }

Physical Surface("Gamma_N", 1) = {1};
Physical Curve("Gamma_N", 2) = {1,2,3};
Physical Point("Gamma_N", 3) = {1,2};
Physical Point("Gamma_D", 4) = {3,4};
Physical Curve("Gamma_D", 5) = {4,5,6};
Physical Surface("Gamma_D", 6) = {2};
//+
MeshSize {1,2,3,4} = 0.015;

//+015;
