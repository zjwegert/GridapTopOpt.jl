SetFactory("OpenCASCADE");
Mesh.SaveAll=1;

Circle(1) = {0, 0, 0, 0.1, 0, 2*Pi};
Circle(2) = {0, 0, 0, 1.0, 0, 2*Pi};

Curve Loop(1) = {2};
Curve Loop(2) = {1};
Plane Surface(1) = {1, 2};

Physical Curve("Gamma_N", 1) = {1};
Physical Point("Gamma_N", 2) = {1};
Physical Point("Gamma_D", 3) = {2};
Physical Curve("Gamma_D", 4) = {2};

MeshSize {1,2} = 0.02;
