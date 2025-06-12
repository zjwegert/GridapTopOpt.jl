SetFactory("OpenCASCADE");
Mesh.SaveAll=1;

//+ Dims
L = 4.0;
H = 1.0;
x0 = 2;
l = 1.0;
w = 0.1;
a = 0.7;
b = 0.1;
cw = 0.1; //+ Width between wall and non-designable domain

//+ Mesh sizes
size_f = 0.05;
size_s = 0.01;

//+ Main area
Point(1) = {0,0,0,size_f};
Point(2) = {x0-l/2,0,0,size_s};
Point(3) = {x0+l/2,0,0,size_s};
Point(4) = {L,0,0,size_f};
Point(5) = {L,H,0,size_f};
Point(6) = {x0+l/2,H,0,size_f};
Point(7) = {x0-l/2,H,0,size_f};
Point(8) = {0,H,0,size_f};
Point(9) = {x0-l/2,a,0,size_s};
Point(10) = {x0+l/2,a,0,size_s};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,1};

//+ Non-designable region
Point(11) = {x0-l/2,b,0,size_s};
Point(12) = {x0-w/2,b,0,size_s};
Point(13) = {x0-w/2,a,0,size_s};
Point(14) = {x0+w/2,a,0,size_s};
Point(15) = {x0+w/2,b,0,size_s};
Point(16) = {x0+l/2,b,0,size_s};

Line(9) = {7,9};
Line(10) = {6,10};
Line(11) = {9,11};
Line(12) = {10,16};
Line(13) = {9,13};
Line(14) = {14,10};

Line(15) = {2,11};
Line(16) = {11,12};
Line(17) = {12,13};
Line(18) = {13,14};
Line(19) = {14,15};
Line(20) = {15,16};
Line(21) = {16,3};

//+
Curve Loop(1) = {7, 8, 1, 15, -11, -9};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {6, 9, 13, 18, 14, -10};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {5, 10, 12, 21, 3, 4};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {13, -17, -16, -11};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {19, 20, -12, -14};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {18, 19, 20, 21, -2, 15, 16, 17};
//+ Non-design domain
Plane Surface(6) = {6};

//+ 3D
Extrude {0, 0, cw} {
  Surface{1}; Surface{2}; Surface{3}; Surface{5}; Surface{6}; Surface{4};
}
Extrude {0, 0, H-2*cw} {
  Surface{13}; Surface{33}; Surface{19}; Surface{28}; Surface{32}; Surface{25};
}
Extrude {0, 0, cw} {
  Surface{40}; Surface{56}; Surface{44}; Surface{53}; Surface{49}; Surface{60};
}

//+ Tags
Physical Point("Gamma_f_D", 133) = {1, 19, 35, 51, 50, 34, 18, 8};
Physical Curve("Gamma_f_D", 134) = {8, 23, 60, 26, 25, 62, 63, 100, 99, 97};
Physical Surface("Gamma_f_D", 135) = {8, 35, 62};
//+
Physical Curve("Gamma_f_N", 137) = {51, 95};
Physical Surface("Gamma_f_N", 138) = {59, 24, 86};
//+
Physical Point("Gamma_Top", 139) = {17, 33, 23, 41, 27, 47};
Physical Curve("Gamma_Top", 140) = {24, 61, 77, 35, 34, 76, 22, 59, 96, 125, 92, 44, 43, 91, 128};
Physical Surface("Gamma_Top", 141) = {34, 61, 7, 14, 45, 81, 84, 57, 20};
//+
Physical Point("Gamma_Bottom", 149) = {2, 4, 52, 59, 3, 64, 30, 48, 36, 20, 29, 46};
Physical Curve("Gamma_Bottom", 150) = {94, 131, 115, 117, 101, 102, 65, 28, 1, 27, 2, 47, 3, 50, 49, 93, 130, 64, 55, 88, 90};
Physical Surface("Gamma_Bottom", 151) = {85, 58, 23, 72, 29, 9, 63, 36, 55};
//+
Physical Point("Gamma_Right", 152) = {53, 58, 61, 56, 55, 54, 60, 57};
Physical Curve("Gamma_Right", 153) = {119, 120, 112, 114, 110, 124, 121, 123, 106, 107, 127, 104, 116, 132};
Physical Surface("Gamma_Right", 154) = {67, 77, 80, 83, 87, 75};
//+
Physical Point("Gamma_Left", 155) = {16, 11, 9, 12, 15, 14, 10, 13};
Physical Curve("Gamma_Left", 156) = {4, 12, 10, 14, 20, 16, 19, 17, 11, 13, 18, 9, 15, 21};
Physical Surface("Gamma_Left", 157) = {3, 5, 4, 2, 6, 1};
//+
Physical Point("Gamma_TopCorners", 159) = {5, 7, 6, 62, 63, 49};
Physical Curve("Gamma_TopCorners", 158) = {98, 126, 129, 5, 6, 7};
//+
Physical Point("Omega_NonDesign", 145) = {37, 21, 32, 31, 40, 44, 45, 28, 39, 42, 25, 24};
Physical Curve("Omega_NonDesign", 146) = {66, 75, 73, 83, 86, 85, 84, 74, 78, 71, 54, 57, 58, 53, 39, 79};
Physical Surface("Omega_NonDesign", 147) = {54, 37, 32, 56, 50, 42, 46};
Physical Volume("Omega_NonDesign", 148) = {11};
//+

Box(19) = {0, 0, 0, 4, 1, 0.5};
BooleanDifference{ Volume{1}; Volume{7}; Volume{13}; Volume{6}; Volume{5}; Volume{2}; Volume{9}; Volume{17}; Volume{3}; Volume{12}; Volume{18}; Volume{16}; Volume{14}; Volume{15}; Volume{10}; Volume{11}; Volume{8}; Volume{4}; Delete; }{ Volume{19}; Delete; }
Physical Point("Gamma_f_D", 133) += {70, 65};
Physical Curve("Gamma_f_D", 134) += {140, 134, 141};
Physical Surface("Gamma_f_D", 135) += {90};
//+
Physical Curve("Gamma_f_N", 137) += {156};
Physical Surface("Gamma_f_N", 138) += {103};
//+
Physical Curve("Gamma_Top", 140) += {135, 146, 155};
Physical Surface("Gamma_Top", 141) += {101, 95, 88};
//+
Physical Point("Gamma_Bottom", 149) += {76, 77, 69};
Physical Curve("Gamma_Bottom", 150) += {157, 139, 168, 161, 144};
Physical Surface("Gamma_Bottom", 151) += {104, 113, 94};
//+
Physical Volume("Omega_NonDesign", 148) += {11};
Physical Curve("Omega_NonDesign", 146) += {164, 169, 152, 153, 143, 162};
Physical Surface("Omega_NonDesign", 147) += {99, 111, 107, 109, 112, 93, 105};
//+
Physical Point("Gamma_Symm", 170) = {67, 72};
Physical Curve("Gamma_Symm", 171) = {147, 159, 148, 150, 137, 136};
Physical Surface("Gamma_Symm", 172) = {102, 89, 114, 108, 96};
//+
Physical Point("Gamma_TopCorners", 159) += {66, 71, 75};
Physical Curve("Gamma_TopCorners", 158) += {154, 145, 133};
//+
Physical Point("Gamma_Symm_NonDesign", 173) = {73, 74, 80, 79, 78, 68};
Physical Curve("Gamma_Symm_NonDesign", 174) = {149, 163, 165, 167, 166, 138, 168, 158};
Physical Surface("Gamma_Symm_NonDesign", 175) = {110};
//+
MeshSize {72, 73, 74, 67, 68, 80, 79, 78, 77, 69} = size_s;
MeshSize {75, 76, 71, 66, 65, 70} = size_f;
