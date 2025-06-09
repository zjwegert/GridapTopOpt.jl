// Gmsh project created on Fri Dec 06 15:33:11 2024
//+
SetFactory("OpenCASCADE");
Mesh.SaveAll=1;

//+ 8-----7---------6-----------5
//+ |     |         |           |
//+ |     9--13-14--10		      |
//+ |     |  |   |  |           |
//+ |     11-12 15-16           |
//+ 1-----2---------3-----------4
//+ Area enclosed by 2,3,16,15,14,13,12,11,2 is nondesign domain

//+ Dims
L = 2.0;
H = 0.5;
x0 = 0.5;
l = 0.4;
w = 0.025;
a = 0.3;
b = 0.01;

//+ Mesh sizes
size_f = 0.05;
size_s = 0.0025;

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

//+
Physical Curve("Gamma_f_D", 22) = {8};
Physical Point("Gamma_f_D", 23) = {8, 1};
//+
Physical Curve("Gamma_f_N", 24) = {4};
Physical Point("Gamma_f_N", 25) = {5, 4};
//+
Physical Curve("Gamma_NoSlipTop", 26) = {7, 6, 5};
Physical Point("Gamma_NoSlipTop", 27) = {7, 6};
Physical Curve("Gamma_NoSlipBottom", 28) = {1, 3};
//+
Physical Curve("Gamma_s_D", 29) = {2};
Physical Point("Gamma_s_D", 30) = {2, 3};
//+
Physical Surface("Omega_NonDesign", 31) = {6};
Physical Curve("Omega_NonDesign", 32) = {17, 18, 19, 20, 16, 15, 21};
Physical Point("Omega_NonDesign", 33) = {11, 12, 13, 14, 15, 16};
