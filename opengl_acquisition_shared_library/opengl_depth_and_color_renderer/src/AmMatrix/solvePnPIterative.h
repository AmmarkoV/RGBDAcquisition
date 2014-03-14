#ifndef SOLVEPNPITERATIVE_H_INCLUDED
#define SOLVEPNPITERATIVE_H_INCLUDED


double solvePNPHomography(double * result3x3Matrix , unsigned int pointsNum ,  double * pointsA,  double * pointsB);

void testPNPSolver();

#endif // SOLVEPNPITERATIVE_H_INCLUDED
