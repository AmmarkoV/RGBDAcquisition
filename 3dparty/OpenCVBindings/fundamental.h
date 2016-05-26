#ifndef FUNDAMENTAL_H_INCLUDED
#define FUNDAMENTAL_H_INCLUDED

#include "primitives.h"
double testFundamentalMatrixPair( double x1, double y1 , double * F , double x2 , double y2 );


double testAverageFundamentalMatrixForAllPairs(  struct Point2DCorrespondance *  correspondances , double * F  );
#endif // FUNDAMENTAL_H_INCLUDED
