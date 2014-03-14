#include "solvePnPIterative.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "matrix3x3Tools.h"
#include "matrix4x4Tools.h"


//http://ilab.usc.edu/wiki/index.php/Fast_Square_Root
inline float sqrt_fast_approximationAgain(const float x)
{
  union
  {
    int i;
    float x;
  } u;

  u.x = x;
  u.i = (1<<29) + (u.i >> 1) - (1<<22);
  return u.x;
}

double distanceBetween2DPoints(double *x1,double *y1, double *x2, double *y2 )
{
    //sqrt_fast_approximation
  double dx,dy;

  if (*x1>=*x2) { dx=*x1-*x2; } else { dx=*x2-*x1; }
  if (*y1>=*y2) { dy=*y1-*y2; } else { dy=*y2-*y1; }

  fprintf(stderr,"Comparing Point ( %0.2f , %0.2f ) with ( %0.2f , %0.2f )\n",*x1,*y1,*x2,*y2);


  return (double) sqrt_fast_approximationAgain( (dx * dx) + (dy * dy)  );
}




double testHomogaphyError(double * homography , unsigned int pointsNum ,  double * pointsA,  double * pointsB)
{
  double currentError=0.0;
  double * pointsAPtr = pointsA;
  double * pointsBPtr = pointsB;

  double source[3]={0};
  double actualTarget[3]={0};
  double thisTarget[3]={0};

  unsigned int i=0;
  for (i=0; i<pointsNum; i++)
  {
    source[0] = *pointsAPtr; ++pointsAPtr;
    source[1] = *pointsAPtr; ++pointsAPtr;
    source[2] = 1.0;

    actualTarget[0] = *pointsBPtr; ++pointsBPtr;
    actualTarget[1] = *pointsBPtr; ++pointsBPtr;
    actualTarget[2] = 1.0;


    transform2DPointUsing3x3Matrix(thisTarget,homography,source);

    currentError += distanceBetween2DPoints(&actualTarget[0],&actualTarget[1],&thisTarget[0],&thisTarget[1]);
  }
  return currentError;
}



double solvePNPHomography(double * result3x3Matrix , unsigned int pointsNum ,  double * pointsA,  double * pointsB)
{
   double * hypothesis = alloc3x3Matrix();    if (hypothesis ==0) { return 0 ; }
   unsigned int numberOfIterations=0;
   double bestError=10000 , currentError = 100;
   while ( ( bestError > 0.5 ) && ( numberOfIterations < 100 ) )
   {
     random3x3Matrix(hypothesis,0.0,200.0);

     currentError =  testHomogaphyError( hypothesis , pointsNum , pointsA, pointsB);
     if (currentError<bestError)
     {
        bestError = currentError;

        copy3x3Matrix(result3x3Matrix,hypothesis);
     }
   }

  free3x3Matrix(&hypothesis);

  return bestError;
}





void testPNPSolver()
{
  double * F3x3 = alloc3x3Matrix();    if (F3x3 ==0) { return; }
  double * pointsA = (double *) malloc(sizeof(double) * 2 * 8);
  memset(pointsA , 0 ,sizeof(double) * 2 * 8 );
  double * pointsB = (double *) malloc(sizeof(double) * 2 * 8);
  memset(pointsB , 0 ,sizeof(double) * 2 * 8 );


//          SOURCE FRAME POINTS                                 TARGET FRAME POINTS
//---------------------------------------------------------------------------------------------
  unsigned int i=0;
  pointsA[i*2+0]=34;    pointsA[i*2+1]=379;  /* | | */  pointsB[i*2+0]=33;    pointsB[i*2+1]=358; /* | | */ ++i;
  pointsA[i*2+0]=178;   pointsA[i*2+1]=379;  /* | | */  pointsB[i*2+0]=84;    pointsB[i*2+1]=374; /* | | */ ++i;
  pointsA[i*2+0]=320;   pointsA[i*2+1]=379;  /* | | */  pointsB[i*2+0]=139;   pointsB[i*2+1]=392; /* | | */ ++i;
  pointsA[i*2+0]=461;   pointsA[i*2+1]=379;  /* | | */  pointsB[i*2+0]=202;   pointsB[i*2+1]=410; /* | | */ ++i;
  pointsA[i*2+0]=605;   pointsA[i*2+1]=379;  /* | | */  pointsB[i*2+0]=271;   pointsB[i*2+1]=432; /* | | */ ++i;
  pointsA[i*2+0]=120;   pointsA[i*2+1]=312;  /* | | */  pointsB[i*2+0]=88;    pointsB[i*2+1]=318; /* | | */ ++i;
  pointsA[i*2+0]=219;   pointsA[i*2+1]=312;  /* | | */  pointsB[i*2+0]=134;   pointsB[i*2+1]=329; /* | | */ ++i;
  pointsA[i*2+0]=319;   pointsA[i*2+1]=312;  /* | | */  pointsB[i*2+0]=184;   pointsB[i*2+1]=342; /* | | */ ++i;
//---------------------------------------------------------------------------------------------


  double finalError = solvePNPHomography(F3x3 , i /*Number of points*/ ,  pointsA,  pointsB );

  print3x3DMatrix("Calculated matrix using 8 points", F3x3);

  print3x3DScilabMatrix("M",F3x3);

  printf("The error of this estimation was %0.2f\n",finalError);

  free3x3Matrix(&F3x3);
  free(pointsA);
  free(pointsB);

  return ;

}

