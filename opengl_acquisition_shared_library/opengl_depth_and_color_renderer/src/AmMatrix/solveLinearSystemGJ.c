#include "solveLinearSystemGJ.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix3x3Tools.h"
#include "matrix4x4Tools.h"

enum packedPointPrecalcs
{
  xBxA=0
 ,xByA
 ,xB
 ,yBxA
 ,yByA
 ,yB
 ,xA
 ,yA
 ,One
 ,Zero
 ,elements
};

void printSystemMatlab(double * mat,unsigned int totalLines)
{
  fprintf(stderr,"\n\n { ");
    int i=0;
    for (i=0; i< totalLines; i++)
    {
     fprintf(stderr,"{ %0.2f , ",mat[i*elements + xBxA] );
     fprintf(stderr,"%0.2f , ",mat[i*elements + xByA] );
     fprintf(stderr,"%0.2f ,",mat[i*elements + xB] );
     fprintf(stderr,"%0.2f ,",mat[i*elements + yBxA] );
     fprintf(stderr,"%0.2f ,",mat[i*elements + yByA] );
     fprintf(stderr,"%0.2f ,",mat[i*elements + yB] );
     fprintf(stderr,"%0.2f ,",mat[i*elements + xA] );
     fprintf(stderr,"%0.2f ,",mat[i*elements + yA] );
     fprintf(stderr,"%0.2f ,",mat[i*elements + One] );
     fprintf(stderr,"%f } , ",mat[i*elements + Zero] );
    }
  fprintf(stderr,"} \n ");
}

void printSystemPlain(double * mat,char * label , unsigned int totalLines)
{
  fprintf(stderr,"System Printout of %s \n",label);
  fprintf(stderr,"---------------------------------------------\n");
    int i=0;
    for (i=0; i<totalLines; i++)
    {
     fprintf(stderr,"%0.2f ",mat[i*elements + xBxA] );
     fprintf(stderr,"%0.2f ",mat[i*elements + xByA] );
     fprintf(stderr,"%0.2f ",mat[i*elements + xB] );
     fprintf(stderr,"%0.2f ",mat[i*elements + yBxA] );
     fprintf(stderr,"%0.2f ",mat[i*elements + yByA] );
     fprintf(stderr,"%0.2f ",mat[i*elements + yB] );
     fprintf(stderr,"%0.2f ",mat[i*elements + xA] );
     fprintf(stderr,"%0.2f ",mat[i*elements + yA] );
     fprintf(stderr,"%0.2f ",mat[i*elements + One] );
     fprintf(stderr,"%f "   ,mat[i*elements + Zero] );
     fprintf(stderr,"\n");
    }
  fprintf(stderr,"---------------------------------------------\n");
}

int swapMatrixLines(double * mat , unsigned int line1, unsigned int line2)
{
  fprintf(stderr,"Swapping line %u with line %u \n",line1,line2);
  double tmp;
  unsigned int i=0;
  for (i=0; i<elements; i++)
    {
      tmp = mat[line1*elements + i];
      mat[line1*elements + i] = mat[line2*elements + i];
      mat[line2*elements + i] = tmp;
    }
  return 1;
}

int makeSureNonZero(double * mat   , unsigned int activeLine , unsigned int totalLines )
{
  if ( ( mat[activeLine*elements + activeLine] >=  0.0000001 ) ||
       ( mat[activeLine*elements + activeLine] <=  -0.0000001 ) )   { return 1; }

  //If we are here it means our line has a zero leading element
  unsigned int line=0;
  for (line=activeLine+1; line<elements; line++)
    {
      if ( ( mat[line*elements + activeLine] >=  0.0000001 )||
           ( mat[line*elements + activeLine] <=  -0.0000001 ) )
              { swapMatrixLines(mat,activeLine,line);  return 1; }
    }

  return 0;
}

int createBaseOne(double * mat   , unsigned int activeLine)
{

  double divisor = mat[activeLine*elements + activeLine];
  if (divisor == 0.0) { return 0; }

  unsigned int i=0;
  for (i=0; i<elements; i++)
    {
      mat[activeLine*elements + i] /= divisor;
    }

  return 1;
}

int subtractBase(double * mat   , unsigned int activeLine , unsigned int totalLines )
{
  double multiplier = 0.0;

  unsigned int i=0;
  unsigned int line=totalLines;

  for (line=0; line<totalLines; line++)
  {
   if (line!=activeLine)
   {
    multiplier = mat[line*elements + activeLine];
    for (i=0; i<elements; i++)
     {
      mat[line*elements + i] -= (double) ( mat[activeLine*elements + i] * multiplier ) ;
     }
   }
  }
  return 1;
}


int calculateFundamentalMatrix(double * result3x3Matrix , int pointsNum ,  double * pointsA,  double * pointsB )
{
    double * pxA , * pyA , * pxB , * pyB ;
    int elements=10;

    double * compiledPoints = (double * ) malloc(pointsNum * elements * sizeof(double));
    if (compiledPoints==0) { return 0; }

    unsigned int i=0;
    for (i=0; i< pointsNum; i++)
    {
      //Shortcut our vars
      pxA = &pointsA[i*2 + 0]; pyA = &pointsA[i*2 + 1];
      pxB = &pointsB[i*2 + 0]; pyB = &pointsB[i*2 + 1];
      //Make the precalculations for each of the elements
      compiledPoints[i*elements + xBxA] = (*pxB) * (*pxA);
      compiledPoints[i*elements + xByA] = (*pxB) * (*pyA);
      compiledPoints[i*elements + xB]   = (*pxB) ;
      compiledPoints[i*elements + yBxA] = (*pyB) * (*pxA);
      compiledPoints[i*elements + yByA] = (*pyB) * (*pyA);
      compiledPoints[i*elements + yB]   = (*pyB);
      compiledPoints[i*elements + xA]   = (*pxA);
      compiledPoints[i*elements + yA]   = (*pyA);
      compiledPoints[i*elements + One]  = 1;
      compiledPoints[i*elements + Zero] = 0.0;
    }

    fprintf(stderr,"\n\n");
    printSystemPlain(compiledPoints,"Original",pointsNum);
    fprintf(stderr,"\n\n");
    printSystemMatlab(compiledPoints, pointsNum);

    for (i=0; i<pointsNum; i++)
    {
      if (! makeSureNonZero(compiledPoints,i,pointsNum) ) { fprintf(stderr,"Error making sure that we have a non zero element first ( %u ) \n", i ); break; }
      if (! createBaseOne(compiledPoints,i) )             { fprintf(stderr,"Error creating base one  ( %u ) \n", i ); break; }
      if (! subtractBase(compiledPoints,i,pointsNum) )    { fprintf(stderr,"Error subtracting base  ( %u ) \n" , i ); break; }
    }

    fprintf(stderr,"\n\n");
    printSystemPlain(compiledPoints,"Solved",pointsNum);

   free(compiledPoints);

   return 1;
}






void testGJSolver()
{


  double * F3x3 = alloc3x3Matrix();    if (F3x3 ==0) { return; }
  double * pointsA = (double *) malloc(sizeof(double) * 2 * 8);
  double * pointsB = (double *) malloc(sizeof(double) * 2 * 8);

  pointsA[0*2+0]=34;    pointsA[0*2+1]=379;
  pointsA[1*2+0]=178;   pointsA[1*2+1]=379;
  pointsA[2*2+0]=320;   pointsA[2*2+1]=379;
  pointsA[3*2+0]=461;   pointsA[3*2+1]=379;
  pointsA[4*2+0]=605;   pointsA[4*2+1]=379;
  pointsA[5*2+0]=120;   pointsA[5*2+1]=312;
  pointsA[6*2+0]=219;   pointsA[6*2+1]=312;
  pointsA[7*2+0]=319;   pointsA[7*2+1]=312;

  pointsB[0*2+0]=33;    pointsB[0*2+1]=358;
  pointsB[1*2+0]=84;    pointsB[1*2+1]=374;
  pointsB[2*2+0]=139;   pointsB[2*2+1]=392;
  pointsB[3*2+0]=202;   pointsB[3*2+1]=410;
  pointsB[4*2+0]=271;   pointsB[4*2+1]=432;
  pointsB[5*2+0]=88;    pointsB[5*2+1]=318;
  pointsB[6*2+0]=134;   pointsB[6*2+1]=329;
  pointsB[7*2+0]=184;   pointsB[7*2+1]=342;

  calculateFundamentalMatrix(F3x3 , 8 ,  pointsA,  pointsB );


  free3x3Matrix(&F3x3);
  free(pointsA);
  free(pointsB);

  return ;

}





