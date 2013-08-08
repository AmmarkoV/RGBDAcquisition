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
 ,Result
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
     fprintf(stderr,"%f } , ",mat[i*elements + Result] );
    }
  fprintf(stderr,"} \n ");
}


void printSystemScilab(double * mat,unsigned int totalLines)
{
//M = [8 1 6; 3 5 7; 4 9 2]
  fprintf(stderr,"M=[");
    int i=0;
    for (i=0; i< totalLines; i++)
    {
     fprintf(stderr,"%0.2f ",mat[i*elements + xBxA] );
     fprintf(stderr,"%0.2f ",mat[i*elements + xByA] );
     fprintf(stderr,"%0.2f ",mat[i*elements + xB] );
     fprintf(stderr,"%0.2f ",mat[i*elements + yBxA] );
     fprintf(stderr,"%0.2f ",mat[i*elements + yByA] );
     fprintf(stderr,"%0.2f ",mat[i*elements + yB] );
     fprintf(stderr,"%0.2f ",mat[i*elements + xA] );
     fprintf(stderr,"%0.2f ",mat[i*elements + yA] );
     fprintf(stderr,"%0.2f ;",mat[i*elements + One] );
     //fprintf(stderr,"%f } , ",mat[i*elements + Result] );
    }
  fprintf(stderr,"]\n");
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
     fprintf(stderr,"%f "   ,mat[i*elements + Result] );
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
              { return swapMatrixLines(mat,activeLine,line);  }
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
  double newval;
  double multiplier = 0.0;
  unsigned int i=0,line=0;

  for (line=activeLine; line<totalLines; line++)
  {
   if (line!=activeLine)
   {
    multiplier = mat[line*elements + activeLine];
     for (i=0; i<elements; i++)
      {
        newval = (double) ( mat[activeLine*elements + i] * multiplier ) ;
        mat[line*elements + i] -= newval;
      }
   }
  }
  return 1;
}



/*
#0  1  xA  xB  xC  xD  xE  xF   xG   RESULT
#1      1  xH  xI  xJ  xK  xL   xM   RESULT
#2          1  xN  xO  xP  xQ   xR   RESULT
#3              1  xS  xT  xU   xV   RESULT
#4                  1  xX  xY   xZ   RESULT
#5                     1   x1   x2   RESULT = -x1 * 1 - x2 * ( -x3 * x4 )
#6                          1   x3   RESULT = -x3 * x4
#7                              x4   RESULT =  x4  */

int gatherResult(double * result , double * mat  , unsigned int totalLines )
{
  double calculatedItem = 0.0;
  unsigned int line=0;
  unsigned int i=0;

  line=totalLines-1;
  while (line>=0)
  {
    calculatedItem = 0.0;
    if (line<totalLines-1)
    {
       for (i=line+1; i<totalLines; i++)
       {
          calculatedItem -= mat[line*elements + i];
       }
    } else
    {
     calculatedItem = mat[line*elements + line];
    }

    //We have calculated the result for this raw so we save it
    mat[line*elements + Result] = calculatedItem;
    //And we propagate it to previous lines
    for (i=0; i<totalLines; i++)
        {
         mat[i*elements + line] *= calculatedItem;
        }

    if (line==0) { break; } else
                 { --line; }

  }

  //Store results in resulting matrix
  for (i=0; i<totalLines; i++)
        {
          result[i] = mat[i*elements + Result];
        }
  return 1;
}





int solveLinearSystemGJ(double * result , double * coefficients , int variables , int totalLines )
{
    //Make the system upper diagonal
    unsigned int i=0;
    for (i=0; i<totalLines; i++)
    {
      if (! makeSureNonZero(coefficients,i,totalLines) ) { fprintf(stderr,"Error making sure that we have a non zero element first ( %u ) \n", i ); break; }
      if (! createBaseOne(coefficients,i) )              { fprintf(stderr,"Error creating base one  ( %u ) \n", i ); break; }
      if (! subtractBase(coefficients,i,totalLines) )    { fprintf(stderr,"Error subtracting base  ( %u ) \n" , i ); break; }
    }

    fprintf(stderr,"\n\n");
    printSystemPlain(coefficients,"Echeloned",totalLines);

    //Populate the results matrix
    return gatherResult(result,coefficients,totalLines);
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
      compiledPoints[i*elements + Result] = 0.0;
    }

    fprintf(stderr,"\n\n");
    printSystemPlain(compiledPoints,"Original",pointsNum);

    //fprintf(stderr,"\n\n");
    printSystemMatlab(compiledPoints, pointsNum);
    //printSystemScilab(compiledPoints, pointsNum);

    solveLinearSystemGJ(result3x3Matrix,compiledPoints,elements,pointsNum);



   free(compiledPoints);

   return 1;
}






void testGJSolver()
{
  double * F3x3 = alloc3x3Matrix();    if (F3x3 ==0) { return; }
  double * pointsA = (double *) malloc(sizeof(double) * 2 * 8);
  double * pointsB = (double *) malloc(sizeof(double) * 2 * 8);


//          SOURCE FRAME POINTS                                 TARGET FRAME POINTS
//---------------------------------------------------------------------------------------------
  pointsA[0*2+0]=34;    pointsA[0*2+1]=379;  /* | | */  pointsB[0*2+0]=33;    pointsB[0*2+1]=358;
  pointsA[1*2+0]=178;   pointsA[1*2+1]=379;  /* | | */  pointsB[1*2+0]=84;    pointsB[1*2+1]=374;
  pointsA[2*2+0]=320;   pointsA[2*2+1]=379;  /* | | */  pointsB[2*2+0]=139;   pointsB[2*2+1]=392;
  pointsA[3*2+0]=461;   pointsA[3*2+1]=379;  /* | | */  pointsB[3*2+0]=202;   pointsB[3*2+1]=410;
  pointsA[4*2+0]=605;   pointsA[4*2+1]=379;  /* | | */  pointsB[4*2+0]=271;   pointsB[4*2+1]=432;
  pointsA[5*2+0]=120;   pointsA[5*2+1]=312;  /* | | */  pointsB[5*2+0]=88;    pointsB[5*2+1]=318;
  pointsA[6*2+0]=219;   pointsA[6*2+1]=312;  /* | | */  pointsB[6*2+0]=134;   pointsB[6*2+1]=329;
  pointsA[7*2+0]=319;   pointsA[7*2+1]=312;  /* | | */  pointsB[7*2+0]=184;   pointsB[7*2+1]=342;
//---------------------------------------------------------------------------------------------


  calculateFundamentalMatrix(F3x3 , 8 ,  pointsA,  pointsB );

  print3x3DMatrix("Calculated matrix", F3x3);

  print3x3DScilabMatrix("M",F3x3);

  free3x3Matrix(&F3x3);
  free(pointsA);
  free(pointsB);

  return ;

}





