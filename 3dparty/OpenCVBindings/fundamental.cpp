#include "fundamental.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double testFundamentalMatrixPair( double x1, double y1 , double * F , double x2 , double y2 )
{
  //{ x1 , y1 , 1 } . { { a , b , c } , { d , e , f } , {g , h , i } } . { {x2} ,  {y2} , {1} }
  // i + c * x1 + f * y1 + x2 * (g + a * x1 + d * y1) + (h + b * x1 + e * y1) * y2;
  // {i + c x1 + f y1 + x2 (g + a x1 + d y1) + (h + b x1 + e y1) y2}
  //    |  a-0   b-1   c-2  |
  //  H |  d-3   e-4   f-5  |
  //    |  g-6   h-7   i-8  |


 return  F[8] + ( ( F[2] * x1 ) + ( F[5] * y1) ) + x2 * ( F[6] + ( F[0] * x1 ) + ( F[3] * y1) ) + ( ( F[7] + F[1] * x1 + F[4] * y1) * y2);

}



double testAverageFundamentalMatrixForAllPairs(  struct Point2DCorrespondance *  correspondances , double * F  )
{
  double sumError = 0.0;
  for (unsigned int i=0; i<correspondances->listCurrent; i++)
  {
       double reprojError= testFundamentalMatrixPair( correspondances->listSource[i].x ,correspondances->listSource[i].y  ,
                                                            F ,
                                                            correspondances->listTarget[i].x , correspondances->listTarget[i].y  );
      // fprintf(stderr," %u pair = %0.2f \n", i , reprojError  );

       sumError += fabs(reprojError);
  }

  sumError = sumError/correspondances->listCurrent;
  fprintf(stderr," Avg reproj error %0.2f \n", sumError  );


}
