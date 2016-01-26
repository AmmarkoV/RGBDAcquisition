#include "dericheRecursiveGaussian.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>



int deriche1DPass(
                   unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                   unsigned char * target,  unsigned int targeteWidth , unsigned int targetHeight ,
                   unsigned int x, unsigned int y,
                   unsigned int direction,
                   float sigma ,
                   unsigned int order
                 )
{
 //First we compute the number of pixels in the line
 unsigned char * sourceStart = source + ( sourceWidth * y ) + x ;
 unsigned char * sourcePTR = sourceStart;
 unsigned char * sourceLimit = 0;
 unsigned char * targetPTR = target;
 unsigned int offsetToNextPixel=0;
 unsigned int bufferSize=0;

 if (direction==0)
    {
      sourceLimit = source + ( sourceWidth * y ) + sourceWidth ;
      offsetToNextPixel=1;
      bufferSize=sourceWidth;
    } else
 if (direction==1)
    {
      sourceLimit = source + ( sourceWidth * (sourceHeight-1) ) + x;
      offsetToNextPixel=sourceWidth;
      bufferSize=sourceHeight;
    }

 unsigned int i=0;
 float * y1 = ( float * ) malloc(sizeof(float) * bufferSize);
 if (y1==0) { return 0; }
 for ( i=0; i<bufferSize; i++ ) { y1[0]=0.0; }

 float * y2 = ( float * ) malloc(sizeof(float) * bufferSize);
 if (y2==0) { free(y1); return 0; }
 for ( i=0; i<bufferSize; i++ ) { y2[0]=0.0; }

//--- Now we prepare the constants which are applied in the equations


float alpha = 0.0e+00;

if ( sigma > 0.0e+00 )
    {
      alpha = 1.695e+00/sigma;
    }

float e =  exp(-alpha);
float e2 = exp(-2 * alpha);

float p_filter[5]={0};
float n_filter[5]={0};

float d2,d1,n0,n1,n2,in0,in1,in2,out0,out1,out2,norm;
// Computing constants that will be applied depending on the order we want
switch(order)
 {
		   case 0 :
			      norm = (1 - e) * (1 - e) / (1 + 2 * alpha * e - e2);

                  p_filter[0] = -e2;
			      p_filter[1] = 2 * e;
			      p_filter[2] = norm;
			      p_filter[3] = norm * (alpha - 1) * e;
			      p_filter[4] = 0;

			      n_filter[0] = -e2;
			      n_filter[1] = 2 * e;
			      n_filter[2] = 0;
			      n_filter[3] = norm * (alpha + 1) * e;
			      n_filter[4] = -norm * e2;
			      break;

		   case 1 :
			      norm = (1 - e) * (1 - e) * (1 - e) / 2 / (1 + e);

                  p_filter[0] = -e2;
			      p_filter[1] = 2 * e;
			      p_filter[2] = 0;
			      p_filter[3] = -norm;
			      p_filter[4] = 0;

                  n_filter[0] = -e2;
			      n_filter[1] = 2 * e;
			      n_filter[2] = 0;
			      n_filter[3] = norm;
			      n_filter[4] = 0;
			      break;

		   case 2 :
			      norm = (1 - e)*(1 - e)*(1 - e)*(1 - e) / (1 + 2 * e - 2 * e * e2 - (e2)*(e2));

                  p_filter[0] = -e2;
			      p_filter[1] = 2 * e;
			      p_filter[2] = -norm;
			      p_filter[3] = norm * (1 + alpha * (1 - e2) / (2 * alpha * e)) * e;
			      p_filter[4] = 0;

                  n_filter[0] = -e2;
			      n_filter[1] = 2 * e;
			      n_filter[2] = 0;
			      n_filter[3] = -norm * (1 - alpha * (1 - e2) / (2 * alpha * e)) * e;
			      n_filter[4] = norm * e2;
			      break;

           default :
            fprintf(stderr,"Unsupported order ( only signal (0) , first derivative [0] , second derivative [1] supported \n");

           break;
 };



// We apply to the one dimensional direction of pixels

d2 = p_filter[0];
d1 = p_filter[1];
n0 = p_filter[2];
n1 = p_filter[3];
n2 = p_filter[4];   // Note this is always == 0


sourcePTR = sourceStart;
in1 = (float) *sourcePTR;
in2 = (float) *sourcePTR;

out1 = (n2 + n1 + n0)*in1/(1.0e+00-d1-d2);
out2 = (n2 + n1 + n0)*in1/(1.0e+00-d1-d2);

for ( i=0; i<bufferSize; i++ )
{
	in0  =  (float) *sourcePTR;
	sourcePTR=sourcePTR+offsetToNextPixel;

	out0 = n2*in2 + n1*in1 + n0*in0 + d1*out1 + d2*out2;

    in2  = in1;
	in1  = in0;
	out2 = out1;
	out1 = out0;

    y1[i] = out0;
}


//
//--- Now let's apply the Deriche equation (16)
//
//    We run right to left accross the line of pixels
//
d2 = n_filter[0];
d1 = n_filter[1];
n0 = n_filter[2];   // Always == 0
n1 = n_filter[3];
n2 = n_filter[4];


sourcePTR = sourceLimit;
in1 = (float) *sourcePTR;
in2 = (float) *sourcePTR;

out1 = (n2 + n1 + n0)*in1 / (1.0e+00-d1-d2);
out2 = (n2 + n1 + n0)*in1 / (1.0e+00-d1-d2);

for ( i=(bufferSize-1); i>0; i-- )
{
	  in0  =  (float) *sourcePTR;
	  sourcePTR=sourcePTR-offsetToNextPixel;

	  in0  =  (float) *sourcePTR;
	  sourcePTR=sourcePTR+offsetToNextPixel;

      out0 = n2*in2 + n1*in1 + n0*in0 + d1*out1 + d2*out2;

      in2  = in1;
	  in1  = in0;
	  out2 = out1;
	  out1 = out0;

	  y2[i] = out0;
}


// The final result is the summation of the vectors produced by equations 15
// and 16 of Deriche's paper.
float yOut;


targetPTR = target;
unsigned char* targetLimit = target;
for (i=0; i<bufferSize; i++)
{
  float yOut = y1[i] + y2[i];
  *targetPTR = (unsigned char) yOut;
  targetPTR+=offsetToNextPixel;
}


 //---- End of procedure
 free(y1);
 free(y2);

 return;
}



int dericheRecursiveGaussianGray(
                                  unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight , unsigned int channels,
                                  unsigned char * target,  unsigned int targeteWidth , unsigned int targetHeight ,
                                  float sigma , unsigned int order
                                )
{
  if (channels!=1)
  {
     fprintf(stderr,"dericheRecursiveGaussianGray only accepts grayscale images..\n");
     return 0;
  }

  unsigned int x=0,y=0;

     y=0;
     for (x=0; x<sourceWidth; x++)
       {
           deriche1DPass(
                          source,  sourceWidth , sourceHeight ,
                          target , targeteWidth , targetHeight ,
                          x,y,  0, // X direction
                          sigma , order
                         );
       }

     x=0;
     for (y=0; y<sourceHeight; y++)
       {

           deriche1DPass(
                          target , targeteWidth , targetHeight  ,
                          target , targeteWidth , targetHeight ,
                          x,y, 1, // Y direction
                          sigma , order
                         );
       }

  return 1;
}
