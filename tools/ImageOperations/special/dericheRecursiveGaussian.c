#include "dericheRecursiveGaussian.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../tools/imageMatrix.h"

 struct derichePrecalculations
 {
  float p_filter[5];
  float n_filter[5];
 };


int dericheDoPrecalculations(  struct derichePrecalculations * derp , float sigma , unsigned int order )
{
//--- Now we prepare the constants which are applied in the equations
float alpha = 0.0e+00;
if ( sigma > 0.0e+00 )
    {
      alpha = 1.695e+00/sigma;
    }

/*  dividend / divisor = quotient */
float dividend , divisor;
float e =  exp(-alpha);
float e2 = exp(-2 * alpha);

float d2,d1,n0,n1,n2,in0,in1,in2,out0,out1,out2,norm;
// Computing constants that will be applied depending on the order we want
switch(order)
 {
		   case 0 :
                  dividend=(1 - e) * (1 - e);
                  divisor=(1 + 2 * alpha * e) - e2;
			      norm =  (float) dividend/divisor;

                  derp->p_filter[0] = -e2;
			      derp->p_filter[1] = 2 * e;
			      derp->p_filter[2] = norm;
			      derp->p_filter[3] = norm * (alpha - 1) * e;
			      derp->p_filter[4] = 0;

			      derp->n_filter[0] = -e2;
			      derp->n_filter[1] = 2 * e;
			      derp->n_filter[2] = 0;
			      derp->n_filter[3] = norm * (alpha + 1) * e;
			      derp->n_filter[4] = -norm * e2;
			      break;

		   case 1 :
		        //norm = (1 - e) * (1 - e) * (1 - e) / 2 / (1 + e);
                  dividend=(1 - e) * (1 - e) * (1 - e);
                  divisor= (float) 2 / (1 + e) ;
			      norm =  (float) dividend/divisor;

                  derp->p_filter[0] = -e2;
			      derp->p_filter[1] = 2 * e;
			      derp->p_filter[2] = 0;
			      derp->p_filter[3] = -norm;
			      derp->p_filter[4] = 0;

                  derp->n_filter[0] = -e2;
			      derp->n_filter[1] = 2 * e;
			      derp->n_filter[2] = 0;
			      derp->n_filter[3] = norm;
			      derp->n_filter[4] = 0;
			      break;

		   case 2 :
			    //norm = (1 - e)*(1 - e)*(1 - e)*(1 - e) / (1 + 2 * e - 2 * e * e2 - (e2)*(e2));
                  dividend= (1 - e) * (1 - e) * (1 - e) * (1 - e);
                  divisor=  (1 + 2 * e - 2 * e * e2 - (e2)*(e2));
			      norm =  (float) dividend/divisor;

                  derp->p_filter[0] = -e2;
			      derp->p_filter[1] = 2 * e;
			      derp->p_filter[2] = -norm;
			      derp->p_filter[3] = (float) norm * (1 + alpha * (1 - e2) / (2 * alpha * e)) * e;
			      derp->p_filter[4] = 0;

                  derp->n_filter[0] = -e2;
			      derp->n_filter[1] = 2 * e;
			      derp->n_filter[2] = 0;
			      derp->n_filter[3] = (float) -norm * (1 - alpha * (1 - e2) / (2 * alpha * e)) * e;
			      derp->n_filter[4] = norm * e2;
			      break;

           default :
            fprintf(stderr,"Unsupported order ( only signal (0) , first derivative [0] , second derivative [1] supported \n");
            return 0;
           break;
 };
return 1;
}




/*
    // - - - - - - - - - - - - - - - - - - - -








                 UCHAR VERSION
    // - - - - - - - - - - - - - - - - - - - -
*/




static inline int deriche1DPassF(
                   float * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                   float * target,  unsigned int targetWidth , unsigned int targetHeight ,
                   struct derichePrecalculations * derp ,
                   unsigned int x, unsigned int y,
                   unsigned int direction,
                   float sigma ,
                   unsigned int order
                 )
{
 unsigned int i=0;
 float * sourceLimit = 0;
 unsigned int offsetToNextPixel=0;
 unsigned int bufferSize=0;

//Directions are inverted ? ? ?? ?? ? ? ?
 //First we compute the number of pixels in the line
 if (direction==0)
    { //we perform a 1D pass on the X direction
      //x=0; //We start from X 0
      sourceLimit = source + ( sourceWidth * y ) + sourceWidth ; //Last Item should have coordinates (sourceWidth-1,y) , so the limit is (sourceWidth,y)
      offsetToNextPixel=1;    //The way the image is packed we go 1 array space to the right for every access
      bufferSize=sourceWidth; //The size of our wanted buffer is the same as the width of the image
    } else
 if (direction==1)
    { //we perform a 1D pass on the X direction
      //y=0; //We start from Y 0
      sourceLimit = source + ( sourceWidth * (sourceHeight) ) + x; //Last Item should have coordinates (x,sourceHeight-1) , so the limit is (y,sourceHeight)
      offsetToNextPixel=sourceWidth;     //The way the image is packed we go sourceWidth array space to go down for every access
      bufferSize=sourceHeight;           //The size of our wanted buffer is the same as the height of the image
    } else
    {
      fprintf(stderr,"Unknown direction for deriche1DPass\n");
      return 0;
    }

 //In each of the two cases we want to start at (x,y) position
 float * sourceStart = source + ( sourceWidth * y ) + x ;
 float * sourcePTR = sourceStart;

 //These could be moved outside to make the code go softer on syscalls
 float * y1 = ( float * ) malloc(sizeof(float) * bufferSize);
 if (y1==0) { fprintf(stderr,"Could not allocate memory for y1 deriche\n"); return 0; }
 for ( i=0; i<bufferSize; i++ ) { y1[i]=0.0; }

 float * y2 = ( float * ) malloc(sizeof(float) * bufferSize);
 if (y2==0) { fprintf(stderr,"Could not allocate memory for y2 deriche\n");  free(y1); return 0; }
 for ( i=0; i<bufferSize; i++ ) { y2[i]=0.0; }


float d2,d1,n0,n1,n2,in0,in1,in2,out0,out1,out2,norm;
// Computing constants that will be applied depending on the order we want


// We apply to the one dimensional direction of pixels

d2 = derp->p_filter[0];
d1 = derp->p_filter[1];
n0 = derp->p_filter[2];
n1 = derp->p_filter[3];
n2 = derp->p_filter[4];   // Note this is always == 0


sourcePTR = sourceStart;
in1 = (float) *sourcePTR;
in2 = in1;

out1 = (float) ((n2 + n1 + n0)*in1)/(1.0e+00-d1-d2);
out2 = out1;

for ( i=0; i<bufferSize; i++ )
{
	in0  =  (float) *sourcePTR;
	sourcePTR+=offsetToNextPixel;

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
d2 = derp->n_filter[0];
d1 = derp->n_filter[1];
n0 = derp->n_filter[2];   // Always == 0
n1 = derp->n_filter[3];
n2 = derp->n_filter[4];


sourcePTR = sourceLimit-offsetToNextPixel; //We want the last good item
in1 = (float) *sourcePTR;
in2 = in1;

out1 = (float) ((n2 + n1 + n0)*in1) / (1.0e+00 - d1 - d2);
out2 = out1;

for ( i=(bufferSize-1); i>0; i-- )
{
	  in0  =  (float) *sourcePTR;
	  sourcePTR-=offsetToNextPixel;

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
float * targetPTR = target + ( ( targetWidth * y ) + x );
//fprintf(stderr,"source %p , target %p , targetPTR %p  => t(%u,%u) of %u bufs %u \n",source,target,targetPTR,x,y,offsetToNextPixel,bufferSize);
for (i=0; i<bufferSize; i++)
{
  yOut = y1[i] + y2[i];
  *targetPTR = (float) yOut;
  //*targetPTR=254;
  targetPTR+=offsetToNextPixel;
}


 //---- End of procedure
 free(y1);
 free(y2);

 return 1;
}








int dericheRecursiveGaussianGrayF(
                                     float * target,  unsigned int targetWidth , unsigned int targetHeight , unsigned int channels,
                                     float * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                                     float * sigma , unsigned int order
                                   )
{
  if (channels!=1)
  {
     fprintf(stderr,"dericheRecursiveGaussianGray only accepts grayscale images..\n");
     return 0;
  }

  if (*sigma==0)
  {
    fprintf(stderr,"dericheRecursiveGaussianGrayF cannot work with zero sigma \n");
    return 0;
  }


  struct derichePrecalculations derp={0};
  if (! dericheDoPrecalculations( &derp , *sigma , order ) ) { return 0; }


  unsigned int x=0,y=0;

     y=0;
     for (x=0; x<sourceWidth; x++)
       {
           deriche1DPassF(
                          source,  sourceWidth , sourceHeight ,
                          target , targetWidth , targetHeight ,
                          &derp ,
                          x,y,  1, // X direction
                          *sigma , order
                         );
       }

     x=0;
     for (y=0; y<sourceHeight; y++)
       {
           deriche1DPassF(
                          target , targetWidth , targetHeight  ,
                          target , targetWidth , targetHeight ,
                          &derp ,
                          x,y, 0, // Y direction
                          *sigma , order
                         );
       }

 return 1;
}






/*
    // - - - - - - - - - - - - - - - - - - - -
                 FLOAT VERSION




By the way this is the first thing I made that could be
           also made using templates .. :P




                 UCHAR VERSION
    // - - - - - - - - - - - - - - - - - - - -
*/












static inline int deriche1DPass(
                   unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                   unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                   struct derichePrecalculations * derp ,
                   unsigned int x, unsigned int y,
                   unsigned int direction,
                   float sigma ,
                   unsigned int order
                 )
{
 unsigned int i=0;
 unsigned char * sourceLimit = 0;
 unsigned int offsetToNextPixel=0;
 unsigned int bufferSize=0;

//Directions are inverted ? ? ?? ?? ? ? ?
 //First we compute the number of pixels in the line
 if (direction==0)
    { //we perform a 1D pass on the X direction
      //x=0; //We start from X 0
      sourceLimit = source + ( sourceWidth * y ) + sourceWidth ; //Last Item should have coordinates (sourceWidth-1,y) , so the limit is (sourceWidth,y)
      offsetToNextPixel=1;    //The way the image is packed we go 1 array space to the right for every access
      bufferSize=sourceWidth; //The size of our wanted buffer is the same as the width of the image
    } else
 if (direction==1)
    { //we perform a 1D pass on the X direction
      //y=0; //We start from Y 0
      sourceLimit = source + ( sourceWidth * (sourceHeight) ) + x; //Last Item should have coordinates (x,sourceHeight-1) , so the limit is (y,sourceHeight)
      offsetToNextPixel=sourceWidth;     //The way the image is packed we go sourceWidth array space to go down for every access
      bufferSize=sourceHeight;           //The size of our wanted buffer is the same as the height of the image
    } else
    {
      fprintf(stderr,"Unknown direction for deriche1DPass\n");
      return 0;
    }

 //In each of the two cases we want to start at (x,y) position
 unsigned char * sourceStart = source + ( sourceWidth * y ) + x ;
 unsigned char * sourcePTR = sourceStart;

 //These could be moved outside to make the code go softer on syscalls
 float * y1 = ( float * ) malloc(sizeof(float) * bufferSize);
 if (y1==0) { fprintf(stderr,"Could not allocate memory for y1 deriche\n"); return 0; }
 for ( i=0; i<bufferSize; i++ ) { y1[i]=0.0; }

 float * y2 = ( float * ) malloc(sizeof(float) * bufferSize);
 if (y2==0) { fprintf(stderr,"Could not allocate memory for y2 deriche\n");  free(y1); return 0; }
 for ( i=0; i<bufferSize; i++ ) { y2[i]=0.0; }


 float d2,d1,n0,n1,n2,in0,in1,in2,out0,out1,out2,norm;


// We apply to the one dimensional direction of pixels

d2 = derp->p_filter[0];
d1 = derp->p_filter[1];
n0 = derp->p_filter[2];
n1 = derp->p_filter[3];
n2 = derp->p_filter[4];   // Note this is always == 0


sourcePTR = sourceStart;
in1 = (float) *sourcePTR;
in2 = in1;

out1 = (n2 + n1 + n0)*in1/(1.0e+00-d1-d2);
out2 = out1;

for ( i=0; i<bufferSize; i++ )
{
	in0  =  (float) *sourcePTR;
	sourcePTR+=offsetToNextPixel;

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
d2 = derp->n_filter[0];
d1 = derp->n_filter[1];
n0 = derp->n_filter[2];   // Always == 0
n1 = derp->n_filter[3];
n2 = derp->n_filter[4];


sourcePTR = sourceLimit-offsetToNextPixel; //We want the last good item
in1 = (float) *sourcePTR;
in2 = in1;

out1 = (n2 + n1 + n0)*in1 / (1.0e+00 - d1 - d2);
out2 = out1;

for ( i=(bufferSize-1); i>0; i-- )
{
	  in0  =  (float) *sourcePTR;
	  sourcePTR-=offsetToNextPixel;

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
unsigned char * targetPTR = target + ( ( targetWidth * y ) + x );
//fprintf(stderr,"source %p , target %p , targetPTR %p  => t(%u,%u) of %u bufs %u \n",source,target,targetPTR,x,y,offsetToNextPixel,bufferSize);
for (i=0; i<bufferSize; i++)
{
  yOut = y1[i] + y2[i];
  *targetPTR = (unsigned char) yOut;
  //*targetPTR=254;
  targetPTR+=offsetToNextPixel;
}


 //---- End of procedure
 free(y1);
 free(y2);

 return 1;
}



int dericheRecursiveGaussianGray(
                                  unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight , unsigned int channels,
                                  unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                                  float * sigma , unsigned int order
                                )
{
  if (channels!=1)
  {
     fprintf(stderr,"dericheRecursiveGaussianGray only accepts grayscale images..\n");
     return 0;
  }

  if (*sigma==0)
  {
    fprintf(stderr,"dericheRecursiveGaussianGray cannot work with zero sigma \n");
    return 0;
  }

  struct derichePrecalculations derp={0};
  if (! dericheDoPrecalculations( &derp , *sigma , order ) ) {  return 0; }

  unsigned int x=0,y=0;

     y=0;
     for (x=0; x<sourceWidth; x++)
       {
           deriche1DPass(
                          source,  sourceWidth , sourceHeight ,
                          target,  targetWidth , targetHeight ,
                          &derp ,
                          x,y,  1, // X direction
                          *sigma , order
                         );
       }

     x=0;
     for (y=0; y<sourceHeight; y++)
       {
           deriche1DPass(
                          target, targetWidth , targetHeight ,
                          target, targetWidth , targetHeight ,
                          &derp ,
                          x,y, 0, // Y direction
                          *sigma , order
                         );
       }

  return 1;
}

