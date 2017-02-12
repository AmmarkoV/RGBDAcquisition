#include "imageFilters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>




float * allocateGaussianKernel(unsigned int dimension,float sigma , int normalize)
{
 float * gK = (float * ) malloc(sizeof(float) * dimension * dimension );
 if (gK ==0 )  { return 0; }
 float * gKPTR;

  float sum = 0;
  float tmp;// , w = 0 ;
  unsigned int x0 = (unsigned int) dimension/2,y0 = (unsigned int) dimension/2;
  unsigned int x,y;
  float sigmaSquared = sigma * sigma;

  for (y=0; y<dimension; y++)
  {
   for (x=0; x<dimension; x++)
   {
     gKPTR = gK + ( dimension * y ) + x ;

     tmp =  (x-x0)*(x-x0);
     tmp += (y-y0)*(y-y0);
     tmp = (float) -1 * tmp / sigmaSquared;
     *gKPTR =  exp( tmp ) / 2 * M_PI * sigmaSquared;
      sum += *gKPTR;
   }
  }


 //Normalize and print..
 fprintf(stderr,"Created a gaussian filter %ux%u \n",dimension,dimension);
  if (sum>0.0005)
  {
   for (y=0; y<dimension; y++)
   {
    for (x=0; x<dimension; x++)
    {
     gKPTR = gK + ( dimension * y ) + x ;
     if (normalize)
     {
       *gKPTR =  *gKPTR  / sum ;
     }
     fprintf(stderr,"%03f  ",*gKPTR);
    }
    fprintf(stderr,"\n");
   }
  }

 return gK;
}




int monochrome(struct Image * img)
{
 if (img==0) { fprintf(stderr,"Function Monochrome given Null Image\n"); return 0; }
 if (img->pixels==0) { fprintf(stderr,"Function Monochrome given Null Image\n"); return 0; }
 if (img->channels==1) {  fprintf(stderr,"Image is already monochrome..!\n"); return 1; }
 if (img->channels!=3) { fprintf(stderr,"Function Monochrome assumes 3byte array\n"); return 0; }


 BYTE * input_frame = img->pixels;
 unsigned int col_med;
 unsigned int image_size= img->width * img->height * img->channels * (img->bitsperpixel/8);

 register BYTE *out_px = (BYTE *) input_frame;
 register BYTE *end_px = (BYTE *) input_frame + image_size;
 register BYTE *px = (BYTE *) input_frame;
 register BYTE *r;
 register BYTE *g;
 register BYTE *b;

 while ( px < end_px)
 {
       r = px++; g = px++; b = px++;
       col_med= ( *r + *g + *b )/3;
       *out_px= (BYTE)col_med ;
       ++out_px;
 }
 img->channels = 1;
 return 1;
}



int contrast(struct Image * img,float scaleValue)
{
 if (img==0) { fprintf(stderr,"Function contrast given Null Image\n"); return 0; }
 if (img->pixels==0) { fprintf(stderr,"Function contrast given Null memory buffer for Image\n"); return 0; }
 if (img->channels!=3) { fprintf(stderr,"Function contrast assumes 3byte array\n"); return 0; }

 BYTE * input_frame = img->pixels;
 //unsigned int col_med;
 unsigned int image_size= img->width * img->height * img->channels * (img->bitsperpixel/8);

 register BYTE *out_px = (BYTE *) input_frame;
 register BYTE *end_px = (BYTE *) input_frame + image_size;
 register BYTE *px = (BYTE *) input_frame;
 register BYTE *r;
 register BYTE *g;
 register BYTE *b;

 unsigned int rScaled,gScaled,bScaled;

 while ( px < end_px)
 {
       r = px++; g = px++; b = px++;

       rScaled = *r;
       rScaled=(unsigned int) (rScaled * scaleValue);
       gScaled = *g;
       gScaled=(unsigned int) (gScaled * scaleValue);
       bScaled = *b;
       bScaled=(unsigned int) (bScaled * scaleValue);

       if (rScaled>255) { *r=255; } else { *r = (unsigned char) rScaled; }
       if (gScaled>255) { *g=255; } else { *g = (unsigned char) gScaled; }
       if (bScaled>255) { *b=255; } else { *b = (unsigned char) bScaled; }
 }
 return 1;
}
