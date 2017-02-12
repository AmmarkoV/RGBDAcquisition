#include "resize.h"
#include <stdio.h>
#include <stdlib.h>


unsigned char * upscaleRGBImage(
                               unsigned char * input ,
                               unsigned int originalWidth , unsigned int originalHeight ,
                               unsigned int newWidth , unsigned int newHeight ,
                               unsigned int quality
                               )
{
  fprintf(stderr,"upscaleRGBImage() is a stub , not implemented yet \n");
  return 0;
}


unsigned char * downscaleRGBImagePure(
                                  unsigned char * input ,
                                  unsigned int originalWidth , unsigned int originalHeight ,
                                  unsigned int newWidth , unsigned int newHeight ,
                                  unsigned int quality
                                 )
{
  //float widthRatioF=originalWidth/newWidth;
  //float heightRatioF=originalHeight/newHeight;

  unsigned char * output = (unsigned char * ) malloc(sizeof(unsigned char) * 3 * newWidth * newHeight );
  if (output==0) { fprintf(stderr,"Could not allocate an image for resizing ( %u,%u ) => ( %u,%u)\n",originalWidth,originalHeight,newWidth , newHeight); return 0; }


 //TODO :

  return output;
}



unsigned char * resizeRGBImage(
                               unsigned char * input ,
                               unsigned int originalWidth , unsigned int originalHeight ,
                               unsigned int newWidth , unsigned int newHeight ,
                               unsigned int quality
                               )
{
  //Validate our input
  if (
       (originalWidth==0) ||
       (originalHeight==0) ||
       (newWidth==0) ||
       (newHeight==0)
      ) { fprintf(stderr,"Resizing Null dimensions does not make sense\n"); return 0; }


  //In case of upscaling there is a different code path to follow
  if (
       (originalWidth<newWidth) ||
       (originalHeight<newHeight)
      )
  {
    return upscaleRGBImage(input,originalWidth,originalHeight,newWidth,newHeight,quality);
  }





  return downscaleRGBImagePure(input,originalWidth,originalHeight,newWidth,newHeight,quality);
}
