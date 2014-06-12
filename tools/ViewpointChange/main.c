#include <stdio.h>
#include <stdlib.h>
#include "ViewpointChange.h"
#include "viewpoint_change.h"

unsigned int viewPointChange_fitImageInMask(unsigned char * img, unsigned char * mask , unsigned int width , unsigned int height)
{
  if ( (img==0)||(mask==0) ) { fprintf(stderr,"Cannot FitImageInMask with empty Images\n"); return 0; }

  unsigned char * imgPtr = img;
  unsigned char * imgLimit = imgPtr + (width * height * 3);
  unsigned char * maskPtr = mask;

  unsigned int thisPixelCounts = 0;
  unsigned int count = 0;

  while (imgPtr < imgLimit)
  {
      if ((*maskPtr)!=0)
      {
        thisPixelCounts = 0;
        if (*imgPtr>0) { thisPixelCounts=1; } ++imgPtr;
        if (*imgPtr>0) { thisPixelCounts=1; } ++imgPtr;
        if (*imgPtr>0) { thisPixelCounts=1; } ++imgPtr;
        if (thisPixelCounts!=0) { ++count; }

      } else
      { imgPtr+=3; }

     maskPtr+=3;
  }

   return count ;
}



unsigned char* viewPointChange_mallocTransformToBirdEyeView(unsigned char *  rgb , unsigned short *  depth, unsigned int width , unsigned int height , unsigned int depthRange)
{
  return  birdsEyeView(rgb , depth , width, height , 0 , depthRange);
}


int viewPointChange_newFramesCompare(unsigned char *  prototype , unsigned char *  rgb , unsigned short *  depth, unsigned int width , unsigned int height  , unsigned int depthRange )
{
  unsigned char * bev = birdsEyeView(rgb , depth , width, height, 0, depthRange);
     if (bev!=0)
         {
           unsigned int fitScore = viewPointChange_fitImageInMask(bev,rgb,width,height);
           fprintf(stderr,"Got a fit of %u\n",fitScore);

           if (fitScore < 4000)
           {
             fprintf(stderr,"\n");
           }

           free(bev);
        } else
        { fprintf(stderr,"Could not perform a birdseyeview translation\n"); }

  return 1;
}
