#include "AcquisitionMux.h"
#include <stdio.h>
#include <stdlib.h>




static unsigned int simplePow(unsigned int base,unsigned int exp)
{
    if (exp==0) return 1;
    unsigned int retres=base;
    unsigned int i=0;
    for (i=0; i<exp-1; i++)
    {
        retres*=base;
    }
    return retres;
}

int saveMuxImageToFile(char * filename,char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{
    if(pixels==0) { fprintf(stderr,"saveRawImageToFile(%s) called for an unallocated (empty) frame , will not write any file output\n",filename); return 0; }
    FILE *fd=0;
    fd = fopen(filename,"wb");

    if (bitsperpixel>16) fprintf(stderr,"PNM does not support more than 2 bytes per pixel..!\n");
    if (fd!=0)
    {
        unsigned int n;
        if (channels==3) fprintf(fd, "P6\n");
        else if (channels==1) fprintf(fd, "P5\n");
        else
        {
            fprintf(stderr,"Invalid channels arg (%u) for SaveRawImageToFile\n",channels);
            return 1;
        }

        fprintf(fd, "%d %d\n%u\n", width, height , simplePow(2 ,bitsperpixel)-1);

        float tmp_n = (float) bitsperpixel/ 8;
        tmp_n = tmp_n *  width * height * channels ;
        n = (unsigned int) tmp_n;

        fwrite(pixels, 1 , n , fd);
        //fwrite(pixels, 1 , n , fd);
        fflush(fd);
        fclose(fd);
        return 1;
    }
    else
    {
        fprintf(stderr,"SaveRawImageToFile could not open output file %s\n",filename);
        return 0;
    }
    return 0;
}
















int mux2RGBAndDepthFramesNonZeroDepth( char * rgbBase, char * rgbOverlay , char * rgbOut , short * depthBase, short * depthOverlay , short * depthOut , unsigned int width , unsigned int height , unsigned int mux_type)
{
   char * rgb_pBase = rgbBase;
   char * rgb_pOverlay = rgbOverlay;
   char * rgb_pOut = rgbOut; char * rgb_pOut_limit=rgb_pOut + width * height * 3;

   short * depth_pBase = depthBase;
   short * depth_pOverlay = depthOverlay;
   short * depth_pOut = depthOut; short * depth_pOut_limit=rgb_pOut + width * height * 2;


   unsigned int TookBaseloops=0;
   unsigned int loops=0;
   while (rgb_pOut<rgb_pOut_limit)
    {
        //if ( (*rgb_pOverlay == 0) && (*rgb_pOverlay+1 == 0) && (*rgb_pOverlay+2 == 0) )
        if ( (depthOverlay[loops]==0  ) || (depthOverlay[loops]==255) )    /* || (loops>640*480/2)*/
         {
           //Overlay has a zero depth on this pixel! that means we will completely discard it and go along with our base
           *rgb_pOut = *rgb_pBase;  ++rgb_pOut; ++rgb_pBase;
           *rgb_pOut = *rgb_pBase;  ++rgb_pOut; ++rgb_pBase;
           *rgb_pOut = *rgb_pBase;  ++rgb_pOut; ++rgb_pBase;
            //RGB Overlay bytes are just ignored
            rgb_pOverlay+=3;


           *depth_pOut = *depth_pBase;
            ++depth_pOut; ++depth_pBase;
            //DEPTH overlay bytes are also just ignored
            ++depth_pOverlay;

            ++TookBaseloops;
         } else
         {
            //Overlay has a non zero value so we "augment it" ignoring Base
           *rgb_pOut = *rgb_pOverlay;  ++rgb_pOut; ++rgb_pOverlay;
           *rgb_pOut = *rgb_pOverlay;  ++rgb_pOut; ++rgb_pOverlay;
           *rgb_pOut = *rgb_pOverlay;  ++rgb_pOut; ++rgb_pOverlay;
            //RGB Base is just ignored
            rgb_pBase+=3;


           *depth_pOut = *depth_pBase + *depth_pOverlay;
            ++depth_pOut;  ++depth_pOverlay;
            //DEPTH base bytes are also just ignored
            ++depth_pBase;
         }
       ++loops;
    }

    fprintf(stderr,"Total of %u pixels ( base are %u , %0.2f %% ) \n",loops,TookBaseloops,TookBaseloops*100/loops);

    return 1;
}


int mux2RGBAndDepthFrames( char * rgbBase, char * rgbOverlay , char * rgbOut , short * depthBase, short * depthOverlay , short * depthOut , unsigned int width , unsigned int height , unsigned int mux_type)
{
 if (mux_type==0)
  {
    return mux2RGBAndDepthFramesNonZeroDepth(rgbBase,rgbOverlay,rgbOut,
                                             depthBase,depthOverlay,depthOut ,
                                             width,height,mux_type);
  }
  return 0;
}





