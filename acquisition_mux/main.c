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
















int mux2RGBAndDepthFramesNonZeroDepth( char * rgb1, char * rgb2 , char * rgbOut , short * depth1, short * depth2 , short * depthOut , unsigned int width , unsigned int height , unsigned int mux_type)
{
   char * rgb_p1 = rgb1;  char * rgb_p1_limit=rgb1 + width * height * 3;
   char * rgb_p2 = rgb2;  char * rgb_p2_limit=rgb2 + width * height * 3;
   char * rgb_pOut = rgbOut; char * rgb_pOut_limit=rgb_pOut + width * height * 3;

   short * depth_p1 = depth1;     short * depth_p1_limit=rgb1 + width * height * 2;
   short * depth_p2 = depth2;     short * depth_p2_limit=rgb2 + width * height * 2;
   short * depth_pOut = depthOut; short * depth_pOut_limit=rgb_pOut + width * height * 2;

   while (rgb_p1<rgb_p1_limit)
    {
        if (*depth_p2==0)
         {
           //IF DEPTH2 is not set we just ignore it an make a direct copy of the pixel data from RGB1/DEPTH1
           *rgb_pOut = *rgb_p1;
            rgb_pOut++; rgb_p1++;
           *rgb_pOut = *rgb_p1;
            rgb_pOut++; rgb_p1++;
           *rgb_pOut = *rgb_p1;
            rgb_pOut++; rgb_p1++;

            //RGB2 is just ignored
            rgb_p2+=3;


           *depth_pOut = *depth1;
            depth_pOut++; depth1++;


            //DEPTH2 is just ignored
            ++depth2;
         } else
         {
            //if depth 2 is not null we keep it as an overlay on RGB1 and depth1
           *rgb_pOut = *rgb_p2;
            rgb_pOut++; rgb_p2++;
           *rgb_pOut = *rgb_p2;
            rgb_pOut++; rgb_p2++;
           *rgb_pOut = *rgb_p2;
            rgb_pOut++; rgb_p2++;

            //RGB1 is just ignored
            rgb_p1+=3;


           *depth_pOut = *depth1;
           *depth_pOut += *depth2;
            depth_pOut++; depth1++;  ++depth2;
         }

    }

    return 1;
}


int mux2RGBAndDepthFrames( char * rgb1, char * rgb2 , char * rgbOut , short * depth1, short * depth2 , short * depthOut , unsigned int width , unsigned int height , unsigned int mux_type)
{
 if (mux_type==0)
  {
    return mux2RGBAndDepthFramesNonZeroDepth(rgb1,rgb2,rgbOut, depth1,depth2,depthOut , width,height,mux_type);
  }
  return 0;
}





