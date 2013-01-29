#include "AcquisitionMux.h"


char * mux2RGBAndDepthFrames( char * rgb1, char * rgb2 , char * rgbOut , short * depth1, short * depth2 , short * depthOut , unsigned int width , unsigned int height , unsigned int mux_type)
{
   char * rgb_p1 = rgb1;  char * rgb_p1_limit=rgb1 + width * height * 3;
   char * rgb_p2 = rgb2;  char * rgb_p2_limit=rgb2 + width * height * 3;
   char * rgb_pOut = rgbOut; char * rgb_pOut_limit=rgb_pOut + width * height * 3;

   char * depth_p1 = depth1;  char * depth_p1_limit=rgb1 + width * height * 2;
   char * depth_p2 = depth2;  char * depth_p2_limit=rgb2 + width * height * 2;
   char * depth_pOut = depthOut; char * depth_pOut_limit=rgb_pOut + width * height * 2;

   while (rgb_p1<rgb_p1_limit)
    {

        //if ()


    }





}


