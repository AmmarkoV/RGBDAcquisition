#include "viewpoint_change.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



//These are some calculations that are very frequently used
//They are defined here as macros to make the code easier to read..
#define MEMPLACE3PLACE(x,y,width) ( y * ( width * 3 ) + x*3 )
#define MEMPLACE1PLACE(x,y,width) ( y * ( width ) + x )
#define RGB(r,g,b) B + (G * 256) + (R * 65536) )
#define ABSDIFF(num1,num2) ( (num1-num2) >=0 ? (num1-num2) : (num2 - num1) )



unsigned int FitImageInMask(struct Image * img, struct Image * mask)
{
  if ( (img==0)||(mask==0) ) { fprintf(stderr,"Cannot FitImageInMask with empty Images\n"); return 0; }
  if ( (img->pixels==0)||(mask->pixels==0) ) { fprintf(stderr,"Cannot FitImageInMask with empty frames\n"); return 0; }

  char * imgPtr = img->pixels;
  char * imgLimit = imgPtr + (img->width * img->height * 3);
  char * maskPtr = mask->pixels;

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





unsigned char * birdsEyeView(unsigned char * rgb,unsigned short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth)
{
  if (rgb==0)  { fprintf(stderr,"RGB is not allocated , cannot perform birdsEyeView transformation \n"); return 0; }
  if (depth==0)  { fprintf(stderr,"Depth is not allocated , cannot perform birdsEyeView transformation \n"); return 0; }
  unsigned char * rgbPTR= rgb; unsigned char * rgbLimit = rgb + width*height*3;
  unsigned short * depthPTR= depth; //unsigned short * depthLimit = depth + width*height;


  unsigned char * birdEyeFrame = (unsigned char*) malloc(width*height*3*sizeof(unsigned char));
  if (birdEyeFrame==0) { fprintf(stderr,"Could not perform birdsEyeView transformation\nNo memory for new frame\n"); return 0; }
  memset(birdEyeFrame, 0 , width*height*3);

  unsigned int range_misses=0;
  unsigned int x = 0;
  unsigned int y = 0;
  unsigned int z = 0;

  unsigned int depth_range = max_depth-min_depth;
  if (depth_range == 0 ) { depth_range=1; }
  float multiplier = (float) (height-1) / depth_range;

  while ( rgbPTR<rgbLimit )
  {
     if (*depthPTR!=0)
      {
         //depth will become height , so we must scale min_depth -> max_depth  to a range from 0 -> height
         z= (unsigned int) ((*depthPTR) * multiplier) ;

         if (z<height)
          {
           unsigned char * source = rgbPTR;
           unsigned char * target = birdEyeFrame + MEMPLACE3PLACE(x,z,width);

            *target = *source; ++target; ++source;
            *target = *source; ++target; ++source;
            *target = *source;
          } else
          {  ++range_misses; }
      }

     if (x==width-1)
                    { if (y==height) {  fprintf(stderr,"Error while counting");  } else { ++y; }
                      x=0;
                    } else
                    { ++x; }

     rgbPTR+= 3;
     depthPTR+=1;
  }

  if (range_misses>0)
   {
       fprintf(stderr,"Configuration of depth range %u->%u resulted in %u depth range misses\n",min_depth,max_depth,range_misses);
   }

  return birdEyeFrame;
}



















unsigned char * birdsEyeViewBack(unsigned char * rgb,unsigned short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth)
{
  if (rgb==0)  { fprintf(stderr,"RGB is not allocated , cannot perform birdsEyeView transformation \n"); return 0; }
  if (depth==0)  { fprintf(stderr,"Depth is not allocated , cannot perform birdsEyeView transformation \n"); return 0; }
  unsigned char * rgbPTR= rgb; unsigned char * rgbLimit = rgb + width*height*3;
  unsigned short * depthPTR= depth; unsigned short * depthLimit = depth + width*height;


  unsigned char * birdEyeFrame = (unsigned char*) malloc(width*height*3*sizeof(unsigned char));
  if (birdEyeFrame==0) { fprintf(stderr,"Could not perform birdsEyeView transformation\nNo memory for new frame\n"); return 0; }
  memset(birdEyeFrame, 0 , width*height*3);

  unsigned int range_misses=0;
  unsigned int x = width-1;
  unsigned int y = height-1;
  unsigned int z = 0;

  unsigned int depth_range = max_depth-min_depth;
  if (depth_range == 0 ) { depth_range=1; }
  float multiplier = (float) (height-1) / depth_range;

  rgbPTR = rgbLimit - 3;
  depthPTR = depthLimit - 1;

  unsigned char * rgbStart = rgb;
  //Only do half of the screen (height / 2)
  rgbLimit = rgb + (width*(height/2)*3);

  //We go backways
  while ( rgbPTR>rgbStart )
  {
     if (*depthPTR!=0)
      {
         //depth will become height , so we must scale min_depth -> max_depth  to a range from 0 -> height
         z= (unsigned int) ((*depthPTR) * multiplier) ;

         if (z<height)
          {
           unsigned char * source = rgbPTR;
           unsigned char * target = birdEyeFrame + MEMPLACE3PLACE(x,z,width);

            *target = *source; ++target; ++source;
            *target = *source; ++target; ++source;
            *target = *source;
          }
           else
          {  ++range_misses; }
      }

     if (x==0)  {
                  if (y==0) {  fprintf(stderr,"Error while counting");  } else { --y; }
                  x=width-1;
                } else
                { --x; }

     rgbPTR-= 3 ;
     depthPTR-=1;
  }

  if (range_misses>0)
   {
       fprintf(stderr,"Configuration of depth range %u->%u resulted in %u depth range misses\n",min_depth,max_depth,range_misses);
   }

  return birdEyeFrame;
}



unsigned short * getVolumesBirdsEyeView(unsigned short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth)
{
  if (depth==0)  { fprintf(stderr,"Depth is not allocated , cannot perform birdsEyeView transformation \n"); return 0; }
  unsigned short * depthPTR= depth; unsigned short * depthLimit = depth + width*height;


  unsigned short * birdEyeFrame = (unsigned short*) malloc(width*height*3*sizeof(unsigned char));
  if (birdEyeFrame==0) { fprintf(stderr,"Could not perform birdsEyeView transformation\nNo memory for new frame\n"); return 0; }
  memset(birdEyeFrame, 0 , width*height*3);

  unsigned int range_misses=0;
  unsigned int x = 0;
  unsigned int y = 0;
  unsigned int z = 0;

  unsigned int depth_range = max_depth-min_depth;
  if (depth_range == 0 ) { depth_range=1; }
  float multiplier = (float) (height-1) / depth_range;


  while ( depthPTR<depthLimit )
  {
     if (*depthPTR!=0)
      {
         //depth will become height , so we must scale min_depth -> max_depth  to a range from 0 -> height
         z= (unsigned int) ((*depthPTR) * multiplier) ;

         if (z<height)
          {
           unsigned short * target = birdEyeFrame + MEMPLACE1PLACE(x,z,width);

            *target = *target +1 ;
          }
           else
          {  ++range_misses; }
      }

     ++x;
     if (x==width)  {
                      x=0;
                      ++y;
                      if (y>=height) {  fprintf(stderr,"Error while counting");  }
                    }
     ++depthPTR;
  }

  if (range_misses>0)
   {
       fprintf(stderr,"Configuration of depth range %u->%u resulted in %u depth range misses\n",min_depth,max_depth,range_misses);
   }

  return birdEyeFrame;
}















