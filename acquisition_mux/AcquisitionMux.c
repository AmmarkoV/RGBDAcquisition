#include "AcquisitionMux.h"
#include "../tools/ImageOperations/imageOps.h"

#include <stdio.h>
#include <stdlib.h>


#if __GNUC__
#define likely(x)    __builtin_expect (!!(x), 1)
#define unlikely(x)  __builtin_expect (!!(x), 0)
#else
 #define likely(x)   x
 #define unlikely(x)   x
#endif

#define CUT_NUMBERS 10000


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

int saveMuxImageToFile(char * filename,unsigned char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel)
{
    char filenameFull[2048]={0};
    sprintf(filenameFull,"%s.pnm",filename);

    if(pixels==0) { fprintf(stderr,"saveMuxImageToFile(%s) called for an unallocated (empty) frame , will not write any file output\n",filename); return 0; }
    if (bitsperpixel>16) { fprintf(stderr,"PNM does not support more than 2 bytes per pixel..!\n"); return 0; }

    FILE *fd=0;
    fd = fopen(filenameFull,"wb");

    if (fd!=0)
    {
        unsigned int n;
        if (channels==3) fprintf(fd, "P6\n");
        else if (channels==1) fprintf(fd, "P5\n");
        else
        {
            fprintf(stderr,"Invalid channels arg (%u) for SaveRawImageToFile\n",channels);
            fclose(fd);
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


int mux2RGBAndDepthFramesColorNonTrans( unsigned char * rgbBase, unsigned char * rgbOverlay , unsigned char * rgbOut ,
                                   unsigned short * depthBase, unsigned short * depthOverlay , unsigned short * depthOut ,
                                   unsigned char transR, unsigned char transG, unsigned char transB , unsigned char transThreshold,
                                   signed int shiftX,signed int shiftY,
                                   unsigned int width , unsigned int height , unsigned int rgbTransparency , unsigned int mux_type)
{

   if ( (shiftX!=0) || (shiftY!=0) )
   {
     shiftImageRGB(rgbOverlay,rgbOverlay,transR,transG,transB,shiftX,shiftY,width,height);
     #warning "Shifting Depth Images is disabled since it seems to segfault ( probably a bad implementation on bitBltDepth ?"
     //shiftImageDepth(depthOverlay,depthOverlay,0,shiftX,shiftY,width,height);
   }


   unsigned char * rgb_pBase = rgbBase;
   unsigned char * rgb_pOverlay = rgbOverlay;
   unsigned char * rgb_pOut = rgbOut; unsigned char * rgb_pOut_limit=rgb_pOut + width * height * 3;

   unsigned short * depth_pBase = depthBase;
   //unsigned short * depth_pOverlay = depthOverlay;
   unsigned short * depth_pOut = depthOut; unsigned short * depth_pOut_limit=rgb_pOut + width * height * 2;

   unsigned int TookBaseloops=0;
   unsigned int loops=0;

   if (rgbTransparency>100) { rgbTransparency=100; }
   float transparencyOverlayFactor = (float) (100-rgbTransparency) / 100;
   float transparencyBaseFactor = (float)  rgbTransparency / 100;

   unsigned char *rOverlay;
   unsigned char *gOverlay;
   unsigned char *bOverlay;

   unsigned char minTransR=transR,minTransG=transG,minTransB=transB;
   unsigned char maxTransR=transR,maxTransG=transG,maxTransB=transB;
   if (transThreshold!=0)
        {
           if (transR>transThreshold) { minTransR=transR-transThreshold; } else { minTransR=0; }
           if (transG>transThreshold) { minTransG=transG-transThreshold; } else { minTransG=0; }
           if (transB>transThreshold) { minTransB=transB-transThreshold; } else { minTransG=0; }

           if (255-transR>transThreshold) { maxTransR=transR+transThreshold; } else { maxTransR=255; }
           if (255-transG>transThreshold) { maxTransG=transG+transThreshold; } else { maxTransG=255; }
           if (255-transB>transThreshold) { maxTransB=transB+transThreshold; } else { maxTransB=255; }
        }

   #warning "MUXer needs work on performance.."
   while (rgb_pOut<rgb_pOut_limit)
    {
        rOverlay = rgb_pOverlay++;
        gOverlay = rgb_pOverlay++;
        bOverlay = rgb_pOverlay++;

        int isTransparent=0;
        if (transThreshold==0)
        {
        if ( (transR==*rOverlay) && (transG==*gOverlay) && (transB==*bOverlay)  )
             {
               isTransparent=1;
             }
        } else
        {
          if (
               ( (minTransR<=*rOverlay) && (*rOverlay<=maxTransR) ) &&
               ( (minTransG<=*gOverlay) && (*gOverlay<=maxTransG) ) &&
               ( (minTransB<=*bOverlay) && (*bOverlay<=maxTransB) )
             )
             {
               isTransparent=1;
             }
        }

        //If overlay has transparent color it means we only keep our base rgb/depth values
        if (isTransparent)
        {
           //Just Copy RGB Value
           *rgb_pOut = *rgb_pBase;  ++rgb_pOut; ++rgb_pBase;
           *rgb_pOut = *rgb_pBase;  ++rgb_pOut; ++rgb_pBase;
           *rgb_pOut = *rgb_pBase;  ++rgb_pOut; ++rgb_pBase;
            //Just Copy Depth Value
           *depth_pOut = *depth_pBase; ++depth_pOut; ++depth_pBase;
            ++TookBaseloops;
        } else
       {
           if (rgbTransparency!=0)
           {
           unsigned int rValue =  (*rOverlay * transparencyOverlayFactor) +  (*rgb_pBase * transparencyBaseFactor );  rgb_pBase++;
           unsigned int gValue =  (*gOverlay * transparencyOverlayFactor) +  (*rgb_pBase * transparencyBaseFactor );  rgb_pBase++;
           unsigned int bValue =  (*bOverlay * transparencyOverlayFactor) +  (*rgb_pBase * transparencyBaseFactor );  rgb_pBase++;

           unsigned char rValueCast = 0 + (unsigned char) rValue;
           unsigned char gValueCast = 0 + (unsigned char) gValue;
           unsigned char bValueCast = 0 + (unsigned char) bValue;

           *rgb_pOut = (unsigned char) rValueCast;  ++rgb_pOut;
           *rgb_pOut = (unsigned char) gValueCast;  ++rgb_pOut;
           *rgb_pOut = (unsigned char) bValueCast;  ++rgb_pOut;
           } else
          {
            //Overlay has a non zero value so we "augment it" ignoring Base
           *rgb_pOut = *rOverlay;  ++rgb_pOut;
           *rgb_pOut = *gOverlay;  ++rgb_pOut;
           *rgb_pOut = *bOverlay;  ++rgb_pOut;
            //RGB Base is just ignored
            rgb_pBase+=3;
          }
           //Color Muxing , ignore depth , always keep base
           *depth_pOut = *depth_pBase; /*+ *depth_pOverlay;*/ ++depth_pOut;  ++depth_pBase; //depthBase is ignored
         }

        ++loops;
    }


    fprintf(stderr,"Total of %u pixels ( base are %u , %0.2f %% ) \n",loops,TookBaseloops,(double) TookBaseloops*100/loops);

    return 1;
}



int mux2RGBAndDepthFramesBasedOnDepth( unsigned char * rgbBase, unsigned char * rgbOverlay , unsigned char * rgbOut ,
                                       unsigned short * depthBase, unsigned short * depthOverlay , unsigned short * depthOut ,
                                       unsigned char transR, unsigned char transG, unsigned char transB ,unsigned char transThreshold,
                                       signed int shiftX,signed int shiftY,
                                       unsigned int width , unsigned int height , unsigned int rgbTransparency , unsigned int mux_type)
{

   if (depthOverlay==0) { fprintf(stderr,"mux2RGBAndDepthFramesBasedOnDepth cannot continue with a NULL depth overlay\n"); return 0; }

   if ( (shiftX!=0) || (shiftY!=0) )
   {
     shiftImageRGB(rgbOverlay,rgbOverlay,transR,transG,transB,shiftX,shiftY,width,height);
     #warning "Shifting Depth Images is disabled since it seems to segfault ( probably a bad implementation on bitBltDepth ?"
     shiftImageDepth(depthOverlay,depthOverlay,0,shiftX,shiftY,width,height);
   }

   unsigned char * rgb_pBase = rgbBase;
   unsigned char * rgb_pOverlay = rgbOverlay;
   unsigned char * rgb_pOut = rgbOut; unsigned char * rgb_pOut_limit=rgb_pOut + width * height * 3;

   unsigned short * depth_pBase = depthBase;
   unsigned short * depth_pOverlay = depthOverlay;
   unsigned short * depth_pOut = depthOut; unsigned short * depth_pOut_limit=rgb_pOut + width * height * 2;

   float transparencyOverlayFactor = (float) (100-rgbTransparency) / 100;
   float transparencyBaseFactor = (float)  rgbTransparency / 100;

   unsigned char *rOverlay;
   unsigned char *gOverlay;
   unsigned char *bOverlay;


   while (rgb_pOut<rgb_pOut_limit)
    {
        rOverlay = rgb_pOverlay++;
        gOverlay = rgb_pOverlay++;
        bOverlay = rgb_pOverlay++;

        //If overlay has no depth it means we only keep our base rgb/depth values
        if ( ( *depth_pOverlay==0  ) || (*depth_pOverlay>*depth_pBase) )
        {
           //Just Copy Base (Original) RGB Value
           *rgb_pOut = *rgb_pBase;  ++rgb_pOut; ++rgb_pBase;
           *rgb_pOut = *rgb_pBase;  ++rgb_pOut; ++rgb_pBase;
           *rgb_pOut = *rgb_pBase;  ++rgb_pOut; ++rgb_pBase;
            //Just Copy Base (Original) Depth Value
           *depth_pOut = *depth_pBase;
        } else
       {
           //Overlay has a non zero value so we "augment it" ignoring Base
           *rgb_pOut = *rOverlay;  ++rgb_pOut;
           *rgb_pOut = *gOverlay;  ++rgb_pOut;
           *rgb_pOut = *bOverlay;  ++rgb_pOut;
           //RGB Base is just ignored
           rgb_pBase+=3;
           *depth_pOut = *depth_pOverlay;
         }

        ++depth_pOut;
        ++depth_pBase; //depthBase is ignored
        ++depth_pOverlay; // <- Depth Overlay is ignored
    }

    return 1;
}


int mux2RGBAndDepthFrames(
                           unsigned char * rgbBase, unsigned char * rgbOverlay , unsigned char * rgbOut ,
                           unsigned short * depthBase, unsigned short * depthOverlay , unsigned short * depthOut ,
                           unsigned char transR, unsigned char transG, unsigned char transB ,unsigned char transThreshold,
                           signed int shiftX,signed int shiftY,
                           unsigned int width , unsigned int height , unsigned int rgbTransparency ,
                           unsigned int mux_type
                         )
{
 switch (mux_type)
 {
     case COLOR_MUXING :
              return mux2RGBAndDepthFramesColorNonTrans(rgbBase,rgbOverlay,rgbOut,
                                                   depthBase,depthOverlay,depthOut ,
                                                   transR,transG,transB,transThreshold,
                                                   shiftX,shiftY,
                                                   width,height,rgbTransparency,mux_type);
              break;

     case DEPTH_MUXING :
               return mux2RGBAndDepthFramesBasedOnDepth(rgbBase,rgbOverlay,rgbOut,
                                                        depthBase,depthOverlay,depthOut ,
                                                        transR,transG,transB,transThreshold,
                                                        shiftX,shiftY,
                                                        width,height,rgbTransparency,mux_type);
               break;
 }

  fprintf(stderr,"Unhandled mux2RGBAndDepthFrames muxType ( %u ) \n",mux_type);
  return 0;
}



int generateInterpolatedFrames(
                               unsigned char * firstRGB, unsigned char *  secondRGB , unsigned char * intermediateRGBOut ,
                               unsigned short * firstDepth, unsigned short *  secondDepth  , unsigned short *  intermediateDepthOut  ,
                               unsigned int width ,
                               unsigned int height
                               )
{
   if ( (firstRGB==0)||(secondRGB==0) ) { fprintf(stderr,"generateInterpolatedFrames( input frames are empty )\n"); return 0; }
   if ( intermediateRGBOut==0)   { fprintf(stderr,"generateInterpolatedFrames( output frames is not allocated)\n"); return 0; }

   unsigned char * ptrA = firstRGB;
   unsigned char * ptrB = secondRGB;
   unsigned char * ptrOut = intermediateRGBOut;
   unsigned char * ptrOutLimit = intermediateRGBOut + width*height*3;

   unsigned int avg;

   while (ptrOut<ptrOutLimit)
   {
     avg=(unsigned int) ( *ptrA + *ptrB ) / 2 ;
     *ptrOut = (unsigned char) avg;
     //----------
     ++ptrA; ++ptrB; ++ptrOut;
   }


   if ( (firstDepth==0)||(secondDepth==0) ) { fprintf(stderr,"generateInterpolatedFrames( input frames are empty )\n"); return 0; }
   if ( intermediateDepthOut==0)   { fprintf(stderr,"generateInterpolatedFrames( output frames is not allocated)\n"); return 0; }


 return 1;
}

int LongExposureFramesCollect( unsigned char * rgb , unsigned long * rgbCollector ,
                               unsigned short * depth, unsigned long * depthCollector ,
                               unsigned int width , unsigned int height , unsigned int * framesCollected)
{
    unsigned long * rgbCollectorPTR = rgbCollector , rgbOutLimit = rgbCollector + width * height *3 ;
    unsigned long * depthCollectorPTR = depthCollector , depthOutLimit = depthCollector + width * height ;

    unsigned char * rgbPTR = rgb;
    unsigned short * depthPTR = depth;


    while ( rgbCollectorPTR < rgbOutLimit )     { *rgbCollectorPTR += *rgbPTR; ++rgbCollectorPTR; ++rgbPTR; }
    while ( depthCollectorPTR < depthOutLimit ) { *depthCollectorPTR += *depthPTR; ++depthCollectorPTR; ++depthPTR; }

    //*framesCollected+=1;
  if (CUT_NUMBERS!=0)
  {
    if (*framesCollected%CUT_NUMBERS==0)
    {
        rgbCollectorPTR = rgbCollector;
        depthCollectorPTR = depthCollector;
        while ( rgbCollectorPTR < rgbOutLimit )
        {
            *rgbCollectorPTR=*rgbCollectorPTR/CUT_NUMBERS;
            ++rgbCollectorPTR;
        }

        while ( depthCollectorPTR < depthOutLimit )
        {
            *depthCollectorPTR=*depthCollectorPTR/CUT_NUMBERS;
            ++depthCollectorPTR;
        }
    }
  }


    return 1;
}

int LongExposureFramesFinalize(unsigned long * rgbCollector ,  unsigned char * rgbOut ,
                               unsigned long * depthCollector ,  unsigned short * depthOut,
                               unsigned int width , unsigned int height , unsigned int * framesCollected)
{

    unsigned long * rgbCollectorPTR = rgbCollector , rgbOutLimit = rgbCollector + width * height *3 ;
    unsigned long * depthCollectorPTR = depthCollector , depthOutLimit = depthCollector + width * height ;



    unsigned char * rgbPTR = rgbOut;
    unsigned short * depthPTR = depthOut;

    unsigned int what2DivideWith = *framesCollected%CUT_NUMBERS;
    if (what2DivideWith==0) {  what2DivideWith=1; }


     if (rgbCollector!=0)
     {
      while ( rgbCollectorPTR < rgbOutLimit )     { *rgbPTR=*rgbCollectorPTR/what2DivideWith;     ++rgbCollectorPTR;   ++rgbPTR; }
     }

     if (depthCollector!=0)
     {
      while ( depthCollectorPTR < depthOutLimit ) { *depthPTR=*depthCollectorPTR/what2DivideWith; ++depthCollectorPTR; ++depthPTR; }
     }


   return 1;
}

