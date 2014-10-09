#include "combineRGBAndDepthOutput.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define likely(x)    __builtin_expect (!!(x), 1)
#define unlikely(x)  __builtin_expect (!!(x), 0)





int executeSegmentationRGB(unsigned char * RGB , unsigned char * selectedRGB , unsigned int width , unsigned int height ,  struct SegmentationFeaturesRGB * segConf ,unsigned int selectedRGBCount )
{
  if (RGB==0) { fprintf(stderr,"Wrong RGB Array while @ executeSegmentationRGB\n"); return 0; }
  if (selectedRGB==0) { fprintf(stderr,"Wrong selectedRGB Array while @ executeSegmentationRGB\n"); return 0; }
  if (selectedRGBCount==width*height) { /*Immediate , selected all response , RGB buffer remains intact */ return 1; }

  unsigned char * ptrRGB = RGB;
  unsigned char * ptrRGBLimit = RGB + ( width * height * 3 );
  unsigned char * selectedPtr = selectedRGB;

  if (segConf->enableReplacingColors)
  { //We replace colors with something
    while (ptrRGB < ptrRGBLimit )
    {
      if ( *selectedPtr!=0 )
         {
            *ptrRGB=segConf->replaceR; ++ptrRGB;
            *ptrRGB=segConf->replaceG; ++ptrRGB;
            *ptrRGB=segConf->replaceB; ++ptrRGB;
         } else
         {
            *ptrRGB=segConf->eraseColorR; ++ptrRGB;
            *ptrRGB=segConf->eraseColorG; ++ptrRGB;
            *ptrRGB=segConf->eraseColorB; ++ptrRGB;
         }

      ++selectedPtr;
    }
  }  else
  {
    if (selectedRGBCount>width*height*2)
    { //TODO COPY RGBA BYTE HERE
      while (ptrRGB < ptrRGBLimit )
      {
        if (unlikely( *selectedPtr==0 ) )
         {
            *ptrRGB=segConf->eraseColorR; ++ptrRGB;
            *ptrRGB=segConf->eraseColorG; ++ptrRGB;
            *ptrRGB=segConf->eraseColorB; ++ptrRGB;
            ptrRGB-=3;
         }
       ++selectedPtr;
       ptrRGB+=3;
     }
    } else
    {
      while (ptrRGB < ptrRGBLimit )
      {
        if ( likely( *selectedPtr==0 ) )
         {
            *ptrRGB=segConf->eraseColorR; ++ptrRGB;
            *ptrRGB=segConf->eraseColorG; ++ptrRGB;
            *ptrRGB=segConf->eraseColorB; ++ptrRGB;
            ptrRGB-=3;
         }

       ++selectedPtr;
       ptrRGB+=3;
     }
    }

  }
  return 1;
}







int executeSegmentationDepth(unsigned short * Depth , unsigned char * selectedDepth , unsigned int width , unsigned int height ,unsigned int selectedDepthCount)
{
  if (Depth==0) { fprintf(stderr,"Wrong Depth Array while @ executeSegmentationDepth\n"); return 0; }
  if (selectedDepth==0) { fprintf(stderr,"Wrong selectedDepth Array while @ executeSegmentationDepth\n"); return 0; }
  if (selectedDepthCount==width*height) { /*Immediate , selected all response , Depth buffer remains intact */ return 1; }

  unsigned short * ptrDepth = Depth;
  unsigned short * ptrDepthLimit = Depth + ( width * height );
  unsigned char * selectedPtr = selectedDepth;


  if (selectedDepthCount>width*height*2)
  { //We have more than half of the pixels selected so we assume selection is the default
    while (ptrDepth < ptrDepthLimit )
      {
      //Erase non-selected depth pixels
        if (unlikely(*selectedPtr==0))
                 { *ptrDepth = (unsigned short) 0 ; }

       ++selectedPtr;
       ++ptrDepth;
      }
  } else
  {
    while (ptrDepth < ptrDepthLimit )
      {
      //Erase non-selected depth pixels
        if (likely(*selectedPtr==0))
                 { *ptrDepth = (unsigned short) 0 ; }

       ++selectedPtr;
       ++ptrDepth;
      }
  }

  return 1;
}











unsigned char * combineRGBAndDepthToOutput( unsigned char * selectedRGB , unsigned char * selectedDepth , int combinationMode, unsigned int width , unsigned int height )
{
  unsigned char * result = (unsigned char *)  malloc (width * height * sizeof(unsigned char) );
  if (result == 0 ) { fprintf(stderr,"Cannot allocate a result for combining RGB and Depth\n"); return 0; }
  memset(result,0,width*height*sizeof(unsigned char));



  unsigned char * ptrResult = result;
  unsigned char * ptrRGB = selectedRGB;
  unsigned char * ptrDepth = selectedDepth;


  unsigned char * ptrRGBLimit = selectedRGB + ( width * height );

  switch (combinationMode)
  {
    case COMBINE_AND :
      while (ptrRGB < ptrRGBLimit )
        {
          if ( (*ptrRGB!=0)&&(*ptrDepth!=0) ) {  *ptrResult=1;  }
          ++ptrRGB; ++ptrDepth; ++ptrResult;
        }
    break;

    case COMBINE_OR :
      while (ptrRGB < ptrRGBLimit )
        {
          if ( (*ptrRGB!=0)||(*ptrDepth!=0) ) {  *ptrResult=1;  }
          ++ptrRGB; ++ptrDepth; ++ptrResult;
        }
    break;

    case COMBINE_XOR :
      while (ptrRGB < ptrRGBLimit )
        {
          if ( (*ptrRGB!=0)^(*ptrDepth!=0) ) {  *ptrResult=1;  }
          ++ptrRGB; ++ptrDepth; ++ptrResult;
        }
    break;


    case COMBINE_KEEP_ONLY_RGB :
       memcpy(result,ptrRGB,width*height*sizeof(unsigned char));
    break;

    case COMBINE_KEEP_ONLY_DEPTH :
       memcpy(result,ptrDepth,width*height*sizeof(unsigned char));
    break;

  };


  return result;
}
