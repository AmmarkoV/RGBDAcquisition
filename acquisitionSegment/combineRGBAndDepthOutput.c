#include "combineRGBAndDepthOutput.h"

#include <stdio.h>
#include <stdlib.h>








int executeSegmentationRGB(unsigned char * RGB , unsigned char * selectedRGB , unsigned int width , unsigned int height ,  struct SegmentationFeaturesRGB * segConf )
{
  unsigned char * tmpRGB;
  unsigned char * ptrRGB = RGB;
  unsigned char * ptrRGBLimit = RGB + ( width * height * 3 );
  unsigned char * selectedPtr = selectedRGB;

  if (segConf->enableReplacingColors)
  { //We replace colors with something
    while (ptrRGB < ptrRGBLimit )
    {
      tmpRGB=ptrRGB;
      if ( *selectedPtr!=0 )
         {
            *tmpRGB=segConf->replaceR; ++tmpRGB;
            *tmpRGB=segConf->replaceG; ++tmpRGB;
            *tmpRGB=segConf->replaceB;
         } else
         {
            *tmpRGB=segConf->eraseColorR; ++tmpRGB;
            *tmpRGB=segConf->eraseColorG; ++tmpRGB;
            *tmpRGB=segConf->eraseColorB;
         }

      ++selectedPtr;
      ptrRGB+=3;
    }
  }  else
  {
    while (ptrRGB < ptrRGBLimit )
    {
      if ( *selectedPtr==0 )
         {
            tmpRGB=ptrRGB;
            *tmpRGB=segConf->eraseColorR; ++tmpRGB;
            *tmpRGB=segConf->eraseColorG; ++tmpRGB;
            *tmpRGB=segConf->eraseColorB;
         }

      ++selectedPtr;
      ptrRGB+=3;
    }
  }
  return 1;
}







int executeSegmentationDepth(unsigned short * Depth , unsigned char * selectedDepth , unsigned int width , unsigned int height )
{
  if (Depth==0) { fprintf(stderr,"Wrong Depth Array while @ executeSegmentationDepth\n"); return 0; }
  if (selectedDepth==0) { fprintf(stderr,"Wrong selectedDepth Array while @ executeSegmentationDepth\n"); return 0; }

  unsigned short * ptrDepth = Depth;
  unsigned short * ptrDepthLimit = Depth + ( width * height );
  unsigned char * selectedPtr = selectedDepth;


  while (ptrDepth < ptrDepthLimit )
    {
      if (*selectedPtr==0) { *ptrDepth = (unsigned short) 0 ; }
                           // else  { *ptrDepth = (unsigned short) 40000 ; }
      ++selectedPtr; ++ptrDepth;
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
