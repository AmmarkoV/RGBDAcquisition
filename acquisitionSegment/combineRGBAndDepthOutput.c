#include "combineRGBAndDepthOutput.h"

#include <stdio.h>
#include <stdlib.h>








int executeSegmentationRGB(unsigned char * RGB , unsigned char * selectedRGB , unsigned int width , unsigned int height ,  struct SegmentationFeaturesRGB * segConf )
{
  unsigned char * tmpRGB;
  unsigned char * ptrRGB = RGB;
  unsigned char * ptrRGBLimit = RGB + ( width * height * 3 );

  if (segConf->enableReplacingColors)
  { //We replace colors with something
    while (ptrRGB < ptrRGBLimit )
    {
      tmpRGB=ptrRGB;
      if ( *selectedRGB )
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

      ++selectedRGB;
      ptrRGB+=3;
    }
  }  else
  {
    while (ptrRGB < ptrRGBLimit )
    {
      if ( ! *selectedRGB )
         {
            tmpRGB=ptrRGB;
            *tmpRGB=segConf->eraseColorR; ++tmpRGB;
            *tmpRGB=segConf->eraseColorG; ++tmpRGB;
            *tmpRGB=segConf->eraseColorB;
         }

      ++selectedRGB;
      ptrRGB+=3;
    }
  }

}







int executeSegmentationDepth(unsigned short * Depth , unsigned char * selectedDepth , unsigned int width , unsigned int height ,  struct SegmentationFeaturesDepth * segConf )
{
  unsigned short * ptrDepth = Depth;
  unsigned short * ptrDepthLimit = Depth + ( width * height );


  while (ptrDepth < ptrDepthLimit )
    {
      if (!*selectedDepth) { *ptrDepth = (unsigned short) 0 ; }
      ++selectedDepth; ++ptrDepth;
    }
  return 1;
}











unsigned char * combineRGBAndDepthToOutput( unsigned char * selectedRGB , unsigned char * selectedDepth , int combinationMode, unsigned int width , unsigned int height )
{
  unsigned char * result = (unsigned char *)  malloc (width * height * sizeof(unsigned char) );
  if (result == 0 ) { fprintf(stderr,"Cannot allocate a result for combining RGB and Depth\n"); return 0; }
  memset(result,0,width*height*sizeof(unsigned char));

  unsigned char * ptrRGB = selectedRGB;
  unsigned char * ptrDepth = selectedDepth;


  unsigned char * ptrRGBLimit = selectedRGB + ( width * height );

  switch (combinationMode)
  {
    case COMBINE_AND :
      while (ptrRGB < ptrRGBLimit )
        {
          if ( (*ptrRGB)&&(*ptrDepth) ) {  *result=1;  }  ++ptrRGB; ++ptrDepth; ++result;
        }
    break;

    case COMBINE_OR :
      while (ptrRGB < ptrRGBLimit )
        {
          if ( (*ptrRGB)||(*ptrDepth) ) {  *result=1;  }  ++ptrRGB; ++ptrDepth; ++result;
        }
    break;

    case COMBINE_XOR :
      while (ptrRGB < ptrRGBLimit )
        {
          if ( (*ptrRGB)^(*ptrDepth) ) {  *result=1;  }  ++ptrRGB; ++ptrDepth; ++result;
        }
    break;


    case COMBINE_KEEP_ONLY_RGB :
       memcpy(result,ptrRGB,width*height*sizeof(unsigned char));
    break;

    case COMBINE_KEEP_ONLY_DEPTH :
       memcpy(result,ptrDepth,width*height*sizeof(unsigned char));
    break;

  };


  return 0;
}
