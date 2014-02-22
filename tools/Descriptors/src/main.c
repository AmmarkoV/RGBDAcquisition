#include <stdio.h>
#include <stdlib.h>

struct descriptorContext * descriptorCreate(unsigned char * rgb , unsigned int rgbWidth ,unsigned int rgbHeight ,
                                            unsigned short * depth  , unsigned int depthWidth , unsigned int depthHeight )
{




    //width = rgbWidth;
    //height = rgbHeight;

    return 0;
}


unsigned char * descriptorVisualizeRGB(struct descriptorContext * desc, unsigned int * width, unsigned int * height)
{


  return 0;
}


// A function doing nothing ;)
int descriptorDestroy(struct descriptorContext *  dest)
{
  if (dest!=0) { free(dest); }
  return 1;
}
